"""第 1 批代码的本地验证（不依赖 torch / faiss / pymupdf）。

运行：
    python tests/test_batch1.py

用 conda 的 base 环境即可（只需 numpy）。

验证范围
--------
1. 文本规范化：中文 PDF 常见的「字间空格」修复、英文不被误伤
2. token 估算：中英比例合理性
3. 切分：块长不超预算、不丢内容、中英语料粒度一致
4. 语料加载：abstracts.jsonl 与 queries.txt 解析正确
5. 检索链路：VectorStore 建库 / 排序 / 兜底 backend，用**确定性哈希编码器**
   代替真实嵌入模型（仅为验证管道，不代表真实语义检索效果）

为什么检索召回只"报告"不"断言"：stub 编码器只有字面重合能力，无法跨语言。
真实召回要用多语嵌入模型，在云端验证（见 docs/experiment.md 的 E3）。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunker import (  # noqa: E402
    chunk_text,
    estimate_tokens,
    merge_cjk_spaces,
    normalize_text,
    split_sentences,
)
from config import (  # noqa: E402
    ABSTRACTS_PATH,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    QUERIES_PATH,
    TOPK,
)
from corpus import build_chunks, load_abstracts, load_queries, parse_paper_id  # noqa: E402
from vector_store import VectorStore, l2_normalize  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  -> {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ── 1. 文本规范化 ───────────────────────────────────────────────────
def test_normalize() -> None:
    section("1. 文本规范化")

    spaced = "检 索 增 强 生 成 (RAG) 是 一 种 方 法"
    got = normalize_text(spaced)
    check("中文字间空格被合并", got == "检索增强生成(RAG)是一种方法", got)

    en = "We use RAG (retrieval) in the model works"
    check("英文空格不被误伤", merge_cjk_spaces(en) == en, merge_cjk_spaces(en))

    # 换行/制表符先压成单个空格，随后中文字间空格又被合并 —— 两步叠加是预期行为
    check("多行被压平", normalize_text("第一行\n第二行\t第三行") == "第一行第二行第三行")
    check("空输入不报错", normalize_text("") == "")

    # 真实语料归一化后不应再出现「中 中」式空格
    corpus_text = " ".join(p["abstract"] for p in load_abstracts(ABSTRACTS_PATH))
    check("真实语料规范化幂等", normalize_text(normalize_text(corpus_text)) == normalize_text(corpus_text))


# ── 2. token 估算 ───────────────────────────────────────────────────
def test_estimate() -> None:
    section("2. token 估算")
    check("纯中文 4 字 = 4 token", estimate_tokens("你好世界") == 4, estimate_tokens("你好世界"))
    check("纯英文 8 字符 = 2 token", estimate_tokens("abcdefgh") == 2, estimate_tokens("abcdefgh"))
    check("空串 = 0", estimate_tokens("") == 0)
    mix = "检索增强 generation"
    check("中英混合估算单调", estimate_tokens(mix) >= 4, estimate_tokens(mix))


# ── 3. 断句与切分 ───────────────────────────────────────────────────
def test_chunking() -> None:
    section("3. 断句与切分")
    sents = split_sentences("第一句。第二句！第三句？")
    check("中文断句为 3 句", len(sents) == 3, sents)
    sents_en = split_sentences("First one. Second one! Third one?")
    check("英文断句为 3 句", len(sents_en) == 3, sents_en)
    sents_num = split_sentences("ACC平均提升3.81%,ROUGE-L提升3.28%")
    check("小数不被误切", len(sents_num) == 1, sents_num)

    papers = load_abstracts(ABSTRACTS_PATH)
    chunks = build_chunks(
        papers,
        target_tokens=CHUNK_TARGET_TOKENS,
        overlap_tokens=CHUNK_OVERLAP_TOKENS,
    )
    print(f"  切分结果：{len(papers)} 篇 -> {len(chunks)} 个 chunk")
    print(f"  {'paper':6s} {'lang':4s} {'chars':>6s} {'chunks':>7s} {'max_tok':>8s}")
    by_paper: dict[str, list[dict]] = {}
    for c in chunks:
        by_paper.setdefault(c["paper_id"], []).append(c)
    for p in papers:
        cs = by_paper.get(p["paper_id"], [])
        mx = max((c["tokens_est"] for c in cs), default=0)
        print(
            f"  {p['paper_id']:6s} {p['lang']:4s} "
            f"{len(p['abstract']):6d} {len(cs):7d} {mx:8d}"
        )

    check("每篇至少 1 个 chunk", all(len(by_paper.get(p["paper_id"], [])) >= 1 for p in papers))
    over = [
        c["chunk_id"]
        for c in chunks
        if c["tokens_est"] > CHUNK_TARGET_TOKENS * 1.05
    ]
    check("无 chunk 超 token 预算 5%", not over, f"超预算: {over}")

    # 不丢内容：原文的每一句都应至少完整出现在某个 chunk 里
    # （块间的空格属于分隔符、两边都不含，因此不能直接比较字符总数；
    #   有重叠时字符总数又会偏大，只有"句子覆盖"是准确的判据）
    lost: list[str] = []
    for p in papers:
        cs = by_paper[p["paper_id"]]
        for sent in split_sentences(normalize_text(p["abstract"])):
            if not any(sent in c["text"] for c in cs):
                lost.append(f"{p['paper_id']}:{sent[:24]}")
                break
    check("切分未丢失任何句子", not lost, f"丢失: {lost[:3]}")

    # 中英粒度一致性：同一 token 预算下，中英文单块字符数比例应接近 1:4
    def mean_chars(lang: str) -> float:
        vals = [len(c["text"]) for c in chunks if c["lang"] == lang]
        return sum(vals) / len(vals) if vals else 0.0

    zh_mean, en_mean = mean_chars("zh"), mean_chars("en")
    ratio = en_mean / zh_mean if zh_mean else 0
    print(f"  平均字符数：zh={zh_mean:.0f}  en={en_mean:.0f}  en/zh={ratio:.1f}")
    check("中英块字符比在 2.0~5.0 之间（说明按 token 而非字符切分）", 2.0 <= ratio <= 5.0, f"{ratio:.2f}")


# ── 4. 语料加载 ─────────────────────────────────────────────────────
def test_loaders() -> None:
    section("4. 语料加载")
    papers = load_abstracts(ABSTRACTS_PATH)
    check("读到 10 篇论文", len(papers) == 10, len(papers))
    check("paper_id 无重复", len({p["paper_id"] for p in papers}) == 10)
    check("中英各 5 篇", sum(p["lang"] == "zh" for p in papers) == 5)
    check("摘要非空", all(p["abstract"].strip() for p in papers))

    queries = load_queries(QUERIES_PATH)
    ids = [q["id"] for q in queries]
    print(f"  解析到 {len(queries)} 条问题：{ids}")
    check("解析到 7 条问题", len(queries) == 7, len(queries))
    check("问题正文非空", all(q.get("question") for q in queries))
    check("expect_topk 已转成整数", all(isinstance(q.get("expect_topk"), int) for q in queries if q.get("expect_topk")))
    q6 = next(q for q in queries if q["id"] == "Q06")
    check("无答案问题的 expect_doc 为 None", q6.get("expect_doc") is None, q6.get("expect_doc"))

    check("从文件名解析 paper_id", parse_paper_id("P07_ccl2025-x.pdf") == "P07")
    check("非约定命名回退为文件名", parse_paper_id("random.pdf") == "random")


# ── 5. 检索链路（stub 编码器）────────────────────────────────────────
def hashing_encode(texts: list[str], dim: int = 512) -> np.ndarray:
    """确定性字符 n-gram 哈希编码器，仅用于验证管道，不代表真实语义能力。"""
    rows = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        chars = list(text)
        for n in (1, 2):
            for i in range(len(chars) - n + 1):
                gram = "".join(chars[i : i + n])
                if not gram.strip():
                    continue
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
                vec[h % dim] += 1.0
        rows.append(vec)
    return l2_normalize(np.vstack(rows))


def test_vector_store() -> None:
    section("5. 检索链路（stub 编码器）")
    papers = load_abstracts(ABSTRACTS_PATH)
    chunks = build_chunks(papers)
    queries = load_queries(QUERIES_PATH)

    chunk_vecs = hashing_encode([c["text"] for c in chunks])
    store = VectorStore(chunk_vecs)
    print(f"  {store.describe()}")
    check("faiss 缺失时自动回退 numpy", store.backend in ("faiss", "numpy"), store.backend)
    check("索引条数 = chunk 数", store.size == len(chunks))

    # 自检索：每个 chunk 的第一名应是自己，分数 ≈ 1.0
    scores, idx = store.search(chunk_vecs[0], k=1)
    check("自检索分数 ≈ 1.0", abs(float(scores[0]) - 1.0) < 1e-3, float(scores[0]))
    check("自检索命中自身", int(idx[0]) == 0, int(idx[0]))

    # 排序单调性与 topk 边界
    s, i = store.search(chunk_vecs[0], k=TOPK)
    check("返回 topk 个结果", len(s) == min(TOPK, len(chunks)), len(s))
    check("分数按降序排列", all(s[j] >= s[j + 1] for j in range(len(s) - 1)), s.tolist())
    check("相似度在 [-1, 1] 内", bool(np.all(s <= 1.0001) and np.all(s >= -1.0001)))

    # 注入真实 tokens 计数器时接口不变（模拟 tokenizer 可用的情况）
    chunks_tok = build_chunks(papers, count_tokens=lambda t: max(1, estimate_tokens(t) // 2))
    check("可注入自定义 token 计数器", len(chunks_tok) > 0, len(chunks_tok))

    # 持久化往返
    tmp = ROOT / "data" / "_tmp_index.npz"
    store.save_vectors(tmp)
    restored = VectorStore.from_file(tmp)
    s2, _ = restored.search(chunk_vecs[0], k=1)
    check("向量持久化往返一致", abs(float(s2[0]) - float(scores[0])) < 1e-5, float(s2[0]))
    tmp.unlink(missing_ok=True)

    # 召回报告（仅同语种问题；跨语种需要真实多语模型）
    print("\n  检索召回（stub 编码器，仅供参考）")
    print(f"  {'qid':5s} {'lang':5s} {'expect':8s} {'hit@3':6s} top3")
    qvecs = hashing_encode([q["question"] for q in queries])
    for qi, q in enumerate(queries):
        s, i = store.search(qvecs[qi], k=TOPK)
        top = [chunks[j]["paper_id"] for j in i]
        expect = q.get("expect_doc")
        if expect is None:
            hit = "-"
        else:
            hit = "YES" if expect in top else "no"
        print(f"  {q['id']:5s} {str(q.get('lang')):5s} {str(expect):8s} {hit:6s} {top}")


# ── main ────────────────────────────────────────────────────────────
def main() -> int:
    print("第 1 批本地验证（无 torch / faiss 依赖）")
    print(f"python={sys.version.split()[0]}  numpy={np.__version__}")
    test_normalize()
    test_estimate()
    test_chunking()
    test_loaders()
    test_vector_store()

    print("\n" + "=" * 60)
    if FAILS:
        print(f"FAILED: {len(FAILS)} 项")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
