"""阶段 A/B 验证：全文分节、论文库注册表、多论文多样化检索。

运行：
    python tests/test_phase_a.py

全部只依赖 numpy / 标准库，不需要 pymupdf、torch 或真实 PDF ——
用人工构造的"三页论文"覆盖页眉页脚、arXiv 水印、页码、参考文献等噪声场景。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from corpus import build_chunks, load_abstracts  # noqa: E402
from embedder import hashing_encode  # noqa: E402
from fulltext import (  # noqa: E402
    clean_pages,
    chunks_from_sections,
    detect_lang,
    is_heading,
    parse_paper,
    split_sections,
)
from library import Library  # noqa: E402
from retrieval import (  # noqa: E402
    coverage,
    format_evidence_table,
    group_by_paper,
    retrieve,
)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ── 构造的英文三页论文（含各类噪声）──────────────────────────────────
EN_PAGES = [
    """A Test Paper on Retrieval Augmented Generation
arXiv:2401.01234v2 [cs.CL] 12 Jan 2024
John Doe, Jane Roe
Abstract
We study retrieval augmented generation for long technical documents. It shows that full text retrieval outperforms abstract only retrieval on our benchmark.
1 Introduction
Retrieval augmented generation is widely used in question answering systems. However most systems only index abstracts, which loses detailed experimental evidence.
1
Proceedings of TestConf 2024""",
    """2 Method
We chunk documents by section instead of fixed length. Each chunk keeps its section name and page number so that answers can point to exact locations in the paper.
Figure 1: Architecture overview
Proceedings of TestConf 2024""",
    """3 Experiments
We evaluate on two datasets and report accuracy and latency. Full text retrieval improves accuracy by twelve points over the abstract baseline.
4 Conclusion
Section aware retrieval is effective for multi paper synthesis and evidence tracing.
References
[1] Some Other Paper. In Proceedings of Nothing, 2023.
[2] Another Paper. 2022.
Proceedings of TestConf 2024""",
]

ZH_PAGES = [
    """基于检索增强生成的测试论文
摘要
本文研究面向长篇技术文档的全文检索增强生成。实验表明全文检索显著优于仅使用摘要的检索方案。
1 引言
检索增强生成已广泛用于开放域问答系统，但多数系统只对摘要建立索引。
关键词：检索增强 全文检索 证据溯源
参考文献
[1] 某某. 某学报, 2023.""",
]


# ── 1. 噪声清洗 ─────────────────────────────────────────────────────
def test_clean() -> None:
    section("1. PDF 噪声清洗")
    blocks = clean_pages(EN_PAGES)
    texts = [b.text for b in blocks]
    joined = "\n".join(texts)

    check("arXiv 水印被清除", "arXiv:2401.01234" not in joined)
    check("页眉页脚被清除", "Proceedings of TestConf 2024" not in joined)
    check("纯页码行被清除", "1" not in [t.strip() for t in texts])
    check("正文未受影响", any("full text retrieval outperforms" in t for t in texts))
    check("页码信息被保留", blocks[0].page == 1 and blocks[-1].page == 3, (blocks[0].page, blocks[-1].page))


# ── 2. 标题识别 ─────────────────────────────────────────────────────
def test_heading() -> None:
    section("2. 章节标题识别")
    positives = ["1 Introduction", "2 Method", "3.2 Retrieval Module", "4 Conclusion",
                 "Abstract", "Related Work", "实验结果", "结论"]
    negatives = [
        "We study retrieval augmented generation for long technical documents.",
        "Experiments show that our method improves accuracy by twelve points.",
        "Figure 1: Architecture overview",
        "关键词：检索增强 全文检索 证据溯源",
    ]
    for line in positives:
        check(f"识别为标题: {line!r}", is_heading(line))
    for line in negatives:
        check(f"不识别为标题: {line[:32]!r}", not is_heading(line))


# ── 3. 分节与参考文献裁剪 ───────────────────────────────────────────
def test_sections() -> None:
    section("3. 分节与参考文献裁剪")
    sections = split_sections(EN_PAGES)
    names = [s.name for s in sections]
    print(f"  识别到 {len(sections)} 个章节：{names}")

    for expect in ("Abstract", "1 Introduction", "2 Method", "3 Experiments", "4 Conclusion"):
        check(f"章节已识别: {expect}", expect in names)

    full = "\n".join(s.text for s in sections)
    check("参考文献被裁掉", "Some Other Paper" not in full)
    check("裁掉参考文献后仍保留结论段", "Section aware retrieval is effective" in full)

    # 关闭裁剪时应保留参考文献
    sections_raw = split_sections(EN_PAGES, cut_references=False)
    raw_full = "\n".join(s.text for s in sections_raw)
    check("可关闭裁剪以保留参考文献", "Some Other Paper" in raw_full)

    # 每章要有起止页
    pages = [s.page_start for s in sections]
    check("章节记录起始页", all(isinstance(p, int) and p >= 1 for p in pages), pages)

    zh_sections = split_sections(ZH_PAGES)
    zh_names = [s.name for s in zh_sections]
    print(f"  中文论文识别到章节：{zh_names}")
    check("中文摘要被识别", any("摘" in n and "要" in n for n in zh_names), zh_names)
    check("中文参考文献被裁掉", "某学报" not in "\n".join(s.text for s in zh_sections))


# ── 4. 带溯源的切块 ─────────────────────────────────────────────────
def test_chunks() -> None:
    section("4. 带溯源的切块")
    result = parse_paper(EN_PAGES, paper_id="P99", title="", target_tokens=60, overlap_tokens=12)
    chunks = result["chunks"]
    print(f"  生成 {len(chunks)} 个 chunk，章节 {len(result['sections'])} 个")
    check("产出了多个 chunk", len(chunks) >= 4, len(chunks))
    check("每个 chunk 都有章节名", all(c.get("section") for c in chunks))
    check("每个 chunk 都有页码", all(isinstance(c.get("page"), int) for c in chunks))
    check("chunk_id 含章节与序号", all("#s" in c["chunk_id"] and "c" in c["chunk_id"] for c in chunks))
    check("chunk_id 全局唯一", len({c["chunk_id"] for c in chunks}) == len(chunks))
    check("chunk 不超预算", max(c["tokens_est"] for c in chunks) <= 60 * 1.05)
    check("标题自动从首页提取", result["title"].startswith("A Test Paper"), result["title"])

    zh = parse_paper(ZH_PAGES, paper_id="P98", target_tokens=200)
    check("中文论文语种判定", zh["lang"] == "zh", zh["lang"])
    check("英文论文语种判定", result["lang"] == "en", result["lang"])
    check("detect_lang 纯中文", detect_lang("这是一段纯中文的测试文本内容") == "zh")
    check("detect_lang 中文占比低判为 mixed", detect_lang("中文 English 中文 mixed") == "mixed")
    check("detect_lang 英文", detect_lang("This is a plain English sentence.") == "en")


# ── 5. 论文库注册表 ─────────────────────────────────────────────────
def test_library() -> None:
    section("5. 论文库注册表")
    lib_root = ROOT / "results" / "_tmp_lib"
    shutil.rmtree(lib_root, ignore_errors=True)
    try:
        lib = Library(lib_root)
        check("初始为空", len(lib) == 0)

        lib.add("P01", {"title": "Paper One", "lang": "en", "source_url": "https://x/1"})
        lib.add("P06", {"title": "论文六", "lang": "zh", "source_url": "https://x/6"})
        lib.save()

        reloaded = Library(lib_root)
        check("注册表持久化", len(reloaded) == 2, len(reloaded))
        check("记录字段完整", reloaded.get("P01")["title"] == "Paper One")
        check("added_at 自动填写", bool(reloaded.get("P01").get("added_at")))

        path, sha1 = reloaded.save_pdf("P01", b"%PDF-1.4 fake")
        check("PDF 落盘并计算 sha1", path.exists() and len(sha1) == 40, sha1)
        check("未生成产物时 is_complete=False", not reloaded.is_complete("P01"))

        chunks = [{"chunk_id": "P01#s0c0", "paper_id": "P01", "text": "hello"}]
        reloaded.save_chunks("P01", chunks)
        reloaded.save_sections("P01", [{"index": 0, "name": "body"}])
        reloaded.save_fulltext("P01", "hello world")
        check("产物齐全时 is_complete=True", reloaded.is_complete("P01"))
        check("chunk 往返一致", reloaded.load_chunks("P01") == chunks)
        check("sections 往返一致", reloaded.load_sections("P01")[0]["name"] == "body")

        st = reloaded.stats()
        check("统计中英论文数", st["n_papers"] == 2 and st["n_zh"] == 1 and st["n_en"] == 1, st)

        check("删除成功", reloaded.remove("P01"))
        check("删除后目录被清理", not reloaded.paper_dir("P01").exists())
        check("删除后注册表更新", "P01" not in reloaded)
        check("删除不存在的 ID 返回 False", reloaded.remove("PX") is False)
    finally:
        shutil.rmtree(lib_root, ignore_errors=True)


# ── 6. 多论文多样化检索 ─────────────────────────────────────────────
def test_retrieval() -> None:
    section("6. 多论文多样化检索")
    papers = load_abstracts(ROOT / "data" / "abstracts.jsonl")
    chunks = build_chunks(papers)
    vectors = hashing_encode([c["text"] for c in chunks])

    query = "固定文本长度分块会给检索增强问答系统带来什么问题？"
    qvec = hashing_encode([query])

    hits = retrieve(chunks, vectors, qvec, k=5, per_paper_cap=2, min_papers=2)
    cov = coverage(hits)
    print(f"  命中 {cov['n_hits']} 条，覆盖 {cov['n_papers']} 篇：{cov['per_paper']}")

    check("返回条数符合 k", len(hits) == 5, len(hits))
    check("覆盖至少 2 篇论文（多篇连结成立）", cov["n_papers"] >= 2, cov)
    check("单篇占比不超过 per_paper_cap", max(cov["per_paper"].values()) <= 2, cov["per_paper"])
    check("rank 从 1 连续编号", [h["rank"] for h in hits] == list(range(1, len(hits) + 1)))
    check("按相似度降序", all(hits[i]["score"] >= hits[i + 1]["score"] for i in range(len(hits) - 1)))
    check("每条都带溯源字段",
          all(h.get("paper_id") and h.get("section") is not None and "page" in h for h in hits))

    grouped = group_by_paper(hits)
    check("分组结果与命中数一致", sum(len(v) for v in grouped.values()) == len(hits))

    table = format_evidence_table(hits)
    check("证据表含表头", "相似度" in table)
    check("证据表含论文编号", all(h["paper_id"] in table for h in hits))

    # min_papers=1 时更集中，>=2 时更分散：验证覆盖策略确实起作用
    hits1 = retrieve(chunks, vectors, qvec, k=5, per_paper_cap=5, min_papers=1)
    check("min_papers 参数影响覆盖", coverage(hits1)["n_papers"] <= cov["n_papers"],
          (coverage(hits1)["n_papers"], cov["n_papers"]))

    check("空向量输入返回空列表", retrieve(chunks, np.zeros((0, 8), dtype=np.float32), qvec) == [])
    check("k 超过库容量时自动收敛", len(retrieve(chunks, vectors, qvec, k=999)) == len(chunks))


# ── main ────────────────────────────────────────────────────────────
def main() -> int:
    print("阶段 A/B 验证：全文分节 + 论文库 + 多样化检索")
    print(f"python={sys.version.split()[0]}  numpy={np.__version__}")
    test_clean()
    test_heading()
    test_sections()
    test_chunks()
    test_library()
    test_retrieval()

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
