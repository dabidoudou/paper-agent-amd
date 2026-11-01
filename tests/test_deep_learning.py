"""torch 路径验证：本地构造极小模型，验证 embedder / generator 的真实代码路径。

为什么本地构造模型而不是下载
----------------------------
1. **不需要网络**。HuggingFace 在部分环境不可达，测试就会变成"跳过"，等于没测。
   用 `BertConfig` / `Qwen2Config` 现场随机初始化一个 2 层小模型并 `save_pretrained`，
   整个验证完全离线、确定性、秒级完成。
2. **覆盖的代码路径完全一样**。要验证的是 mean-pooling 的 mask 广播、
   `torch_dtype`/`dtype` 参数名、chat template、`TextIteratorStreamer` 流式、TTFT 计时 ——
   这些都与模型大小和权重数值无关，只与张量形状和 API 有关。
3. **设备无关**。CPU 上通过即说明逻辑正确，GPU 只影响速度。

缺少 torch / transformers 时整文件自动跳过（退出码 0）。

运行：
    python tests/test_deep_learning.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FAILS: list[str] = []
TINY_DIR = ROOT / "results" / "_tiny_models"

# 所有会喂给 tokenizer 的文本，vocab 从这里构造，保证不会全部塌缩成 UNK
TEXTS = [
    "检索增强生成是一种方法",
    "固定长度分块会丢失段落结构",
    "检索增强生成在长篇技术文档上表现更好",
    "固定分块有什么问题？",
    "retrieval augmented generation",
    "passage: retrieval augmented generation works",
    "query: 固定分块有什么问题",
]
HIT_TEXTS = [
    "固定文本长度分块会丢失段落层级结构，导致输出质量下降。",
    "Performance degrades when evidence sits in the middle.",
]

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}<|im_start|>assistant\n"
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ── 本地构造 tiny 模型 ──────────────────────────────────────────────
def build_tiny_models(base_dir: Path) -> tuple[Path, Path]:
    """构造 tiny BERT（嵌入）与 tiny Qwen2（生成），返回两个目录。"""
    from transformers import BertConfig, BertModel, BertTokenizer
    from transformers import Qwen2Config, Qwen2ForCausalLM

    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    corpus = "".join(TEXTS + HIT_TEXTS)
    chars = sorted({c.lower() for c in corpus if not c.isspace()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars
    vocab_file = base_dir / "vocab.txt"
    vocab_file.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizer(str(vocab_file), do_lower_case=True)

    # ① tiny BERT —— 验证 mean-pooling
    embed_dir = base_dir / "bert-tiny"
    BertModel(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=64,
        )
    ).save_pretrained(embed_dir)
    tokenizer.save_pretrained(embed_dir)

    # ② tiny Qwen2 —— 验证 chat template + 流式生成
    gen_dir = base_dir / "qwen2-tiny"
    Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(vocab),
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=64,
            max_position_embeddings=128,
            eos_token_id=3,
        )
    ).save_pretrained(gen_dir)
    tokenizer.save_pretrained(gen_dir)
    # 注入 chat template，让 render() 走真正的模板分支
    cfg_path = gen_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["chat_template"] = CHAT_TEMPLATE
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    return embed_dir, gen_dir


# ── 1. BertEmbedder ─────────────────────────────────────────────────
def test_embedder(embed_dir: Path) -> None:
    section("1. BertEmbedder（手写 mean-pooling）")
    from embedder import BertEmbedder

    emb = BertEmbedder(model_name=str(embed_dir), device="cpu", dtype="float32")
    try:
        info = emb.load()
    except Exception as exc:
        check("load() 可执行", False, f"{type(exc).__name__}: {str(exc)[:300]}")
        return

    dim = info.get("embed_dim")
    print(f"  dim={dim}  device={info.get('embed_device')}  dtype={info.get('embed_dtype')}")
    check("load() 返回维度", (dim or 0) > 0, info)

    try:
        vecs = emb.encode(TEXTS[:3], kind="passage")
    except Exception as exc:
        check("encode() 可执行", False, f"{type(exc).__name__}: {str(exc)[:300]}")
        return

    check("输出形状 [n, dim]", vecs.shape == (3, dim), f"{vecs.shape} vs (3, {dim})")
    check("dtype 为 float32", str(vecs.dtype) == "float32", str(vecs.dtype))
    check("无 NaN/Inf", bool(np.all(np.isfinite(vecs))))
    norms = np.linalg.norm(vecs, axis=1)
    check("每行已 L2 归一化", bool(np.allclose(norms, 1.0, atol=1e-4)), norms.tolist())

    # 关键断言：mean-pooling 必须按 attention_mask 平均，padding 不能污染向量
    single = emb.encode([TEXTS[0]], kind="passage")
    diff = float(np.abs(single[0] - vecs[0]).max())
    check("单条与批量结果一致（padding 未污染）", diff < 1e-4, f"max diff {diff:.2e}")

    check("不同文本向量不同", not bool(np.allclose(vecs[0], vecs[1], atol=1e-3)))

    try:
        qv = emb.encode([TEXTS[3]], kind="query")
        check("query 前缀分支可用", qv.shape == (1, dim), qv.shape)
    except Exception as exc:
        check("query 前缀分支可用", False, f"{type(exc).__name__}: {exc}")

    try:
        from vector_store import VectorStore

        store = VectorStore(vecs)
        scores, _ = store.search(vecs[0], k=2)
        check("接入 VectorStore 可检索", len(scores) == 2, len(scores))
        check("自检索分数 ≈ 1.0", abs(float(scores[0]) - 1.0) < 1e-3, float(scores[0]))
        print(f"  索引：{store.describe()}")
    except Exception as exc:
        check("接入 VectorStore 可检索", False, f"{type(exc).__name__}: {exc}")


# ── 2. QwenGenerator ────────────────────────────────────────────────
def test_generator(gen_dir: Path) -> None:
    section("2. QwenGenerator（chat template + 流式生成）")
    from generator import QwenGenerator, build_citations, build_messages, fallback_prompt

    gen = QwenGenerator(
        model_name=str(gen_dir), device="cpu", dtype="float32", max_new_tokens=8
    )
    try:
        info = gen.load()
    except Exception as exc:
        check("load() 可执行", False, f"{type(exc).__name__}: {str(exc)[:300]}")
        return

    print(f"  params={info.get('gen_n_params')}  权重估算={info.get('gen_weight_gb')} GB")
    check("load() 返回参数量", (info.get("gen_n_params") or 0) > 0, info)
    # 曾经把字节数当 GB 打印过（1.54B 会显示成 3080000000 GB），这里守住量纲
    weight_gb = info.get("gen_weight_gb") or 0
    expect_gb = (info.get("gen_n_params") or 0) * 4 / 1e9
    check("权重 GB 量纲正确（未漏除 1e9）", abs(weight_gb - expect_gb) < 1e-3,
          f"{weight_gb} GB vs 期望 {expect_gb:.6f} GB")

    hits = [
        {"rank": 1, "paper_id": "P06", "title": "树结构 RAG", "section": "2 Method",
         "page": 3, "score": 0.8, "text": HIT_TEXTS[0]},
        {"rank": 2, "paper_id": "P03", "title": "Lost in the Middle", "section": "Abstract",
         "page": 1, "score": 0.7, "text": HIT_TEXTS[1]},
    ]
    messages = build_messages("固定分块有什么问题？", hits)

    has_template = bool(getattr(gen._tokenizer, "chat_template", None))
    check("tokenizer 带有 chat_template", has_template)
    try:
        prompt = gen.render(messages)
        check("render() 可执行", bool(prompt) and len(prompt) > 50, len(prompt or ""))
        check("prompt 含问题原文", "固定分块有什么问题" in prompt)
        check("prompt 含资料内容", "固定文本长度分块" in prompt)
        if has_template:
            check("chat template 已生效（含 im_start）", "im_start" in prompt)
    except Exception as exc:
        check("render() 可执行", False, f"{type(exc).__name__}: {str(exc)[:300]}")
        return

    # 流式路径（TTFT 计时依赖它）
    try:
        out = gen.generate(messages, max_new_tokens=8, use_stream=True)
        check("流式生成可执行", isinstance(out.get("text"), str), out)
        check("记录了 prompt_tokens", (out.get("prompt_tokens") or 0) > 0, out.get("prompt_tokens"))
        check("记录了 total_s", (out.get("total_s") or 0) > 0, out.get("total_s"))
        check("TTFT 已记录或模型无输出",
              out.get("ttft_s") is not None or out.get("out_tokens") == 0, out.get("ttft_s"))
        print(f"  流式：prompt_tokens={out.get('prompt_tokens')} out_tokens={out.get('out_tokens')} "
              f"ttft={out.get('ttft_s')}s total={out.get('total_s')}s")
    except Exception as exc:
        check("流式生成可执行", False, f"{type(exc).__name__}: {str(exc)[:300]}")

    # 非流式回退路径
    try:
        out2 = gen.generate(prompt, max_new_tokens=8, use_stream=False)
        check("非流式回退可执行", isinstance(out2.get("text"), str), out2)
        check("非流式也统计 token", (out2.get("prompt_tokens") or 0) > 0)
    except Exception as exc:
        check("非流式回退可执行", False, f"{type(exc).__name__}: {str(exc)[:300]}")

    # answer() 便捷封装（agent 用的就是它）
    try:
        res = gen.answer("固定分块有什么问题？", hits, max_new_tokens=8)
        check("answer() 返回 answer 字段", isinstance(res.get("answer"), str))
        check("answer() 返回 citations", len(res.get("citations", [])) == 2, len(res.get("citations", [])))
        check("answer() 统计论文数", res.get("n_papers") == 2, res.get("n_papers"))
    except Exception as exc:
        check("answer() 便捷封装", False, f"{type(exc).__name__}: {str(exc)[:300]}")

    check("fallback_prompt 可用", "im_start" in fallback_prompt(messages))
    check("build_citations 可用", len(build_citations(hits)) == 2)


# ── 3. 端到端 ───────────────────────────────────────────────────────
def test_end_to_end(embed_dir: Path, gen_dir: Path) -> None:
    section("3. 端到端：编码 -> 检索 -> prompt -> 生成")
    from embedder import BertEmbedder
    from generator import QwenGenerator, build_messages
    from retrieval import coverage, retrieve
    from vector_store import VectorStore

    emb = BertEmbedder(model_name=str(embed_dir), device="cpu", dtype="float32")
    emb.load()
    gen = QwenGenerator(model_name=str(gen_dir), device="cpu", dtype="float32", max_new_tokens=6)
    gen.load()

    chunks = [
        {"chunk_id": "P06#s2c0", "paper_id": "P06", "title": "树结构 RAG",
         "section": "2 Method", "page": 3, "lang": "zh", "text": HIT_TEXTS[0]},
        {"chunk_id": "P03#s1c0", "paper_id": "P03", "title": "Lost in the Middle",
         "section": "Abstract", "page": 1, "lang": "en", "text": HIT_TEXTS[1]},
        {"chunk_id": "P09#s1c0", "paper_id": "P09", "title": "字词资源 RAG",
         "section": "Abstract", "page": 1, "lang": "zh", "text": "字词资源可提升模型的语言理解能力。"},
    ]
    vectors = emb.encode([c["text"] for c in chunks], kind="passage")
    VectorStore(vectors)

    qvec = emb.encode(["固定分块有什么问题？"], kind="query")
    hits = retrieve(chunks, vectors, qvec, k=3, per_paper_cap=2, min_papers=2)
    cov = coverage(hits)
    print(f"  检索：{cov['n_hits']} 条 / {cov['n_papers']} 篇  {cov['per_paper']}")
    check("端到端检索有结果", len(hits) >= 1, len(hits))
    check("端到端覆盖 >= 2 篇", cov["n_papers"] >= 2, cov)
    check("命中带溯源信息", all(h.get("paper_id") and "page" in h for h in hits))

    messages = build_messages("固定分块有什么问题？", hits)
    result = gen.generate(messages, max_new_tokens=6)
    check("端到端生成返回文本", isinstance(result.get("text"), str))
    check("端到端记录了 prompt_tokens", (result.get("prompt_tokens") or 0) > 0)
    print(f"  生成：prompt_tokens={result.get('prompt_tokens')} total={result.get('total_s')}s")


# ── main ────────────────────────────────────────────────────────────
def main() -> int:
    print("torch 路径验证（本地构造 tiny 模型，无需网络）")
    try:
        import torch
        import transformers
    except ImportError as exc:
        print(f"\n跳过：缺少依赖（{exc}）")
        return 0

    print(f"python={sys.version.split()[0]}  torch={torch.__version__}  "
          f"transformers={transformers.__version__}  cuda={torch.cuda.is_available()}")

    section("0. 构造 tiny 模型")
    try:
        embed_dir, gen_dir = build_tiny_models(TINY_DIR)
        print(f"  bert-tiny  -> {embed_dir.name}")
        print(f"  qwen2-tiny -> {gen_dir.name}")
        check("tiny 模型构造成功", True)
    except Exception as exc:
        check("tiny 模型构造成功", False, f"{type(exc).__name__}: {str(exc)[:300]}")
        print("\n" + "=" * 60)
        print(f"FAILED: {len(FAILS)} 项")
        return 1

    try:
        test_embedder(embed_dir)
        test_generator(gen_dir)
        test_end_to_end(embed_dir, gen_dir)
    finally:
        shutil.rmtree(TINY_DIR, ignore_errors=True)

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
