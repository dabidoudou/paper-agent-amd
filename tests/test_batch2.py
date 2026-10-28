"""第 2 批代码的本地验证（只验证不依赖 torch 的纯逻辑部分）。

运行：
    python tests/test_batch2.py

设计要点：`generator.py` 与 `embedder.py` 都把 torch 做成**惰性导入**，
因此本文件在没有任何深度学习依赖的环境里也能完整跑通——
这既是测试，也证明了"云端装不上 torch 时至少还能验证 prompt 与管道"这件事成立。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from embedder import (  # noqa: E402
    BertEmbedder,
    HashingEmbedder,
    build_embedder,
    hashing_encode,
    l2_normalize,
)
from generator import (  # noqa: E402
    SYSTEM_PROMPT,
    QwenGenerator,
    build_context,
    build_messages,
    fallback_prompt,
)
from timing import Metrics  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  -> {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


HITS = [
    {"rank": 1, "paper_id": "P06", "title": "面向工艺规范的树结构检索增强生成方法研究",
     "text": "固定长度分块会丢失段落层级结构。"},
    {"rank": 2, "paper_id": "P01", "title": "Retrieval-Augmented Generation",
     "text": "The parametric memory is a seq2seq model."},
]


# ── 1. Prompt 构造（纯函数）─────────────────────────────────────────
def test_prompt() -> None:
    section("1. Prompt 构造")
    ctx = build_context(HITS)
    check("资料块带编号与来源", "[1] (P06 /" in ctx and "[2] (P01 /" in ctx)
    check("资料块含正文", "固定长度分块" in ctx and "seq2seq" in ctx)
    check("资料块之间空行分隔", "\n\n" in ctx)

    # rank 缺失时不应崩溃（v2 起资料按论文分组，故只断言占位符存在）
    ctx2 = build_context([{"paper_id": "P09", "title": "t", "text": "x"}])
    check("rank 缺失时回退为 ?", "[?]" in ctx2, ctx2[:60])

    msgs = build_messages("固定长度分块有什么问题？", HITS)
    check("messages 为 2 条（system+user）", len(msgs) == 2, len(msgs))
    check("system 含拒答指令", "无法回答" in msgs[0]["content"])
    user = msgs[1]["content"]
    check("user 含资料段", "【资料】" in user)
    check("user 含问题", "固定长度分块有什么问题？" in user)
    check("user 以回答段结尾", user.rstrip().endswith("【回答】"))

    # 问题首尾空白应被清理
    msgs2 = build_messages("  带空格的问题  ", HITS)
    check("问题空白被清理", "【问题】带空格的问题" in msgs2[1]["content"])

    fb = fallback_prompt(msgs)
    check("降级 prompt 含角色标记", fb.count("<|im_start|>") >= 3, fb.count("<|im_start|>"))
    check("降级 prompt 以 assistant 开头结尾", fb.rstrip().endswith("assistant"))
    check("SYSTEM_PROMPT 非空且含禁止编造", "禁止编造" in SYSTEM_PROMPT)


# ── 2. 免依赖编码器 ─────────────────────────────────────────────────
def test_hashing_embedder() -> None:
    section("2. 免依赖编码器（hashing）")
    texts = ["检索增强生成", "固定长度分块", "retrieval augmented generation"]
    vecs = hashing_encode(texts, dim=256)
    check("输出形状 [n, dim]", vecs.shape == (3, 256), vecs.shape)
    check("dtype 为 float32", vecs.dtype == np.float32, vecs.dtype)
    norms = np.linalg.norm(vecs, axis=1)
    check("每行已 L2 归一化", bool(np.allclose(norms, 1.0, atol=1e-5)), norms.tolist())
    again = hashing_encode(texts, dim=256)
    check("同输入输出确定（可复现）", bool(np.array_equal(vecs, again)))
    check("空输入返回空矩阵", hashing_encode([], dim=64).shape == (0, 64))

    emb = HashingEmbedder(dim=128)
    info = emb.load()
    check("load() 返回元信息", info.get("embed_dim") == 128, info)
    out = emb.encode(texts)
    check("encode 接口与 BertEmbedder 一致", out.shape == (3, 128), out.shape)

    # l2_normalize 对零向量的保护
    zero = l2_normalize(np.zeros((1, 4), dtype=np.float32))
    check("零向量不被除零（有限值）", bool(np.all(np.isfinite(zero))), zero.tolist())


# ── 3. BertEmbedder 惰性加载 ────────────────────────────────────────
def test_bert_lazy() -> None:
    section("3. BertEmbedder 惰性加载")
    emb = BertEmbedder(model_name="dummy/model", device="cpu")
    check("未装 torch 也能实例化", emb is not None)
    check("未加载时 encode 抛 RuntimeError", _raises(RuntimeError, lambda: emb.encode(["x"])))
    check("未加载时给出去重前缀配置", emb.query_prefix == "query: " and emb.passage_prefix == "passage: ")


def _raises(exc_type, fn) -> bool:
    try:
        fn()
    except exc_type:
        return True
    except Exception:
        return False
    return False


# ── 4. 工厂函数 ─────────────────────────────────────────────────────
def test_factory() -> None:
    section("4. 编码器工厂")
    check("hashing 分支", isinstance(build_embedder("hashing", dim=32), HashingEmbedder))
    check("bert 分支", isinstance(build_embedder("bert", model_name="m"), BertEmbedder))
    check("未知类型报错", _raises(ValueError, lambda: build_embedder("bogus")))


# ── 5. 计时与指标落盘 ───────────────────────────────────────────────
def test_metrics() -> None:
    section("5. 计时与指标落盘")
    m = Metrics(run_id="test_run")
    with m.measure("demo_load"):
        pass
    val = m.repeat("demo_repeat", lambda: None, times=3, warmup=1)
    check("measure 记录一次", any(r["metric"] == "demo_load" for r in m.rows))
    check("repeat 返回中位数", isinstance(val, float) and val >= 0)
    check("repeat 记录 min/max", {"demo_repeat_min", "demo_repeat_max"} <= {r["metric"] for r in m.rows})

    env = m.capture_env()
    check("capture_env 含 python 版本", "python" in env, env)
    check("peak_gpu_mb 对无 torch 对象返回 None", m.peak_gpu_mb(object()) is None)

    # 注意：不用 tempfile —— 本环境的沙箱禁止写入系统临时目录，
    # 统一落在工作区内，跑完清理。
    tmpdir = ROOT / "results" / "_tmp_test_batch2"
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        csv_path = m.to_csv(tmpdir / "timing.csv")
        json_path = m.to_json(tmpdir / "run.json")
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        check("CSV 首行为表头", lines[0] == "run_id,metric,value,unit", lines[0])
        check("CSV 行数 = 记录数 + 1", len(lines) == len(m.rows) + 1, (len(lines), len(m.rows)))
        data = json.loads(json_path.read_text(encoding="utf-8"))
        check("JSON 含 run_id 与 metrics", data["run_id"] == "test_run" and "demo_load" in data["metrics"])
        check("JSON 含 env 元信息", "env" in data["meta"])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── 6. QwenGenerator 惰性构建 ───────────────────────────────────────
def test_generator_lazy() -> None:
    section("6. QwenGenerator 惰性构建")
    gen = QwenGenerator(model_name="dummy/qwen")
    check("未装 torch 也能实例化", gen is not None)
    check("未加载时 generate 抛 RuntimeError", _raises(RuntimeError, lambda: gen.generate("hi")))
    check("无 tokenizer 时 render 走降级格式", "<|im_start|>user" in gen.render(build_messages("q", HITS)))
    check("bytes_per_param 对应 bf16", gen._bytes_per_param() == 2.0)
    check("fp32 权重估算翻倍", QwenGenerator(dtype="float32")._bytes_per_param() == 4.0)


# ── main ────────────────────────────────────────────────────────────
def main() -> int:
    print("第 2 批本地验证（无 torch / transformers 依赖）")
    print(f"python={sys.version.split()[0]}  numpy={np.__version__}")
    test_prompt()
    test_hashing_embedder()
    test_bert_lazy()
    test_factory()
    test_metrics()
    test_generator_lazy()

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
