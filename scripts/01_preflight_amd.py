"""AMD AUP Learning Cloud 云端预检。

为什么需要这个脚本
------------------
云端实例只有 60 分钟，而且 `torch.cuda.is_available()` 返回 True **并不代表 GPU 真的能用**——
它只说明驱动和运行时库存在，不代表 kernel 能跑、也不代表 bf16 不溢出。
这个脚本用 4 项硬指标把"能不能跑本项目"一次问清楚：

1. **真实矩阵乘**（fp32 / bf16 / fp16）—— 能算完并同步成功，才说明 kernel 可用
2. **数值稳定性** —— bf16/fp16 结果与 fp64 参考的相对误差，决定 `GEN_DTYPE` 该用哪个
3. **显存带宽** —— 解码速度的物理上限 = 带宽 / 权重字节数，据此判断 1.5B 模型能跑多快
4. **依赖与导入** —— transformers 是否认识 Qwen2、TextIteratorStreamer 是否可用、
   faiss/pymupdf 是否装上、本项目全部模块能否干净 import

用法：
    python scripts/01_preflight_amd.py
    python scripts/01_preflight_amd.py --no-network   # 跳过联网检查
    python scripts/01_preflight_amd.py --json results/preflight.json

退出码 0 表示关键项全部通过；非 0 时按报告里的 [FAIL] 逐条处理。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CRITICAL = True
OPTIONAL = False


class Report:
    """收集检查项并输出报告。"""

    def __init__(self) -> None:
        self.items: list[dict] = []
        self.failed: list[str] = []

    def add(self, name: str, ok: bool, detail: str = "", critical: bool = True) -> bool:
        status = "PASS" if ok else ("FAIL" if critical else "WARN")
        self.items.append(
            {"name": name, "ok": ok, "critical": critical, "status": status, "detail": str(detail)}
        )
        suffix = "" if ok else f"  ({detail})"
        print(f"  [{status}] {name}{suffix}")
        if not ok and critical:
            self.failed.append(name)
        return ok

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "n_pass": sum(1 for i in self.items if i["ok"]),
            "n_fail": len(self.failed),
            "items": self.items,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


# ── 1. 基础环境 ─────────────────────────────────────────────────────
def check_env(report: Report) -> None:
    report.section("1. 基础环境")
    report.add("python >= 3.9", sys.version_info >= (3, 9), sys.version.split()[0])
    report.add("平台信息", True, platform.platform())
    report.add("CPU 核数", True, os.cpu_count())

    total, used, free = shutil.disk_usage(ROOT)
    free_gb = free / 1024**3
    report.add(
        "可用磁盘 >= 8GB（模型约 3.6GB + PDF）",
        free_gb >= 8,
        f"{free_gb:.1f} GB 可用 / {total / 1024**3:.0f} GB 总",
        critical=OPTIONAL,
    )

    probe = ROOT / ".preflight_write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        writable = True
    except Exception:
        writable = False
    report.add("工作目录可写", writable, str(ROOT))

    hf_home = os.environ.get("HF_HOME", "")
    report.add(
        "HF_HOME 已指到持久化目录",
        bool(hf_home),
        hf_home or "未设置：模型会下到 ~/.cache，实例重启后需重下 3.6GB",
        critical=OPTIONAL,
    )


# ── 2. torch 与真实 GPU 能力 ────────────────────────────────────────
def check_torch(report: Report) -> dict:
    report.section("2. torch / ROCm 与真实 GPU 能力")
    info: dict = {}
    try:
        import torch
    except ImportError as exc:
        report.add("import torch", False, exc)
        return info

    info["torch"] = torch.__version__
    info["hip"] = getattr(torch.version, "hip", None)
    report.add("torch 已安装", True, torch.__version__)
    report.add("ROCm 构建（torch.version.hip 非空）", bool(info["hip"]), info["hip"])

    cuda_ok = bool(torch.cuda.is_available())
    info["cuda_available"] = cuda_ok
    report.add("torch.cuda.is_available()", cuda_ok, cuda_ok)
    if not cuda_ok:
        print("    → 只能用 CPU 跑，建议排查 ROCm 安装或是否用了 CPU 版 torch")
        return info

    info["device_name"] = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    info["total_mem_gb"] = round(props.total_memory / 1024**3, 2)
    info["cu_count"] = getattr(props, "multi_processor_count", None)
    print(f"    设备：{info['device_name']}")
    print(f"    显存：{info['total_mem_gb']} GB  计算单元：{info['cu_count']}")

    # ① 真实矩阵乘 —— 这是判断 GPU 能否真正干活的第一项硬指标
    report.section("3. 真实矩阵乘（kernel 可用性 + 算力）")
    dim = 4096
    gflops = 2 * dim**3 / 1e9
    for dtype_name in ("float32", "bfloat16", "float16"):
        dtype = getattr(torch, dtype_name)
        try:
            a = torch.randn(dim, dim, device="cuda", dtype=dtype)
            b = torch.randn(dim, dim, device="cuda", dtype=dtype)
            c = a @ b  # warmup：首次含 kernel 装载
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            c = a @ b
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            ok = bool(torch.isfinite(c).all().item())
            tflops = gflops / elapsed / 1000
            info[f"matmul_{dtype_name}_s"] = round(elapsed, 4)
            info[f"matmul_{dtype_name}_tflops"] = round(tflops, 2)
            report.add(
                f"{dtype_name} 矩阵乘 4096³",
                ok,
                f"{elapsed * 1000:.0f} ms / {tflops:.1f} TFLOP/s",
            )
            del a, b, c
            torch.cuda.empty_cache()
        except Exception as exc:
            report.add(f"{dtype_name} 矩阵乘 4096³", False, f"{type(exc).__name__}: {exc}")

    # ② 数值稳定性 —— 决定生成用什么 dtype
    report.section("4. 数值稳定性（决定 GEN_DTYPE）")
    try:
        small = 1024
        base = torch.randn(small, small, device="cuda", dtype=torch.float64)
        ref = base @ base
        for dtype_name in ("float32", "bfloat16", "float16"):
            dtype = getattr(torch, dtype_name)
            x = base.to(dtype)
            out = (x @ x).double()
            rel = ((out - ref).abs().mean() / ref.abs().mean()).item()
            info[f"rel_err_{dtype_name}"] = float(f"{rel:.3e}")
            budget = {"float32": 1e-5, "bfloat16": 1e-2, "float16": 1e-3}[dtype_name]
            report.add(
                f"{dtype_name} 相对误差可接受（<{budget:g}）",
                rel < budget,
                f"mean|err|/mean|ref| = {rel:.3e}",
                critical=OPTIONAL,
            )
        print("    → 建议：bf16 在 ROCm 上通常最稳，本项目默认 GEN_DTYPE=bfloat16")
    except Exception as exc:
        report.add("数值稳定性测量", False, f"{type(exc).__name__}: {exc}", critical=OPTIONAL)

    # ③ 显存带宽 —— 解码速度的物理上限
    report.section("5. 显存带宽（解码速度上限）")
    try:
        n_elems = 128 * 1024 * 1024  # bf16 => 256 MB
        src = torch.ones(n_elems, device="cuda", dtype=torch.bfloat16)
        dst = torch.empty_like(src)
        dst.copy_(src)
        torch.cuda.synchronize()

        rounds = 10
        t0 = time.perf_counter()
        for _ in range(rounds):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        moved_bytes = rounds * n_elems * 2 * 2  # 读 + 写
        bw_gbs = moved_bytes / elapsed / 1e9
        info["mem_bandwidth_gbs"] = round(bw_gbs, 1)
        report.add("带宽实测完成", bw_gbs > 1, f"{bw_gbs:.1f} GB/s")

        # 1.54B 参数 × 2 字节 = 3.08 GB；每生成一个 token 要把全部权重读一遍
        weight_gb = 1.54e9 * 2 / 1024**3
        ceiling = bw_gbs / weight_gb
        info["decode_ceiling_tok_s_1p5b"] = round(ceiling, 1)
        print(
            f"    理论上限：Qwen2.5-1.5B(bf16, {weight_gb:.2f} GB) "
            f"≈ {ceiling:.0f} tok/s（实测通常为上限的 40%~70%）"
        )
        for params_b in (7, 14):
            w = params_b * 1e9 * 2 / 1024**3
            print(f"    参考：{params_b}B 模型 ≈ {bw_gbs / w:.1f} tok/s")
        del src, dst
        torch.cuda.empty_cache()
    except Exception as exc:
        report.add("显存带宽测量", False, f"{type(exc).__name__}: {exc}", critical=OPTIONAL)

    info["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    return info


# ── 3. 依赖与项目模块导入 ───────────────────────────────────────────
def check_deps(report: Report) -> dict:
    report.section("6. 依赖可用性")
    info: dict = {}

    for mod in ("numpy", "certifi", "huggingface_hub"):
        try:
            m = importlib.import_module(mod)
            info[mod] = getattr(m, "__version__", "?")
            report.add(f"{mod} 可用", True, info[mod], critical=OPTIONAL)
        except ImportError:
            report.add(f"{mod} 可用", False, "未安装", critical=OPTIONAL)

    try:
        import faiss

        info["faiss"] = getattr(faiss, "__version__", "?")
        index = faiss.IndexFlatIP(8)
        index.add(__import__("numpy").zeros((2, 8), dtype="float32"))
        report.add("faiss 可用（IndexFlatIP 正常）", True, info["faiss"], critical=OPTIONAL)
    except ImportError:
        report.add(
            "faiss 可用", False, "未安装 —— 代码会自动回退 numpy，不影响演示", critical=OPTIONAL
        )
    except Exception as exc:
        report.add("faiss 可用", False, f"{type(exc).__name__}: {exc}", critical=OPTIONAL)

    try:
        import pymupdf

        info["pymupdf"] = getattr(pymupdf, "__version__", "?")
        report.add("pymupdf 可用（PDF 抽取）", True, info["pymupdf"])
    except ImportError:
        try:
            import pypdf

            info["pypdf"] = pypdf.__version__
            report.add("pymupdf 缺失但 pypdf 可用（回退方案）", True, info["pypdf"], critical=OPTIONAL)
        except ImportError:
            report.add(
                "PDF 解析库", False, "pymupdf 与 pypdf 都缺失 —— load 步骤会失败", critical=True
            )

    # transformers 与 Qwen2 架构支持
    try:
        import transformers

        info["transformers"] = transformers.__version__
        report.add("transformers 可用", True, transformers.__version__)
        try:
            from transformers import Qwen2ForCausalLM  # noqa: F401

            report.add("transformers 认识 Qwen2 架构", True, ">= 4.37")
        except ImportError:
            report.add(
                "transformers 认识 Qwen2 架构",
                False,
                "当前版本过旧，需 >= 4.37；可单独 pip install -U transformers（纯 Python 包，不碰 torch）",
            )
        try:
            from transformers import TextIteratorStreamer  # noqa: F401

            report.add("TextIteratorStreamer 可用（TTFT 计时需要）", True)
        except ImportError:
            report.add(
                "TextIteratorStreamer 可用", False, "缺失 → 用 use_stream=False 回退", critical=OPTIONAL
            )
    except ImportError:
        report.add("transformers 可用", False, "未安装 —— 嵌入与生成都无法运行")

    report.section("7. 项目模块导入（全部在无 GPU 条件下应可 import）")
    modules = [
        "config", "chunker", "corpus", "embedder", "fulltext", "generator",
        "library", "retrieval", "timing", "vector_store", "discovery", "fetcher", "agent",
    ]
    bad: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            bad.append(f"{name}: {type(exc).__name__}")
    report.add(
        f"{len(modules)} 个项目模块可导入", not bad, f"失败 {len(bad)} 个：{bad}" if bad else "全部通过"
    )
    return info


# ── 4. 联网 ─────────────────────────────────────────────────────────
def check_network(report: Report) -> None:
    report.section("8. 联网能力")
    try:
        from discovery import search_arxiv

        t0 = time.perf_counter()
        results = search_arxiv("retrieval augmented generation", limit=2)
        elapsed = time.perf_counter() - t0
        report.add(
            "arXiv API 可访问", len(results) > 0,
            f"{len(results)} 条 / {elapsed:.1f}s", critical=OPTIONAL,
        )
        if results:
            print(f"    示例：{results[0]['arxiv_id']}  {results[0]['title'][:52]}")
    except Exception as exc:
        report.add(
            "arXiv API 可访问", False, f"{type(exc).__name__}: {exc}", critical=OPTIONAL
        )


# ── main ────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AMD AUP 云端预检")
    parser.add_argument("--no-network", action="store_true", help="跳过联网检查")
    parser.add_argument("--json", default=None, help="报告输出路径")
    args = parser.parse_args(argv)

    print("AMD AUP Learning Cloud 预检 —— paper-agent-amd")
    print(f"时间：{datetime.now().isoformat(timespec='seconds')}")

    report = Report()
    check_env(report)
    torch_info = check_torch(report)
    deps = check_deps(report)
    if not args.no_network:
        check_network(report)

    out = args.json or str(ROOT / "results" / "preflight.json")
    report.items.append({"name": "_torch_info", "ok": True, "critical": False,
                         "status": "INFO", "detail": json.dumps(torch_info, ensure_ascii=False)})
    report.items.append({"name": "_deps", "ok": True, "critical": False,
                         "status": "INFO", "detail": json.dumps(deps, ensure_ascii=False)})
    path = report.to_json(out)

    print("\n" + "=" * 64)
    if report.failed:
        print(f"关键项失败 {len(report.failed)} 个：")
        for item in report.items:
            if not item["ok"] and item["critical"]:
                print(f"  - {item['name']}: {item['detail']}")
        print(f"\n完整报告：{path}")
        return 1
    print("关键项全部通过，可以按 docs/cloud_runbook.md 的阶梯继续。")
    print(f"完整报告：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
