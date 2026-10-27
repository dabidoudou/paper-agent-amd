"""计时与算力指标采集。

设计意图
--------
报告里要写「算力消耗数据」，就必须让每个阶段都能独立计时，且结果能直接导成表格。
这里统一用长格式（metric / value / unit）记录，导出 CSV 后可直接进 Excel，
不必事后从日志里扒数字。

GPU 首次执行含 kernel 装载开销，会虚高 2~3 倍，因此所有计时点都约定：
**先 warmup，再测中位数**（见 `Metrics.repeat`）。
"""

from __future__ import annotations

import csv
import json
import platform
import statistics
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class Metrics:
    """收集形如 (name, value, unit) 的测量值，并落盘。"""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rows: list[dict] = []
        self.meta: dict = {}

    # ── 记录 ────────────────────────────────────────────────────────
    def record(self, name: str, value, unit: str = "s", **extra) -> None:
        self.rows.append(
            {"run_id": self.run_id, "metric": name, "value": value, "unit": unit, **extra}
        )

    def set_meta(self, key: str, value) -> None:
        self.meta[key] = value

    @contextmanager
    def measure(self, name: str, unit: str = "s", **extra):
        """with m.measure("embed_load"): ... -> 自动记录耗时。"""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, round(time.perf_counter() - t0, 6), unit, **extra)

    def repeat(self, name: str, fn, times: int = 5, warmup: int = 2, unit: str = "s") -> float:
        """执行 fn 若干次，warmup 若干次丢弃，记录中位数。返回中位数。"""
        for _ in range(max(0, warmup)):
            fn()
        samples = []
        for _ in range(max(1, times)):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        med = statistics.median(samples)
        self.record(name, round(med, 6), unit)
        self.record(f"{name}_min", round(min(samples), 6), unit)
        self.record(f"{name}_max", round(max(samples), 6), unit)
        return med

    # ── 环境 ────────────────────────────────────────────────────────
    def capture_env(self, torch_module=None) -> dict:
        """记录软硬件基线 —— 算力数据脱离了硬件就没有意义。"""
        env = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        if torch_module is not None:
            t = torch_module
            env["torch"] = t.__version__
            env["hip"] = getattr(t.version, "hip", None)
            env["cuda_available"] = bool(t.cuda.is_available())
            if t.cuda.is_available():
                env["device_name"] = t.cuda.get_device_name(0)
                props = t.cuda.get_device_properties(0)
                env["device_total_mem_gb"] = round(props.total_memory / 1024**3, 2)
                env["device_cu_count"] = getattr(props, "multi_processor_count", None)
        self.meta["env"] = env
        return env

    def peak_gpu_mb(self, torch_module) -> float | None:
        """峰值显存分配（ROCm 下同样通过 torch.cuda 接口读取）。"""
        try:
            return round(torch_module.cuda.max_memory_allocated() / 1024**2, 1)
        except Exception:
            return None

    # ── 落盘 ────────────────────────────────────────────────────────
    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "meta": self.meta,
            "metrics": {r["metric"]: {"value": r["value"], "unit": r["unit"]} for r in self.rows},
        }

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["run_id", "metric", "value", "unit"])
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: row[k] for k in writer.fieldnames})
        return path

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def print_table(self) -> None:
        print(f"{'metric':34s} {'value':>14s}  unit")
        print("-" * 60)
        for row in self.rows:
            value = row["value"]
            shown = f"{value:,.4f}" if isinstance(value, float) else str(value)
            print(f"{row['metric']:34s} {shown:>14s}  {row['unit']}")
