"""文本向量化：手写 mean-pooling 编码器 + 无依赖哈希编码器。

为什么不装 sentence-transformers
--------------------------------
1. 它会声明 `torch>=1.11` 依赖。AMD 云端镜像里的 torch 是 ROCm 定制构建，
   一旦 pip 认为版本不满足，会把 torch 换成 CUDA/CPU 轮子，
   `torch.cuda.is_available()` 立刻变 False —— 这是本项目最大的环境风险。
2. 它内部做的事其实只有三步：分词 -> 前向 -> 按 attention_mask 做 mean pooling -> L2 归一化。
   手写不到 20 行，且每一步都能单独计时，正好满足"算力数据采集"的需求。

`HashingEmbedder` 的作用
-----------------------
不依赖 torch 的确定性编码器（字符 n-gram 哈希 + L2 归一化）。它只保留字面重合能力，
**不代表真实语义检索效果**，但能让整条管道在没有 GPU、没有模型的情况下跑通：
云端装依赖失败时可用 `--embedder hashing` 先验证流程，本地也能做回归测试。
"""

from __future__ import annotations

import hashlib

import numpy as np


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """按行 L2 归一化（与 vector_store.l2_normalize 等价，避免循环导入）。"""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


def _from_pretrained(factory, model_name: str, torch_dtype):
    """兼容不同 transformers 版本的精度参数名。

    transformers 4.5x 起把 `torch_dtype` 改名为 `dtype`，5.x 可能已移除旧名；
    两种写法各试一次，避免因版本差异在云端直接崩掉。
    """
    try:
        return factory.from_pretrained(model_name, torch_dtype=torch_dtype)
    except TypeError:
        return factory.from_pretrained(model_name, dtype=torch_dtype)


def hashing_encode(texts: list[str], dim: int = 512) -> np.ndarray:
    """确定性字符 n-gram 哈希编码（仅用于管道验证）。"""
    rows = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        chars = list(text or "")
        for n in (1, 2):
            for i in range(len(chars) - n + 1):
                gram = "".join(chars[i : i + n])
                if not gram.strip():
                    continue
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
                vec[h % dim] += 1.0
        rows.append(vec)
    return l2_normalize(np.vstack(rows) if rows else np.zeros((0, dim), dtype=np.float32))


class HashingEmbedder:
    """与 BertEmbedder 接口一致的免依赖编码器。"""

    name = "hashing"

    def __init__(self, dim: int = 512, **_ignored):
        self.dim = dim

    def load(self) -> dict:
        return {"embed_backend": self.name, "embed_dim": self.dim, "embed_load_s": 0.0}

    def encode(self, texts: list[str], kind: str = "passage", batch_size: int = 16) -> np.ndarray:
        return hashing_encode(texts, self.dim)


class BertEmbedder:
    """基于 transformers 的手写句向量编码器。

    与 sentence-transformers 的 `encode()` 输出等价（mean pooling + L2 归一化），
    但依赖更少、每步可计时。
    """

    name = "bert"

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str | None = None,
        dtype: str = "float32",
        max_length: int = 512,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ):
        self.model_name = model_name
        self.device = device
        self.dtype_name = dtype
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._torch = None
        self._model = None
        self._tokenizer = None
        self.dim: int | None = None

    # ── 加载 ────────────────────────────────────────────────────────
    def load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype_name]

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = _from_pretrained(AutoModel, self.model_name, torch_dtype)
        self._model.to(self.device).eval()
        self.dim = int(self._model.config.hidden_size)
        return {
            "embed_backend": self.name,
            "embed_model": self.model_name,
            "embed_dim": self.dim,
            "embed_device": self.device,
            "embed_dtype": self.dtype_name,
        }

    # ── 编码 ────────────────────────────────────────────────────────
    def _prepare(self, texts: list[str], kind: str) -> list[str]:
        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        if not prefix:
            return texts
        return [prefix + t for t in texts]

    def encode(
        self, texts: list[str], kind: str = "passage", batch_size: int = 16
    ) -> np.ndarray:
        """返回 [n, dim] 的 L2 归一化向量矩阵。

        kind="query" 时加查询前缀，"passage" 时加段落前缀（E5 系列要求）。
        """
        if self._model is None:
            raise RuntimeError("模型未加载，请先调用 load()")
        torch = self._torch
        prepared = self._prepare(texts, kind)
        out: list[np.ndarray] = []

        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self._model(**enc).last_hidden_state          # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
            # mean pooling：必须乘 attention_mask，否则 padding 会污染句向量
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            pooled = (summed / counts).float().cpu().numpy()
            out.append(pooled)

        stacked = np.vstack(out) if out else np.zeros((0, self.dim or 0), dtype=np.float32)
        return l2_normalize(stacked)


def build_embedder(kind: str = "bert", **kwargs):
    """工厂：kind = bert / hashing。"""
    kind = (kind or "bert").lower()
    if kind == "hashing":
        return HashingEmbedder(**kwargs)
    if kind == "bert":
        return BertEmbedder(**kwargs)
    raise ValueError(f"未知的 embedder 类型：{kind}（可选 bert / hashing）")
