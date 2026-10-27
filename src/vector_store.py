"""向量检索封装：FAISS 优先，numpy 兜底。

为什么保留 numpy 兜底
--------------------
1. `faiss-cpu` 在部分 Python 版本 / 平台缺少预编译轮子，装不上就会让整个 demo 挂掉；
   而本 demo 的语料只有几十个 chunk，`numpy` 内积完全够用（微秒级）。
2. 兜底路径让"云端装依赖失败"不再等于"演示失败"，这是可复现性的一部分。

为什么只持久化向量、不持久化索引
--------------------------------
几十到几千条向量的 `IndexFlatIP` 重建耗时 < 10 ms，而序列化索引文件会带来
跨 faiss 版本的兼容问题。**向量是数据，索引是派生物** —— 存数据、重建派生。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """按行做 L2 归一化。

    归一化后「内积 == 余弦相似度」，因此 FAISS 用 IndexFlatIP 即可等价实现余弦检索。
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        raise ValueError("向量为空，无法归一化")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


class VectorStore:
    """极简向量库：build / search / 持久化向量。"""

    def __init__(self, vectors: np.ndarray | None = None, backend: str = "auto"):
        self._vectors: np.ndarray | None = None
        self._index = None
        self.backend: str | None = None
        self.dim: int | None = None
        if vectors is not None:
            self.build(vectors, backend=backend)

    # ── 构建 ────────────────────────────────────────────────────────
    def build(self, vectors: np.ndarray, backend: str = "auto") -> "VectorStore":
        """建立索引。backend: auto / faiss / numpy。"""
        mat = l2_normalize(vectors)
        if mat.shape[0] == 0:
            raise ValueError("向量数量为 0，无法建库")
        self._vectors = mat
        self.dim = int(mat.shape[1])
        self._index = None

        if backend in ("auto", "faiss"):
            try:
                import faiss  # type: ignore

                index = faiss.IndexFlatIP(self.dim)
                index.add(mat)
                self._index = index
                self.backend = "faiss"
                return self
            except ImportError:
                if backend == "faiss":
                    raise RuntimeError(
                        "指定的 backend='faiss' 不可用；请 pip install faiss-cpu，"
                        "或改用 backend='numpy'"
                    ) from None

        self.backend = "numpy"
        return self

    # ── 检索 ────────────────────────────────────────────────────────
    def search(self, query_vector: np.ndarray, k: int = 3):
        """返回 (scores, indices)，均为按分数降序排列的一维数组。"""
        if self._vectors is None:
            raise RuntimeError("索引尚未构建，请先调用 build()")
        q = l2_normalize(query_vector)
        k = max(1, min(int(k), int(self._vectors.shape[0])))

        if self.backend == "faiss":
            scores, indices = self._index.search(q, k)  # type: ignore[union-attr]
            scores = np.asarray(scores[0], dtype=np.float32)
            indices = np.asarray(indices[0], dtype=np.int64)
            # faiss 在候选不足时用 -1 填充
            valid = indices >= 0
            return scores[valid], indices[valid]

        scores = (self._vectors @ q[0]).astype(np.float32)
        order = np.argsort(-scores)[:k]
        return scores[order], order.astype(np.int64)

    # ── 属性 ────────────────────────────────────────────────────────
    @property
    def size(self) -> int:
        return 0 if self._vectors is None else int(self._vectors.shape[0])

    @property
    def n_bytes(self) -> int:
        """向量占用字节数 —— 算力规划里「索引内存」这一项的实测值。"""
        return 0 if self._vectors is None else int(self._vectors.nbytes)

    def describe(self) -> str:
        return (
            f"VectorStore(backend={self.backend}, n={self.size}, dim={self.dim}, "
            f"mem={self.n_bytes / 1024:.1f} KiB)"
        )

    # ── 持久化 ──────────────────────────────────────────────────────
    def save_vectors(self, path: str | Path) -> Path:
        """只保存归一化后的向量（.npz）。"""
        if self._vectors is None:
            raise RuntimeError("没有可保存的向量")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), vectors=self._vectors)
        return path

    @classmethod
    def from_file(cls, path: str | Path, backend: str = "auto") -> "VectorStore":
        """从 .npz 恢复向量并重建索引。"""
        data = np.load(str(path))
        return cls(data["vectors"], backend=backend)
