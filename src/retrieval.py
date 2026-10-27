"""多论文多样化检索：per-paper 配额 + MMR 去冗余 + 证据溯源。

为什么不能只用朴素 top-k
------------------------
全文建库后每个问题都能召回大量高分 chunk，但它们**往往全部落在同一篇论文里**，
于是"把多篇论文连结起来"这个需求就做不到了，而且答案会反复引用同一段内容。

本模块的两段式策略
------------------
1. **覆盖阶段（coverage）**：按各论文最高分排序，先给前 `min_papers` 篇论文各取一条，
   保证回答至少能跨 2 篇论文 —— 这是"多篇连结"的硬保证。
2. **填充阶段（fill）**：对剩余候选做 MMR（最大边际相关性），在相关性和多样性之间取平衡，
   同时遵守 `per_paper_cap` 限制单篇论文的占比。

返回的每条命中都带 `paper_id / title / section / page / chunk_id`，
生成阶段据此产出「论文 · 章节 · 页码」级的证据表。
"""

from __future__ import annotations

import numpy as np

from embedder import l2_normalize


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def retrieve(
    chunks: list[dict],
    vectors: np.ndarray,
    query_vector: np.ndarray,
    k: int = 8,
    per_paper_cap: int = 3,
    min_papers: int = 2,
    mmr_lambda: float = 0.7,
    pool_factor: int = 6,
) -> list[dict]:
    """检索并返回多样化的命中列表。

    Args:
        chunks: 与 vectors 行对齐的 chunk 元数据列表。
        vectors: [n, dim] 已 L2 归一化的 chunk 向量。
        query_vector: [dim] 或 [1, dim] 的查询向量。
        k: 最终返回条数。
        per_paper_cap: 单篇论文最多贡献多少条，防止一篇论文霸榜。
        min_papers: 至少覆盖多少篇论文。
        mmr_lambda: 1.0 = 纯相关性；0.0 = 纯多样性。0.7 是常用折中。
        pool_factor: 候选池大小 = k * pool_factor。

    Returns:
        [{rank, score, mmr_score, paper_id, title, section, page, chunk_id, text, ...}]
    """
    if vectors is None or len(vectors) == 0:
        return []
    vmat = l2_normalize(vectors)
    q = l2_normalize(query_vector)[0]
    n = vmat.shape[0]

    k = max(1, min(int(k), n))
    pool_size = min(n, max(k * max(1, pool_factor), k))

    rel = vmat @ q
    order = np.argsort(-rel)[:pool_size]

    # 候选池：(idx, relevance)
    pool = [(int(i), float(rel[i])) for i in order]
    by_paper: dict[str, float] = {}
    for idx, score in pool:
        pid = chunks[idx]["paper_id"]
        by_paper[pid] = max(by_paper.get(pid, -1e9), score)

    selected: list[int] = []
    taken_per_paper: dict[str, int] = {}

    def _take(idx: int) -> bool:
        pid = chunks[idx]["paper_id"]
        if taken_per_paper.get(pid, 0) >= per_paper_cap:
            return False
        selected.append(idx)
        taken_per_paper[pid] = taken_per_paper.get(pid, 0) + 1
        return True

    # ── 阶段 1：覆盖保证，按论文最高分依次各取一条 ────────────────
    for pid in sorted(by_paper, key=lambda p: -by_paper[p]):
        if len(selected) >= k or len({chunks[i]["paper_id"] for i in selected}) >= min_papers:
            break
        best_idx = max(
            (i for i, _ in pool if chunks[i]["paper_id"] == pid),
            key=lambda i: float(rel[i]),
        )
        _take(best_idx)

    # ── 阶段 2：MMR 填充 ────────────────────────────────────────────
    remaining = [idx for idx, _ in pool if idx not in selected]
    while len(selected) < k and remaining:
        best_idx, best_mmr = None, -1e9
        for idx in remaining:
            pid = chunks[idx]["paper_id"]
            if taken_per_paper.get(pid, 0) >= per_paper_cap:
                continue
            redundancy = (
                max(_cosine(vmat[idx], vmat[j]) for j in selected) if selected else 0.0
            )
            mmr = mmr_lambda * float(rel[idx]) - (1.0 - mmr_lambda) * redundancy
            if mmr > best_mmr:
                best_idx, best_mmr = idx, mmr
        if best_idx is None:
            break
        _take(best_idx)
        remaining.remove(best_idx)

    # ── 输出：按相关性降序编号 ────────────────────────────────────
    selected.sort(key=lambda i: -float(rel[i]))
    hits: list[dict] = []
    for rank, idx in enumerate(selected, start=1):
        chunk = chunks[idx]
        hits.append(
            {
                "rank": rank,
                "score": round(float(rel[idx]), 6),
                "paper_id": chunk["paper_id"],
                "title": chunk.get("title", ""),
                "section": chunk.get("section", ""),
                "page": chunk.get("page"),
                "chunk_id": chunk.get("chunk_id", str(idx)),
                "lang": chunk.get("lang", ""),
                "text": chunk["text"],
            }
        )
    return hits


def group_by_paper(hits: list[dict]) -> dict[str, list[dict]]:
    """按论文分组，保序；用于"多篇连结"的 prompt 组织与证据表展示。"""
    grouped: dict[str, list[dict]] = {}
    for hit in hits:
        grouped.setdefault(hit["paper_id"], []).append(hit)
    return grouped


def format_evidence_table(hits: list[dict]) -> str:
    """生成纯文本证据表（回答之外的"指出来源"部分）。"""
    lines = [
        f"{'#':>2}  {'论文':8s} {'章节':24s} {'页':>3s}  相似度",
        "-" * 72,
    ]
    for hit in hits:
        section = (hit.get("section") or "-")[:22]
        page = hit.get("page") if hit.get("page") is not None else "-"
        lines.append(
            f"{hit['rank']:>2}  {hit['paper_id']:8s} {section:24s} {str(page):>3s}  "
            f"{hit['score']:.3f}"
        )
    return "\n".join(lines)


def coverage(hits: list[dict]) -> dict:
    """覆盖率统计：跨了几篇论文、各贡献几条 —— 用于验证"多篇连结"是否成立。"""
    grouped = group_by_paper(hits)
    return {
        "n_hits": len(hits),
        "n_papers": len(grouped),
        "per_paper": {pid: len(v) for pid, v in grouped.items()},
    }
