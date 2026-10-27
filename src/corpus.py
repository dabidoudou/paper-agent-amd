"""语料读取：摘要 JSONL、评测问题集、切分后的 chunk 流。

`queries.txt` 采用与 `data.txt` 一致的块式格式：
    [Q03]
    type: hard
    question:
    第一行问题正文
    第二行继续（直到出现下一个 `键:` 或下一个 [Qxx]）

这样既人可读、又能用十几行解析器稳定读取，不引入 YAML/JSON 依赖。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from chunker import chunk_text, estimate_tokens

# 允许出现在 queries.txt 中的字段（顺序无关）
_QUERY_FIELDS = {"type", "lang", "expect_doc", "expect_topk", "question", "points"}

_BLOCK_RE = re.compile(r"(?m)^\[([A-Za-z0-9_\-]+)\]\s*$")
_KEY_RE = re.compile(r"^([a-z_]+):\s?(.*)$")


# ── 摘要 ────────────────────────────────────────────────────────────
def load_abstracts(path: str | Path) -> list[dict]:
    """读取 abstracts.jsonl，每条一行 JSON。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到摘要文件：{path}")
    papers: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                papers.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSON 解析失败：{exc}") from exc
    if not papers:
        raise ValueError(f"{path} 中没有读到任何记录")
    return papers


def paper_index(papers: Iterable[dict]) -> dict[str, dict]:
    """paper_id -> 论文记录。"""
    return {p["paper_id"]: p for p in papers}


# ── 评测问题 ────────────────────────────────────────────────────────
def _strip_comments(raw: str) -> str:
    """丢弃以 # 开头的整行注释（不处理行内 #，避免误伤正文）。"""
    return "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    )


def load_queries(path: str | Path) -> list[dict]:
    """解析 queries.txt，返回问题字典列表（按出现顺序）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到问题集：{path}")
    raw = _strip_comments(path.read_text(encoding="utf-8"))

    queries = _iter_blocks(raw)
    if not queries:
        raise ValueError(f"{path} 中没有解析到任何问题块")
    return queries


def _iter_blocks(raw: str) -> list[dict]:
    """把块式文本解析成字典列表。"""
    blocks: list[dict] = []
    parts = _BLOCK_RE.split(raw)
    # parts = ['前置内容', 'Q01', '正文', 'Q02', '正文', ...]
    for i in range(1, len(parts), 2):
        block_id, body = parts[i], parts[i + 1]
        record: dict = {"id": block_id}
        current_key: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal current_key, buffer
            if current_key is None:
                buffer = []
                return
            text = "\n".join(buffer).strip()
            record[current_key] = _coerce(text)
            buffer = []

        for line in body.splitlines():
            m = _KEY_RE.match(line)
            if m and m.group(1) in _QUERY_FIELDS:
                flush()
                current_key = m.group(1)
                buffer = [m.group(2)] if m.group(2).strip() else []
            elif current_key is not None:
                buffer.append(line)
        flush()
        blocks.append(record)
    return blocks


def _coerce(text: str):
    """把 'null' / '-' / 纯数字 转成合适的类型。"""
    stripped = text.strip()
    if stripped in ("", "null", "None", "-"):
        return None
    if stripped.isdigit():
        return int(stripped)
    return stripped


# ── chunk 流 ────────────────────────────────────────────────────────
def parse_paper_id(filename: str) -> str:
    """从文件名解析 paper_id：`P01_xxx.pdf` -> `P01`。

    命名约定见 data/papers/README.md。解析失败时回退为去掉扩展名的文件名。
    """
    stem = Path(filename).stem
    head = stem.split("_", 1)[0]
    if len(head) == 3 and head[0] in ("P", "U") and head[1:].isdigit():
        return head
    return stem


def build_chunks(
    papers: Iterable[dict],
    text_key: str = "abstract",
    target_tokens: int = 384,
    overlap_tokens: int = 48,
    count_tokens=None,
) -> list[dict]:
    """把每篇论文的指定字段切分成 chunk。

    之所以按论文逐条切而不是把所有摘要拼成一个大文档再切：保持 chunk 与论文的
    归属关系，检索命中后可以直接给出引用（[1] = DOC-xx / 标题），而不必事后反查。
    """
    counter = count_tokens or estimate_tokens
    chunks: list[dict] = []
    for paper in papers:
        text = paper.get(text_key) or ""
        pieces = chunk_text(
            text,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            count_tokens=counter,
        )
        for idx, piece in enumerate(pieces):
            chunks.append(
                {
                    "chunk_id": f"{paper['paper_id']}#{idx}",
                    "paper_id": paper["paper_id"],
                    "title": paper.get("title", ""),
                    "lang": paper.get("lang", ""),
                    "chunk_index": idx,
                    "n_chunks": len(pieces),
                    "tokens_est": counter(piece),
                    "text": piece,
                }
            )
    return chunks
