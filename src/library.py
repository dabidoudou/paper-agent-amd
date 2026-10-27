"""论文库：注册表入库，正文不入库。

存储结构
--------
    data/library/
    ├── library.json        # ✅ 提交到 git：纯元数据注册表
    ├── P01/
    │   ├── paper.pdf       # 🚫 不入库（版权）
    │   ├── fulltext.txt    # 🚫 不入库
    │   ├── sections.json   # 🚫 不入库
    │   └── chunks.jsonl    # 🚫 不入库
    └── U01/ ...

为什么这样切
------------
摘要属引文性质可以公开，**全文是完整的受版权保护的表达**，不能提交到公开仓库。
所以仓库里只留"有哪些论文"的注册表，正文由 `fetch_papers.py` 从 arXiv/ACL 重建。
`library.json` 里记录 sha1 与来源链接，保证重建结果可校验。

论文"单独存储"的含义
--------------------
每篇论文一个目录 + 一条注册表记录：加论文 = 建目录 + 追加一行；删论文 = 删目录 + 删一行。
两者互不影响，不需要重建整个库。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

REGISTRY_VERSION = 1

ARTIFACT_PDF = "paper.pdf"
ARTIFACT_FULLTEXT = "fulltext.txt"
ARTIFACT_SECTIONS = "sections.json"
ARTIFACT_CHUNKS = "chunks.jsonl"


class Library:
    """论文库注册表 + 单篇论文目录的管理器。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "library.json"
        self._data = self._load_registry()

    # ── 注册表读写 ──────────────────────────────────────────────────
    def _load_registry(self) -> dict:
        if not self.registry_path.exists():
            return {"version": REGISTRY_VERSION, "papers": {}}
        raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        raw.setdefault("version", REGISTRY_VERSION)
        raw.setdefault("papers", {})
        return raw

    def save(self) -> Path:
        """原子写入，避免中途崩溃把注册表写坏。"""
        payload = json.dumps(self._data, ensure_ascii=False, indent=2)
        tmp = self.registry_path.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.registry_path)
        return self.registry_path

    # ── 查询 ────────────────────────────────────────────────────────
    @property
    def papers(self) -> dict[str, dict]:
        return self._data["papers"]

    def ids(self) -> list[str]:
        return sorted(self.papers)

    def __len__(self) -> int:
        return len(self.papers)

    def __contains__(self, paper_id: str) -> bool:
        return paper_id in self.papers

    def get(self, paper_id: str) -> dict | None:
        return self.papers.get(paper_id)

    def paper_dir(self, paper_id: str) -> Path:
        return self.root / paper_id

    def stats(self) -> dict:
        papers = list(self.papers.values())
        return {
            "n_papers": len(papers),
            "n_chunks": sum(int(p.get("n_chunks") or 0) for p in papers),
            "n_zh": sum(1 for p in papers if p.get("lang") == "zh"),
            "n_en": sum(1 for p in papers if p.get("lang") == "en"),
        }

    # ── 增删 ────────────────────────────────────────────────────────
    def add(self, paper_id: str, meta: dict) -> dict:
        """注册一篇论文（只登记元数据，产物由调用方写入目录）。"""
        previous = self.papers.get(paper_id, {})
        record = {
            "paper_id": paper_id,
            "title": meta.get("title", previous.get("title", "")),
            "authors": meta.get("authors", previous.get("authors", [])),
            "year": meta.get("year", previous.get("year")),
            "venue": meta.get("venue", previous.get("venue", "")),
            "lang": meta.get("lang", previous.get("lang", "")),
            "source": meta.get("source", previous.get("source", "")),
            "source_url": meta.get("source_url", previous.get("source_url", "")),
            "license": meta.get("license", previous.get("license", "")),
            "added_from": meta.get("added_from", previous.get("added_from", "manual")),
            "pdf_sha1": meta.get("pdf_sha1", previous.get("pdf_sha1")),
            "n_pages": meta.get("n_pages", previous.get("n_pages")),
            "n_sections": meta.get("n_sections", previous.get("n_sections")),
            "n_chunks": meta.get("n_chunks", previous.get("n_chunks")),
            "status": meta.get("status", previous.get("status", "loaded")),
            "added_at": previous.get("added_at") or datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.papers[paper_id] = record
        self.paper_dir(paper_id).mkdir(parents=True, exist_ok=True)
        return record

    def remove(self, paper_id: str, purge: bool = True) -> bool:
        """删除注册记录；purge=True 时同时删除磁盘产物。"""
        if paper_id not in self.papers:
            return False
        del self.papers[paper_id]
        if purge:
            target = self.paper_dir(paper_id)
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
        return True

    # ── 产物读写 ────────────────────────────────────────────────────
    def save_pdf(self, paper_id: str, data: bytes) -> tuple[Path, str]:
        """保存 PDF 并返回 (路径, sha1)。sha1 用于校验重建结果。"""
        path = self.paper_dir(paper_id) / ARTIFACT_PDF
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path, hashlib.sha1(data).hexdigest()

    def pdf_path(self, paper_id: str) -> Path:
        return self.paper_dir(paper_id) / ARTIFACT_PDF

    def save_fulltext(self, paper_id: str, text: str) -> Path:
        path = self.paper_dir(paper_id) / ARTIFACT_FULLTEXT
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def save_sections(self, paper_id: str, sections: list[dict]) -> Path:
        path = self.paper_dir(paper_id) / ARTIFACT_SECTIONS
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def save_chunks(self, paper_id: str, chunks: list[dict]) -> Path:
        path = self.paper_dir(paper_id) / ARTIFACT_CHUNKS
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        return path

    def load_chunks(self, paper_id: str | None = None) -> list[dict]:
        """读取 chunk。paper_id=None 时读取全库。"""
        targets = [paper_id] if paper_id else self.ids()
        chunks: list[dict] = []
        for pid in targets:
            path = self.paper_dir(pid) / ARTIFACT_CHUNKS
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
        return chunks

    def load_sections(self, paper_id: str) -> list[dict]:
        path = self.paper_dir(paper_id) / ARTIFACT_SECTIONS
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def is_complete(self, paper_id: str) -> bool:
        """论文产物是否齐全（可参与检索）。"""
        d = self.paper_dir(paper_id)
        return all(
            (d / name).exists()
            for name in (ARTIFACT_FULLTEXT, ARTIFACT_SECTIONS, ARTIFACT_CHUNKS)
        )
