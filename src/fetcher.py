"""下载 PDF、抽取页面文本，并把论文写入论文库。

依赖策略
--------
PDF 文本抽取优先用 `pymupdf`（单轮子、双栏重排好、CJK 友好），装不上则回退 `pypdf`。
两者都是惰性导入，因此本模块在没有任何 PDF 库的环境里仍可 import，
只有在真正调用 `extract_pages` 时才会报缺失依赖。

论文 ID 约定
------------
- 由关键词检索入库：`P01`, `P02`, ...（P = discovered Paper）
- 手工添加：`U01`, `U02`, ...（U = User added）

两者走**完全相同的**抽取与切块流程，区别只在编号来源 —— 这样"自己添加论文"
和"检索到的论文"在后续检索中没有任何差别对待。
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

from discovery import USER_AGENT, make_ssl_context
from fulltext import parse_paper
from library import Library


class FetcherError(RuntimeError):
    """下载或解析失败。"""


# ── 下载 ────────────────────────────────────────────────────────────
def download_pdf(url: str, timeout: int = 60, retries: int = 2) -> bytes:
    """下载 PDF 字节流。带重试，失败抛出可读错误。"""
    if not url:
        raise FetcherError("PDF 链接为空")
    ctx = make_ssl_context()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout, context=ctx) as resp:
                data = resp.read()
            if not data.startswith(b"%PDF"):
                raise FetcherError(f"返回内容不是 PDF（前 8 字节：{data[:8]!r}）：{url}")
            return data
        except Exception as exc:  # 网络抖动 / 429 限流都走重试
            last_error = exc
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise FetcherError(f"下载失败：{url}（{type(last_error).__name__}: {last_error}）")


# ── 文本抽取 ────────────────────────────────────────────────────────
def extract_pages(pdf_path: str | Path) -> list[str]:
    """把 PDF 抽成「每页一个字符串」。

    `sort=True` 让 PyMuPDF 按版面重排，双栏论文的阅读顺序才不会串行。
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FetcherError(f"PDF 不存在：{pdf_path}")

    try:
        import pymupdf  # 新版包名
        backend = "pymupdf"
    except ImportError:
        try:
            import fitz as pymupdf  # 旧版包名
            backend = "fitz"
        except ImportError:
            pymupdf = None
            backend = ""

    if pymupdf is not None:
        doc = pymupdf.open(str(pdf_path))
        try:
            return [page.get_text("text", sort=True) or "" for page in doc]
        finally:
            doc.close()

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise FetcherError(
            "缺少 PDF 解析库：请 pip install pymupdf（推荐）或 pypdf"
        ) from exc
    reader = PdfReader(str(pdf_path))
    return [(page.extract_text() or "") for page in reader.pages]


# ── 论文 ID 分配 ────────────────────────────────────────────────────
def next_paper_id(library: Library, prefix: str = "P") -> str:
    """按前缀生成下一个可用编号。"""
    used = set()
    for pid in library.ids():
        if pid.startswith(prefix) and pid[1:].isdigit():
            used.add(int(pid[1:]))
    n = 1
    while n in used:
        n += 1
    return f"{prefix}{n:02d}"


# ── 入库主流程 ──────────────────────────────────────────────────────
def load_pdf_bytes(
    library: Library,
    pdf_bytes: bytes,
    paper_id: str,
    meta: dict | None = None,
    target_tokens: int = 384,
    overlap_tokens: int = 48,
) -> dict:
    """把 PDF 字节流解析成全文/章节/chunk 并写入论文库。

    这是"检索到的论文"和"手工添加的论文"共用的唯一入库路径 ——
    保证两者后续行为完全一致（同一份代码、同一套参数）。
    """
    meta = dict(meta or {})
    path, sha1 = library.save_pdf(paper_id, pdf_bytes)
    pages = extract_pages(path)
    if not pages or not any(p.strip() for p in pages):
        raise FetcherError(
            f"{paper_id} 抽取不到任何文本，可能是扫描版 PDF（需 OCR，本项目不处理）"
        )

    parsed = parse_paper(
        pages,
        paper_id=paper_id,
        title=meta.get("title") or "",
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )
    library.save_fulltext(paper_id, parsed["fulltext"])
    library.save_sections(paper_id, parsed["sections"])
    library.save_chunks(paper_id, parsed["chunks"])

    record = library.add(
        paper_id,
        {
            **meta,
            "title": parsed["title"],
            "lang": meta.get("lang") or parsed["lang"],
            "pdf_sha1": sha1,
            "n_pages": parsed["stats"]["n_pages"],
            "n_sections": parsed["stats"]["n_sections"],
            "n_chunks": parsed["stats"]["n_chunks"],
            "status": "loaded",
        },
    )
    return {"record": record, "stats": parsed["stats"], "sections": parsed["sections"]}


def load_from_candidate(
    library: Library,
    candidate: dict,
    paper_id: str | None = None,
    target_tokens: int = 384,
    overlap_tokens: int = 48,
) -> dict:
    """从 arXiv 候选条目下载并入库。"""
    paper_id = paper_id or next_paper_id(library, prefix="P")
    pdf_bytes = download_pdf(candidate.get("pdf_url", ""))
    meta = {
        "title": candidate.get("title", ""),
        "authors": candidate.get("authors", []),
        "year": candidate.get("year"),
        "source": candidate.get("source", "arxiv"),
        "source_url": candidate.get("abs_url", ""),
        "license": candidate.get("license", "arXiv-non-exclusive"),
        "added_from": "discovery",
    }
    return load_pdf_bytes(
        library,
        pdf_bytes,
        paper_id,
        meta=meta,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )


def add_local_pdf(
    library: Library,
    pdf_path: str | Path,
    paper_id: str | None = None,
    meta: dict | None = None,
    target_tokens: int = 384,
    overlap_tokens: int = 48,
) -> dict:
    """手工添加本地 PDF（U 前缀）。与检索入库走同一条处理流程。"""
    pdf_path = Path(pdf_path)
    paper_id = paper_id or next_paper_id(library, prefix="U")
    info = dict(meta or {})
    info.setdefault("added_from", "manual")
    info.setdefault("source", "local")
    info.setdefault("title", "")
    return load_pdf_bytes(
        library,
        pdf_path.read_bytes(),
        paper_id,
        meta=info,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )
