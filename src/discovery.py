"""arXiv 关键词检索：把"关键词 -> 候选论文列表"这一步做出来。

为什么选 arXiv API
------------------
免费、无需 API key、无需注册，返回 Atom XML（标题 / 作者 / 年份 / 摘要 / PDF 直链），
且论文本身就是开放获取，PDF 下载不涉及额外授权。

两个工程细节
------------
1. **SSL 证书**：部分环境（企业代理 / 精简 Python 发行版）的默认 CA 路径不可用，
   表现为 `CERTIFICATE_VERIFY_FAILED`。这里显式加载 `certifi` 的证书包，
   在不降低安全性的前提下恢复正常校验。
2. **`parse_atom` 是纯函数**：网络相关逻辑与解析逻辑分离，
   因此可以用离线的 XML 片段做单测，不必每次都打网络。
"""

from __future__ import annotations

import json
import re
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ARXIV_API = "https://export.arxiv.org/api/query"
DEFAULT_TIMEOUT = 25
USER_AGENT = "paper-agent-amd/0.1 (course project; contact via github)"

_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_WS = re.compile(r"\s+")


def make_ssl_context() -> ssl.SSLContext:
    """优先使用 certifi 的 CA 包，失败则退回系统默认。"""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def build_search_query(keywords: str, field: str = "all") -> str:
    """把自由关键词转成 arXiv 查询串。

    多个关键词之间用 `+OR+` 连接（召回优先）；关键词整体加引号则按精确短语处理。
    注意：这里用 `quote` 而不是 `quote_plus`，否则空格的 `+` 会和 `+OR+` 分隔符混淆。
    """
    keywords = (keywords or "").strip()
    if not keywords:
        raise ValueError("关键词不能为空")
    if len(keywords) > 2 and keywords.startswith('"') and keywords.endswith('"'):
        return f'{field}:"{urllib.parse.quote(keywords[1:-1], safe="")}"'
    terms = [t for t in re.split(r"[\s,，;]+", keywords) if t]
    encoded = [f"{field}:{urllib.parse.quote(t, safe='')}" for t in terms]
    return "+OR+".join(encoded)


def parse_atom(xml_text: str) -> list[dict]:
    """解析 arXiv Atom 响应为候选列表（纯函数，可离线单测）。"""
    root = ET.fromstring(xml_text)
    results: list[dict] = []
    for entry in root.findall("atom:entry", _NS):
        raw_id = (entry.findtext("atom:id", default="", namespaces=_NS) or "").strip()
        m = re.search(r"abs/([0-9]{4}\.[0-9]{4,5})(v\d+)?", raw_id)
        arxiv_id = m.group(1) if m else raw_id.rsplit("/", 1)[-1]
        published = (entry.findtext("atom:published", default="", namespaces=_NS) or "")[:10]
        year = int(published[:4]) if published[:4].isdigit() else None

        pdf_url = ""
        for link in entry.findall("atom:link", _NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        results.append(
            {
                "source": "arxiv",
                "arxiv_id": arxiv_id,
                "title": _WS.sub(" ", entry.findtext("atom:title", default="", namespaces=_NS)).strip(),
                "abstract": _WS.sub(" ", entry.findtext("atom:summary", default="", namespaces=_NS)).strip(),
                "authors": [
                    (a.findtext("atom:name", default="", namespaces=_NS) or "").strip()
                    for a in entry.findall("atom:author", _NS)
                ],
                "published": published,
                "year": year,
                "categories": [c.get("term") for c in entry.findall("atom:category", _NS) if c.get("term")],
                "comment": _WS.sub(" ", entry.findtext("arxiv:comment", default="", namespaces=_NS)).strip(),
                "abs_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else raw_id,
                "pdf_url": pdf_url,
                "license": "arXiv-non-exclusive",
            }
        )
    return results


def search_arxiv(
    keywords: str,
    limit: int = 10,
    field: str = "all",
    sort_by: str = "relevance",
    timeout: int = DEFAULT_TIMEOUT,
    context: ssl.SSLContext | None = None,
) -> list[dict]:
    """按关键词检索 arXiv，返回候选论文列表。

    Args:
        keywords: 自由关键词，如 "retrieval augmented generation"。
        limit: 返回条数上限（arXiv 单次最多 2000，实际按需取值）。
        field: all / ti（标题）/ abs（摘要）/ au（作者）/ cat（分类）。
        sort_by: relevance（相关度）或 submittedDate（按提交时间）。
    """
    query = build_search_query(keywords, field=field)
    limit = max(1, min(int(limit), 100))
    # 手工拼 URL：query 已是 URL 安全形式，交给 urlencode 会把 +OR+ 二次编码成 %2B
    url = (
        f"{ARXIV_API}?search_query={query}&start=0&max_results={limit}"
        f"&sortBy={sort_by}&sortOrder=descending"
    )
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    ctx = context or make_ssl_context()
    with urllib.request.urlopen(request, timeout=timeout, context=ctx) as response:
        raw = response.read().decode("utf-8", "replace")
    return parse_atom(raw)


# ── 候选缓存 ────────────────────────────────────────────────────────
def save_candidates(results: list[dict], out_dir: str | Path, query: str) -> Path:
    """把一次检索结果写入 data/candidates/，供后续 `load` 挑选。

    只存元数据（标题/摘要/链接），不存 PDF —— 因此可以安全地进版本库。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", query).strip("-")[:40] or "query"
    path = out_dir / f"{stamp}_{slug}.json"
    payload = {
        "query": query,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_results": len(results),
        "results": results,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_candidates(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def latest_candidates(out_dir: str | Path) -> dict | None:
    """取最新一次检索缓存。"""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return None
    files = sorted(out_dir.glob("*.json"))
    if not files:
        return None
    return load_candidates(files[-1])


def format_candidate_list(results: list[dict], start: int = 1) -> str:
    """渲染成便于命令行挑选的列表。"""
    lines = [
        f"{'#':>3}  {'arXiv ID':14s} {'年':4s} 标题",
        "-" * 78,
    ]
    for i, item in enumerate(results, start=start):
        title = (item.get("title") or "")[:58]
        lines.append(f"{i:>3}  {item.get('arxiv_id', ''):14s} {str(item.get('year') or ''):4s} {title}")
    return "\n".join(lines)
