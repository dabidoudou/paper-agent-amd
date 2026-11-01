"""PDF 文本 -> 章节结构 -> 带溯源的 chunk。

全文 RAG 与摘要 RAG 的关键差异
------------------------------
1. **必须裁掉参考文献**。一篇论文末尾几十条引用会把大量无关标题塞进向量库，
   导致检索被"看起来很像"的引用条目污染。这是全文 RAG 最常见的质量事故。
2. **必须保留章节与页码**。用户要求"把相关论文内容指出来"，只有知道了
   §3.2 第 4 页，回答才能给出可核对的位置；只有纯粹按 384 token 平铺是不行的。
3. **必须先清洗噪声**。页眉页脚、页码、arXiv 水印、表格碎片都会被编码进向量。

本模块只处理"已经抽出来的文本"（`pages: list[str]`），不依赖 pymupdf，
因此可以在没有 PDF 库的环境里用构造样本做完整单测。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from chunker import chunk_text, estimate_tokens

# ── 噪声清洗规则 ────────────────────────────────────────────────────
_ARXIV_WM = re.compile(r"arXiv:\d{4}\.\d{4,5}(?:v\d+)?(?:\s*\[[^\]]{1,20}\])?(?:\s*\d{1,2}\s+\w+\s+\d{4})?", re.I)
_PAGE_NO = re.compile(r"^[\s\-\u2013\u2014.]*\d{1,4}[\s\-\u2013\u2014.]*$")
_MULTI_SPACE = re.compile(r"[ \t\u00a0]{2,}")


def _numericish(token: str) -> bool:
    """判断一个词是否"数值型"（含数字且几乎没有字母）。

    用于识别表格行：`PixArt-α 0.6B 74.97 97.32 …` 里除方法名外全是数值型词。
    """
    if not any(ch.isdigit() for ch in token):
        return False
    return sum(1 for ch in token if ch.isalpha()) <= 1


def is_table_noise(line: str, min_tokens: int = 6, ratio: float = 0.55) -> bool:
    """表格行判定：数值型词占比过高。

    阈值取 0.55 且要求至少 6 个词，避免误伤正常句子里夹带的数字
    （如 "ACC 平均提升 3.81%，ROUGE-L 提升 3.28%" 这类结论句）。
    """
    tokens = line.split()
    if len(tokens) < min_tokens:
        return False
    numeric = sum(1 for t in tokens if _numericish(t))
    return numeric / len(tokens) >= ratio


# ── 参考文献起点 ────────────────────────────────────────────────────
_REF_HEADS = re.compile(
    r"^\s*(?:\d{1,2}\.?\s*)?(references|bibliography|literature\s+cited|"
    r"参\s*考\s*文\s*献|引\s*用\s*文\s*献)\s*[:：]?\s*$",
    re.I,
)
# 宽松版：标题与首条引用挤在同一行（"References Akari Asai, …"）
_REF_HEADS_LOOSE = re.compile(
    r"^\s*(?:\d{1,2}\.?\s*)?(references|bibliography|参\s*考\s*文\s*献)\b",
    re.I,
)

# ── 章节标题识别 ────────────────────────────────────────────────────
_NUM_HEAD = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2}){0,2}\.?\s+\S.{0,60}$")
_KNOWN_NAMES = re.compile(
    r"^\s*(?:\d{1,2}(?:\.\d{1,2}){0,2}\.?\s*)?"
    r"(abstract|introduction|related\s+work|background|preliminar\w*|problem\s+\w+|"
    r"method\w*|approach|architecture|model|training|implementation|"
    r"experiment\w*|evaluation|setups?|dataset\w*|results?|ablation\w*|"
    r"analysis|discussion|conclusion\w*|limitation\w*|future\s+work|"
    r"appendix|acknowledg\w*|"
    r"摘\s*要|引\s*言|绪\s*论|相关工作|研究背景|背景|问题描述|方法|模型|"
    r"实验|结果|分析|讨论|结论|总结|展望|不足)\s*[:：]?\s*$",
    re.I,
)
_CJK = r"\u4e00-\u9fff"
_CJK_RE = re.compile(rf"[{_CJK}]")
_SENT_PUNCT = re.compile(r"[。！？；，,;!?]")


def _edge_key(line: str) -> str:
    """把页眉/页脚行归一化成可比较的键。

    真实论文的页眉往往带页码（"2 J. Singh and C. Merola" / "4 J. Singh and C. Merola"），
    直接按原文计数会把它们当成互不相同的行，导致模板噪声漏检。
    这里去掉首尾页码、把残余数字统一替换，再小写化。
    """
    s = re.sub(r"^[\s\W]{0,3}\d{1,4}\s+", "", line)   # 前导页码
    s = re.sub(r"\s+\d{1,4}[\s\W]{0,3}$", "", s)       # 尾部页码
    s = re.sub(r"\d+", "#", s)
    return _MULTI_SPACE.sub(" ", s).strip().lower()


@dataclass
class Block:
    """带页码的文本行。"""

    text: str
    page: int


@dataclass
class Section:
    """一个章节：名称 + 正文行 + 起止页。"""

    name: str
    index: int
    page_start: int = 1
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "page_start": self.page_start,
            "n_lines": len(self.lines),
            "chars": sum(len(line) for line in self.lines),
        }


# ── 1. 清洗 ─────────────────────────────────────────────────────────
def clean_pages(pages: list[str]) -> list[Block]:
    """把每页文本清洗成带页码的有效行。

    处理：arXiv 水印、纯页码行、重复页眉页脚、多余空白、过短碎行。
    """
    raw_pages: list[list[str]] = []
    for page_text in pages:
        lines = []
        for line in (page_text or "").splitlines():
            line = _ARXIV_WM.sub(" ", line)
            line = _MULTI_SPACE.sub(" ", line).strip()
            if len(line) < 2:
                continue
            if _PAGE_NO.match(line):
                continue
            if is_table_noise(line):
                continue          # 表格行：数值型词占比过高，会严重污染嵌入
            lines.append(line)
        raw_pages.append(lines)

    # 页眉/页脚检测：把边缘行归一化后再计数。
    # 真实论文的页眉每页都不同（"2 J. Singh" / "4 J. Singh"），
    # 只有把页码归一化掉才能把它们识别成同一行模板噪声。
    edge_counter: Counter[str] = Counter()
    nonempty_pages = [p for p in raw_pages if p]
    for lines in nonempty_pages:
        for edge in (lines[0], lines[-1]):
            if len(edge) <= 100:
                edge_counter[_edge_key(edge)] += 1
    # 阈值取 25%：15 页论文的页眉常只出现 5 次，用"一半页数"会漏掉
    threshold = max(2, int(round(len(nonempty_pages) * 0.25)))
    boilerplate = {key for key, n in edge_counter.items() if n >= threshold}

    blocks: list[Block] = []
    for pno, lines in enumerate(raw_pages, start=1):
        for idx, line in enumerate(lines):
            is_edge = idx == 0 or idx == len(lines) - 1
            if is_edge and _edge_key(line) in boilerplate:
                continue
            blocks.append(Block(text=line, page=pno))
    return blocks


# ── 2. 标题识别 ─────────────────────────────────────────────────────
def is_heading(line: str) -> bool:
    """判断一行是否为章节标题。

    策略偏保守：宁可漏判（内容仍会进入 body 章节）也不要错判（会把正文切碎）。
    真实 arXiv 论文里有两类高频误判，这里显式排除：
      1. 脚注 URL（"2 https://github.com/..."）会被编号标题规则命中
      2. 图表残留的孤立小写通用词（"model"）会被已知章节名规则命中
    """
    text = line.strip()
    # 下限取 2：中文标题常常只有两个字（「摘要」「结论」「方法」）
    if not (2 <= len(text) <= 70):
        return False
    if "://" in text or "www." in text.lower():
        return False

    # 数字/符号为主的短行不是标题（如图表轴标签 "1 2 ... 10"）
    cjk_all = len(_CJK_RE.findall(text))
    letters = sum(1 for ch in text if ch.isalpha())
    if letters < 3 and cjk_all < 2:
        return False

    known = _KNOWN_NAMES.match(text)
    if known:
        # 通用词只有首字母大写/全大写才算标题；小写的 "model"、"method" 是正文碎片
        head = known.group(0).strip()
        latin = re.match(r"[A-Za-z]", head.lstrip("0123456789. "))
        if latin is None or latin.group(0).isupper():
            return True
        return False

    if _SENT_PUNCT.search(text):
        return False
    if _NUM_HEAD.match(text):
        return True
    # 纯中文短标题（如「实验设置」「结果与分析」）
    cjk = len(_CJK_RE.findall(text))
    if cjk >= 2 and cjk == len(text) and cjk <= 12:
        return True
    return False


def is_references_heading(line: str) -> bool:
    text = line.strip()
    if _REF_HEADS.match(text):
        return True
    # 宽松匹配：参考文献标题常与首条引用挤在同一行（"References Akari Asai, …"）
    return bool(_REF_HEADS_LOOSE.match(text))


# ── 参考文献的兜底判定：引用密度 ────────────────────────────────────
_CITE_LINE = re.compile(
    r"(?:\(|,)\s*(?:19|20)\d{2}[a-z]?\)?"          # (2023) / , 2024
    r"|\bet\s+al\."
    r"|arXiv:\d{4}\.\d{4,5}"
    r"|Proceedings\s+of|Journal\s+of|Transactions\s+of"
    r"|In\s+Proceedings|Springer|ACM\s+Press|MIT\s+Press"
    r"|vol\.\s*\d+|pp\.\s*\d+",
    re.I,
)
CITATION_RUN_MIN = 6  # 连续多少行像引用，就认定参考文献从此开始


# ── 3. 分节 ─────────────────────────────────────────────────────────
def find_references_cut(blocks: list[Block]) -> int:
    """返回参考文献起始的行号；找不到返回 len(blocks)。

    两条判据：① 命中 References / 参考文献 标题行；
    ② 兜底——连续 CITATION_RUN_MIN 行都像引用条目。
    第二条是为真实论文准备的：不少 PDF 的 References 标题会和首条引用挤在同一行，
    纯行首匹配会漏检，导致整段引用被当成正文（实测 P02 就是这种情况）。
    """
    cite_run = 0
    cite_start = 0
    for idx, block in enumerate(blocks):
        line = block.text
        if is_references_heading(line):
            return idx
        if _CITE_LINE.search(line):
            if cite_run == 0:
                cite_start = idx
            cite_run += 1
            if cite_run >= CITATION_RUN_MIN:
                return cite_start
        else:
            cite_run = 0
    return len(blocks)


def split_sections(pages: list[str], cut_references: bool = True) -> list[Section]:
    """把清洗后的文本按标题切成章节；默认在参考文献处截断。

    Returns:
        Section 列表。若未识别到任何标题，返回单个名为 "body" 的章节。
    """
    blocks = clean_pages(pages)
    if cut_references:
        # 先定截断点，再分节：避免"边分节边回溯删除"带来的边界错误
        blocks = blocks[: find_references_cut(blocks)]

    sections: list[Section] = []
    current = Section(name="body", index=0, page_start=blocks[0].page if blocks else 1)

    for block in blocks:
        line = block.text
        if is_heading(line):
            if current.lines:
                sections.append(current)
            current = Section(name=line.strip(), index=len(sections), page_start=block.page)
            continue
        if not current.lines:
            current.page_start = block.page
        current.lines.append(line)

    if current.lines:
        sections.append(current)

    # 去掉过短的噪声章节（标题误判产生的孤立行）
    kept = [s for s in sections if sum(len(x) for x in s.lines) >= 40 or s.name == "body"]
    if not kept:
        kept = [Section(name="body", index=0, lines=[b.text for b in blocks])]
    for i, sec in enumerate(kept):
        sec.index = i
    return kept


# ── 4. 章节内切块（带溯源）──────────────────────────────────────────
def chunks_from_sections(
    sections: list[Section],
    paper_id: str,
    title: str = "",
    lang: str = "",
    target_tokens: int = 384,
    overlap_tokens: int = 48,
) -> list[dict]:
    """在章节内部切分，每个 chunk 带上章节名与页码。"""
    chunks: list[dict] = []
    for sec in sections:
        pieces = chunk_text(
            sec.text,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        for local_idx, piece in enumerate(pieces):
            chunks.append(
                {
                    "chunk_id": f"{paper_id}#s{sec.index}c{local_idx}",
                    "paper_id": paper_id,
                    "title": title,
                    "lang": lang,
                    "section": sec.name,
                    "section_index": sec.index,
                    "page": sec.page_start,
                    "chunk_index": local_idx,
                    "n_chunks": len(pieces),
                    "tokens_est": estimate_tokens(piece),
                    "text": piece,
                }
            )
    return chunks


def detect_lang(text: str) -> str:
    """按 CJK 字符占比判定语种。"""
    cjk = len(_CJK_RE.findall(text[:5000]))
    total = len(text[:5000]) or 1
    ratio = cjk / total
    if ratio > 0.25:
        return "zh"
    if ratio > 0.05:
        return "mixed"
    return "en"


def parse_paper(
    pages: list[str],
    paper_id: str,
    title: str = "",
    target_tokens: int = 384,
    overlap_tokens: int = 48,
    cut_references: bool = True,
) -> dict:
    """一次完成：清洗 -> 分节 -> 语言判定 -> 切块。

    Returns:
        {"paper_id", "title", "lang", "sections": [...], "chunks": [...], "stats": {...}}
    """
    sections = split_sections(pages, cut_references=cut_references)
    full_text = "\n".join(sec.text for sec in sections)
    lang = detect_lang(full_text)
    if not title:
        # 首页第一行通常就是标题
        cleaned = clean_pages(pages[:1])
        title = cleaned[0].text if cleaned else paper_id
    chunks = chunks_from_sections(
        sections,
        paper_id=paper_id,
        title=title,
        lang=lang,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )
    return {
        "paper_id": paper_id,
        "title": title,
        "lang": lang,
        "sections": [sec.to_dict() for sec in sections],
        "chunks": chunks,
        "fulltext": full_text,
        "stats": {
            "n_pages": len(pages),
            "n_sections": len(sections),
            "n_chunks": len(chunks),
            "chars": len(full_text),
            "tokens_est": sum(c["tokens_est"] for c in chunks),
            "references_cut": cut_references,
        },
    }
