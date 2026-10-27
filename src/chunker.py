"""中英混合文本的规范化与切分。

设计要点
--------
1. **按 token 预算切分，而不是固定字符数**。中英混排时字符数与 token 数的比例差异很大
   （中文约 1 字 ≈ 1 token，英文约 4 字符 ≈ 1 token）。若按固定字符数切，
   同一批语料里的中文和英文会得到完全不同的粒度。
2. **没有 tokenizer 也能验证**。`chunk_text(..., count_tokens=fn)` 允许外部注入真实的
   分词计数器；不传时用 `estimate_tokens` 估算。这样本地无 torch 也能跑通切分逻辑，
   有 tokenizer 时切分结果更精确，但接口不变。
3. **`merge_cjk_spaces` 修复中文 PDF 抽取的字间空格**（"检 索 增 强" → "检索增强"）。
   这是 PyMuPDF / pdfplumber 处理中文双栏 PDF 的高频问题，不修会污染嵌入向量。
"""

from __future__ import annotations

import math
import re

# CJK 统一汉字 + 扩展 A + 日文假名 + 韩文音节
_CJK = r"\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af"
_FW_PUNCT = r"，。、；：！？（）《》〈〉「」『』“”‘’—…·"

_CJK_RE = re.compile(rf"[{_CJK}]")
_WS_RE = re.compile(r"\s+")

# 需要消除的空格：中-中、中-全角标点、全角标点-中、中-左括号、右括号-中
_CJK_SPACE_RE = re.compile(
    rf"(?<=[{_CJK}])\s+(?=[{_CJK}{_FW_PUNCT}])"
    rf"|(?<=[{_FW_PUNCT}])\s+(?=[{_CJK}])"
    rf"|(?<=[{_CJK}])\s+(?=[\(\（\[\【])"
    rf"|(?<=[\)\）\]\】])\s+(?=[{_CJK}])"
)

# 句子边界：中文句末标点后直接断；英文句末标点后需跟随空白才断（避免切断 3.81% 这类小数）
_SENT_RE = re.compile(r"(?<=[。！？；…])\s*|(?<=[.!?;])\s+")


# ── 规范化 ──────────────────────────────────────────────────────────
def merge_cjk_spaces(text: str) -> str:
    """删除中文字符之间多余的空格（PDF 抽取常见问题）。"""
    return _CJK_SPACE_RE.sub("", text or "")


def normalize_text(text: str) -> str:
    """把文本压平成单行并修复中文字间空格。

    摘要类短文本不需要保留换行，压平后解析和切分都更稳定。
    """
    if not text:
        return ""
    text = _WS_RE.sub(" ", text)
    text = merge_cjk_spaces(text)
    return text.strip()


# ── token 估算 ──────────────────────────────────────────────────────
def estimate_tokens(text: str) -> int:
    """无 tokenizer 时的 token 数估算。

    中文按 1 字符 1 token，其余（英文/数字/标点）按 4 字符 1 token。
    对 bge / e5 这类基于 XLM-R、BERT 的分词器，该估算误差通常在 ±15% 以内，
    足以决定「切不切」这个粒度判断。
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return int(cjk + math.ceil(other / 4.0))


# ── 断句 ────────────────────────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    """中英兼容断句，保留句末标点。"""
    return [p.strip() for p in _SENT_RE.split(text or "") if p.strip()]


def _join_sentences(sents: list[str]) -> str:
    """按语种决定拼接分隔符：中文不加空格，英文加空格。"""
    if not sents:
        return ""
    cjk = sum(len(_CJK_RE.findall(s)) for s in sents)
    total = sum(len(s) for s in sents) or 1
    sep = "" if cjk / total > 0.3 else " "
    return sep.join(sents)


def _hard_split(sentence: str, target_tokens: int, count_tokens) -> list[str]:
    """单句超过预算时按字符硬切（摘要里极少出现，但要保证不丢内容）。"""
    pieces: list[str] = []
    buf = ""
    for ch in sentence:
        buf += ch
        if count_tokens(buf) >= target_tokens:
            pieces.append(buf.strip())
            buf = ""
    if buf.strip():
        pieces.append(buf.strip())
    return [p for p in pieces if p]


def _overlap_tail(sents: list[str], overlap_tokens: int, count_tokens):
    """从上一块尾部取若干整句作为重叠，返回 (句子列表, token 数)。"""
    if overlap_tokens <= 0:
        return [], 0
    tail: list[str] = []
    tokens = 0
    for sent in reversed(sents):
        st = count_tokens(sent)
        if tokens + st > overlap_tokens:
            break
        tail.insert(0, sent)
        tokens += st
    return tail, tokens


# ── 主入口 ──────────────────────────────────────────────────────────
def chunk_text(
    text: str,
    target_tokens: int = 384,
    overlap_tokens: int = 48,
    count_tokens=None,
) -> list[str]:
    """把长文本切成语义相邻的块。

    Args:
        text: 原始文本（会自动规范化）。
        target_tokens: 每块的目标 token 上限。
        overlap_tokens: 相邻块之间的重叠 token 数，缓解跨块信息被切断。
        count_tokens: 可注入的真实分词计数器；None 时使用 `estimate_tokens`。

    Returns:
        块文本列表（顺序与原文一致，不含空块）。
    """
    count = count_tokens or estimate_tokens
    text = normalize_text(text)
    if not text:
        return []

    sents = split_sentences(text)
    if not sents:
        return [text]

    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0

    for sent in sents:
        st = count(sent)

        # 单句超预算：先把已积累的块收尾，再对这句硬切
        if st > target_tokens:
            if cur:
                chunks.append(_join_sentences(cur))
                cur, cur_tokens = [], 0
            chunks.extend(_hard_split(sent, target_tokens, count))
            continue

        # 放不下：收尾当前块，并保留尾部重叠
        if cur and cur_tokens + st > target_tokens:
            chunks.append(_join_sentences(cur))
            tail, tail_tokens = _overlap_tail(cur, overlap_tokens, count)
            # 重叠后放不下这句时，从重叠段最前面逐句剔除，尽量保留而非整体丢弃
            while tail and tail_tokens + st > target_tokens:
                tail_tokens -= count(tail.pop(0))
            cur, cur_tokens = tail, tail_tokens

        cur.append(sent)
        cur_tokens += st

    if cur:
        chunks.append(_join_sentences(cur))

    # 去掉可能的重复块与空块
    out: list[str] = []
    for c in chunks:
        c = c.strip()
        if c and (not out or c != out[-1]):
            out.append(c)
    return out
