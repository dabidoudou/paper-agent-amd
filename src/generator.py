"""本地 Qwen2.5 生成器 + Prompt 构造（含多论文综合与证据溯源）。

三个层次
--------
1. `build_messages` / `build_citations` / `fallback_prompt` 是**纯函数**，不依赖 torch，
   因此可以在没有 GPU 的机器上单测 prompt 内容、引用编号、跨论文分组是否正确。
2. `QwenGenerator.load/generate` 才需要 torch，且惰性导入，使模块本身可被无依赖 import。
3. `QwenGenerator.answer` 是给 agent 用的便捷封装：一次拿到 prompt、回答与 token 统计。

为什么 prompt 要按论文分组
--------------------------
用户要求"对多篇内容可以连结起来"。如果把检索到的证据混在一起平铺，1.5B 模型往往会
只顺着第一条证据回答，跨论文关系就丢了。按论文分组并在指令里明确要求指出
「一致 / 互补 / 冲突」关系，是把"多篇连结"从祈祷变成约束。

计时要点
--------
`ttft_s`（首 token 延迟）用 `TextIteratorStreamer` 单独测；decode 速度按
「输出 token / (总耗时 - TTFT)」算，避免把 prefill 算进解码速度。
"""

from __future__ import annotations

import threading
import time

# ── Prompt 模板 ─────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "你是一个严谨的学术问答助手。你只能依据下面提供的【资料】回答问题，"
    "禁止编造资料之外的任何信息。如果资料中没有答案，请直接回答"
    "「根据提供的资料无法回答」，不要猜测或补充。"
    "回答末尾用 [编号] 标注所依据的资料。"
)

SYNTHESIS_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + "当资料来自多篇论文时，你必须显式说明它们之间的关系："
    "结论一致、互为补充、还是存在差异；并指出每篇论文各自解决了什么。"
)

_ANSWER_RULES = (
    "【要求】\n"
    "1. 只依据上述资料作答，不要引入资料之外的知识。\n"
    "2. 每条关键结论后用 [编号] 标注依据。\n"
    "3. 若资料覆盖多篇论文，请明确指出它们之间的一致、互补或差异关系。\n"
    "4. 若资料不足以回答，直接回答「根据提供的资料无法回答」，不要推测。"
)


def _hit_label(hit: dict) -> str:
    """证据的来源标注：论文编号 + 章节 + 页码。"""
    parts = [hit.get("paper_id", "?")]
    section = (hit.get("section") or "").strip()
    if section:
        parts.append(f"§{section}")
    if hit.get("page") is not None:
        parts.append(f"p.{hit['page']}")
    return " ".join(parts)


def build_context(hits: list[dict]) -> str:
    """把证据按论文分组拼成资料块。

    每条形如 `[1] (P06 / 标题) §2 Method p.3` + 正文，
    既保证模型知道来源，也方便前端/日志按论文聚合。
    """
    if not hits:
        return "（无相关资料）"

    grouped: dict[str, list[dict]] = {}
    for hit in hits:
        grouped.setdefault(hit.get("paper_id", "?"), []).append(hit)

    blocks: list[str] = [f"共 {len(grouped)} 篇论文 / {len(hits)} 条证据"]
    for pid, items in grouped.items():
        title = items[0].get("title", "")
        blocks.append(f"\n■ 论文 {pid} 《{title}》")
        for hit in items:
            rank = hit.get("rank", "?")
            blocks.append(f"[{rank}] ({pid} / {title}) {_hit_label(hit)}")
            blocks.append(hit.get("text", ""))
    return "\n".join(blocks)


def build_messages(
    question: str,
    hits: list[dict],
    system_prompt: str | None = None,
) -> list[dict]:
    """构造 chat messages（纯函数，可单测）。"""
    n_papers = len({h.get("paper_id") for h in hits})
    system = system_prompt or (SYNTHESIS_SYSTEM_PROMPT if n_papers > 1 else SYSTEM_PROMPT)
    user = (
        f"【资料】\n{build_context(hits)}\n\n"
        f"{_ANSWER_RULES}\n\n"
        f"【问题】{question.strip()}\n"
        f"【回答】"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_citations(hits: list[dict], excerpt_chars: int = 90) -> list[dict]:
    """生成结构化证据列表 —— 对应需求里的"把相关论文内容指出来"。"""
    citations = []
    for hit in hits:
        text = (hit.get("text") or "").strip().replace("\n", " ")
        citations.append(
            {
                "rank": hit.get("rank"),
                "paper_id": hit.get("paper_id"),
                "title": hit.get("title", ""),
                "section": hit.get("section", ""),
                "page": hit.get("page"),
                "score": hit.get("score"),
                "chunk_id": hit.get("chunk_id"),
                "excerpt": text[:excerpt_chars] + ("…" if len(text) > excerpt_chars else ""),
            }
        )
    return citations


def format_citations(citations: list[dict]) -> str:
    """把证据列表渲染成可读的文本表（命令行 / 报告截图用）。"""
    if not citations:
        return "（无引用）"
    lines = ["", "—— 引用来源 ——"]
    for c in citations:
        loc = c["paper_id"]
        if c.get("section"):
            loc += f" §{c['section']}"
        if c.get("page") is not None:
            loc += f" p.{c['page']}"
        lines.append(f"[{c['rank']}] {c['title']}  ({loc})")
        lines.append(f"     {c['excerpt']}")
    return "\n".join(lines)


def fallback_prompt(messages: list[dict]) -> str:
    """当 tokenizer 没有 chat template 时的纯文本降级方案（可单测）。"""
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content'].strip()}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ── 生成器 ──────────────────────────────────────────────────────────
class QwenGenerator:
    """本地 Qwen2.5-Instruct 推理封装（惰性导入 torch）。"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        dtype: str = "bfloat16",
        device: str | None = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
    ):
        self.model_name = model_name
        self.dtype_name = dtype
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self._torch = None
        self._model = None
        self._tokenizer = None

    # ── 加载 ────────────────────────────────────────────────────────
    def load(self) -> dict:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype_name]

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch_dtype
        )
        self._model.to(self.device).eval()

        n_params = sum(p.numel() for p in self._model.parameters())
        return {
            "gen_model": self.model_name,
            "gen_device": self.device,
            "gen_dtype": self.dtype_name,
            "gen_n_params": n_params,
            "gen_weight_gb": round(n_params * self._bytes_per_param(), 2),
        }

    def _bytes_per_param(self) -> float:
        return {"float32": 4.0, "float16": 2.0, "bfloat16": 2.0}.get(self.dtype_name, 2.0)

    # ── 渲染 ────────────────────────────────────────────────────────
    def render(self, messages: list[dict]) -> str:
        """应用 chat template；没有模板时退回手写格式。"""
        tok = self._tokenizer
        if tok is not None and getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return fallback_prompt(messages)

    # ── 推理 ────────────────────────────────────────────────────────
    def generate(
        self,
        messages_or_prompt,
        max_new_tokens: int | None = None,
        use_stream: bool = True,
    ) -> dict:
        """生成回答，并返回计时与 token 统计。"""
        if self._model is None:
            raise RuntimeError("模型未加载，请先调用 load()")
        torch = self._torch
        prompt = (
            messages_or_prompt
            if isinstance(messages_or_prompt, str)
            else self.render(messages_or_prompt)
        )
        enc = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = int(enc["input_ids"].shape[1])
        max_new = int(max_new_tokens or self.max_new_tokens)

        # pad_token_id 可能为 0，不能用 `or` 兜底，必须显式判断 None
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id

        gen_kwargs = dict(
            max_new_tokens=max_new,
            do_sample=self.do_sample,
            pad_token_id=pad_id,
        )

        ttft: float | None = None
        t0 = time.perf_counter()

        if use_stream:
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(
                self._tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            errors: dict = {}

            def _worker() -> None:
                try:
                    with torch.no_grad():
                        self._model.generate(**enc, streamer=streamer, **gen_kwargs)
                except Exception as exc:  # 线程内异常需带回主线程
                    errors["exc"] = exc

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()
            pieces: list[str] = []
            for piece in streamer:
                if ttft is None and piece.strip():
                    ttft = time.perf_counter() - t0
                pieces.append(piece)
            thread.join()
            if errors:
                raise RuntimeError("生成失败") from errors["exc"]
            text = "".join(pieces).strip()
        else:
            with torch.no_grad():
                out = self._model.generate(**enc, **gen_kwargs)
            text = self._tokenizer.decode(
                out[0][prompt_tokens:], skip_special_tokens=True
            ).strip()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total = time.perf_counter() - t0

        out_tokens = len(self._tokenizer(text).input_ids) if text else 0
        decode_s = max(total - (ttft or 0.0), 1e-9)
        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "out_tokens": out_tokens,
            "ttft_s": round(ttft, 4) if ttft is not None else None,
            "total_s": round(total, 4),
            "decode_s": round(decode_s, 4),
            "tok_per_s": round(out_tokens / decode_s, 2) if out_tokens else 0.0,
        }

    def answer(self, question: str, hits: list[dict], **gen_kwargs) -> dict:
        """给 agent 用：一次性返回 prompt、回答与引用。"""
        messages = build_messages(question, hits)
        result = self.generate(messages, **gen_kwargs)
        citations = build_citations(hits)
        result["answer"] = result.get("text", "")
        result["citations"] = citations
        result["n_papers"] = len({h.get("paper_id") for h in hits})
        return result
