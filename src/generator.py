"""本地 Qwen2.5 生成器 + 纯字符串的 prompt 构造。

分层设计
--------
`build_messages` / `fallback_prompt` 是**纯函数**，不依赖 torch，因此可以在没有 GPU 的
机器上做单元测试（本地 conda 环境即可验证 prompt 内容、引用编号、拒答指令是否正确）。
只有 `QwenGenerator.load/generate` 才需要 torch + transformers，且采用惰性导入，
使得整个模块在没有安装 torch 的环境里也能被 import。

计时要点
--------
- `ttft_s`（首 token 延迟）通过 `TextIteratorStreamer` 单独测，否则 prefill 和 decode
  的耗时会糊在一起，看不出瓶颈在哪。
- decode 速度按「输出 token 数 / (总耗时 - TTFT)」计算，避免把 prefill 算进解码速度。
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


def build_context(hits: list[dict]) -> str:
    """把检索结果拼成带编号的资料块。

    hits 每项需含 rank / paper_id / title / text。
    """
    blocks = []
    for h in hits:
        rank = h.get("rank", h.get("_rank", "?"))
        blocks.append(
            f"[{rank}] ({h['paper_id']} / {h['title']})\n{h['text']}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[dict], system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    """构造 chat messages（纯函数，可单测）。"""
    user = (
        f"【资料】\n{build_context(hits)}\n\n"
        f"【问题】{question.strip()}\n"
        f"【回答】"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def fallback_prompt(messages: list[dict]) -> str:
    """当 tokenizer 没有 chat template 时的纯文本降级方案（可单测）。"""
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"].strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
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
            self.model_name,
            torch_dtype=torch_dtype,
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
    def generate(self, messages_or_prompt, max_new_tokens: int | None = None,
                 use_stream: bool = True) -> dict:
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
