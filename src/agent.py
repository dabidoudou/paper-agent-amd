"""论文智能体：确定性编排的五步工作流。

    search  关键词 -> arXiv 候选列表
    load    挑选候选 -> 下载 PDF -> 全文分节 -> 写入论文库
    index   全库 chunk 向量化 -> 建索引
    ask     问题 -> 多样化检索 -> 多论文综合回答 + 证据表
    list    查看论文库 / 候选缓存

为什么是"确定性编排"而不是工具调用
----------------------------------
1.5B 参数的模型在 function calling 上不可靠（容易漏参数、死循环、把 JSON 写坏）。
把"该调哪个工具"交给代码、把"该说什么话"交给模型，稳定性高一个档次，
而且每一步都能单独计时与复现 —— 这对要交算力数据的课程实践更重要。
报告里会如实写作「工作流智能体（workflow agent）」。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from config import DATA_DIR, GEN_DTYPE, GEN_MODEL
from discovery import (
    format_candidate_list,
    latest_candidates,
    load_candidates,
    save_candidates,
    search_arxiv,
)
from embedder import build_embedder
from fetcher import add_local_pdf, load_from_candidate
from generator import build_citations, build_messages, fallback_prompt, format_citations
from library import Library
from retrieval import coverage, format_evidence_table, retrieve
from vector_store import VectorStore

INDEX_CHUNKS = "chunks.jsonl"
INDEX_VECTORS = "vectors.npz"
INDEX_META = "meta.json"

DEFAULT_K = 6
DEFAULT_PER_PAPER_CAP = 3
DEFAULT_MIN_PAPERS = 2


class AgentError(RuntimeError):
    """工作流前置条件不满足（库为空、索引缺失等）。"""


class PaperAgent:
    """论文智能体的编排核心。"""

    def __init__(
        self,
        data_dir: str | Path = DATA_DIR,
        embedder: str = "bert",
        dry_run: bool = False,
        gen_model: str = GEN_MODEL,
        dtype: str = GEN_DTYPE,
        device: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.library = Library(self.data_dir / "library")
        self.candidates_dir = self.data_dir / "candidates"
        self.index_dir = self.data_dir / "index"
        self.embedder_kind = embedder
        self.dry_run = dry_run
        self.gen_model = gen_model
        self.dtype = dtype
        self.device = device

        self._embedder = None
        self._generator = None
        self._chunks: list[dict] | None = None
        self._vectors: np.ndarray | None = None

    # ── 惰性资源 ────────────────────────────────────────────────────
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = build_embedder(self.embedder_kind)
            self._embedder.load()
        return self._embedder

    @property
    def generator(self):
        if self.dry_run:
            raise AgentError("dry-run 模式下不加载生成模型")
        if self._generator is None:
            from generator import QwenGenerator

            self._generator = QwenGenerator(
                model_name=self.gen_model, dtype=self.dtype, device=self.device
            )
            self._generator.load()
        return self._generator

    # ── ① 发现 ──────────────────────────────────────────────────────
    def search(self, keywords: str, limit: int = 10, save: bool = True) -> dict:
        results = search_arxiv(keywords, limit=limit)
        path = save_candidates(results, self.candidates_dir, keywords) if save else None
        return {"query": keywords, "n_results": len(results), "results": results, "path": path}

    def last_candidates(self) -> dict | None:
        return latest_candidates(self.candidates_dir)

    # ── ② 入库 ──────────────────────────────────────────────────────
    def load(self, refs: list[str], candidate_file: str | Path | None = None) -> list[dict]:
        """refs 可以是候选列表里的序号（"1"）或 arXiv ID（"2402.03216"）。"""
        payload = (
            load_candidates(candidate_file) if candidate_file else self.last_candidates()
        )
        if not payload:
            raise AgentError("没有候选缓存，请先运行 search")
        picked = _pick_candidates(payload.get("results", []), refs)
        if not picked:
            raise AgentError(f"没有匹配到候选：{refs}")

        known = {p.get("arxiv_id") for p in self.library.papers.values()}
        loaded: list[dict] = []
        for cand in picked:
            if cand.get("arxiv_id") in known:
                print(f"  跳过 {cand['arxiv_id']}（已在库中）")
                continue
            try:
                result = load_from_candidate(self.library, cand)
            except Exception as exc:
                print(f"  失败 {cand.get('arxiv_id')}: {exc}")
                continue
            self.library.papers[result["record"]["paper_id"]]["arxiv_id"] = cand.get("arxiv_id")
            loaded.append(result)
        self.library.save()
        self._chunks = None
        self._vectors = None
        return loaded

    def add_local(self, pdf_path: str | Path, paper_id: str | None = None) -> dict:
        result = add_local_pdf(self.library, pdf_path, paper_id=paper_id)
        self.library.save()
        self._chunks = None
        self._vectors = None
        return result

    def remove_paper(self, paper_id: str) -> bool:
        ok = self.library.remove(paper_id)
        if ok:
            self.library.save()
            self._chunks = None
            self._vectors = None
        return ok

    def list_papers(self) -> list[dict]:
        return [self.library.papers[pid] for pid in self.library.ids()]

    # ── ③ 建索引 ────────────────────────────────────────────────────
    def build_index(self) -> dict:
        chunks = self.library.load_chunks()
        if not chunks:
            raise AgentError("论文库为空或尚未抽取，请先 load 论文")

        started = time.perf_counter()
        vectors = self.embedder.encode(
            [c["text"] for c in chunks], kind="passage", batch_size=16
        )
        embed_s = time.perf_counter() - started

        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.savez(str(self.index_dir / INDEX_VECTORS), vectors=vectors)
        with (self.index_dir / INDEX_CHUNKS).open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        store = VectorStore(vectors)
        meta = {
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "embedder": self.embedder_kind,
            "n_papers": len({c["paper_id"] for c in chunks}),
            "n_chunks": len(chunks),
            "dim": int(vectors.shape[1]),
            "backend": store.backend,
            "index_bytes": store.n_bytes,
            "embed_seconds": round(embed_s, 3),
            "chunks_per_second": round(len(chunks) / embed_s, 1) if embed_s > 0 else None,
        }
        (self.index_dir / INDEX_META).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._chunks, self._vectors = chunks, vectors
        return meta

    def _ensure_index(self) -> None:
        if self._chunks is not None and self._vectors is not None:
            return
        vectors_path = self.index_dir / INDEX_VECTORS
        chunks_path = self.index_dir / INDEX_CHUNKS
        if not vectors_path.exists() or not chunks_path.exists():
            self.build_index()
            return

        with chunks_path.open(encoding="utf-8") as fh:
            chunks = [json.loads(line) for line in fh if line.strip()]
        vectors = np.load(str(vectors_path))["vectors"]

        # 库发生变化（新增/删除论文）后索引会过期，按 chunk 数判断
        current = len(self.library.load_chunks())
        if len(chunks) != current:
            self.build_index()
            return
        self._chunks, self._vectors = chunks, vectors

    # ── ④ 问答 ──────────────────────────────────────────────────────
    def ask(
        self,
        question: str,
        k: int = DEFAULT_K,
        per_paper_cap: int = DEFAULT_PER_PAPER_CAP,
        min_papers: int = DEFAULT_MIN_PAPERS,
    ) -> dict:
        self._ensure_index()
        started = time.perf_counter()
        query_vec = self.embedder.encode([question], kind="query")
        retrieve_s = time.perf_counter() - started

        hits = retrieve(
            self._chunks,
            self._vectors,
            query_vec,
            k=k,
            per_paper_cap=per_paper_cap,
            min_papers=min_papers,
        )
        result = {
            "question": question,
            "n_hits": len(hits),
            "hits": hits,
            "coverage": coverage(hits),
            "retrieve_seconds": round(retrieve_s, 4),
            "citations": build_citations(hits),
            "evidence_table": format_evidence_table(hits),
            "answer": None,
        }

        if self.dry_run:
            # 不加载模型，但仍把发给模型的 prompt 打印出来，便于校验内容与长度
            messages = build_messages(question, hits)
            result["prompt_preview"] = fallback_prompt(messages)[:2000]
            result["prompt_chars"] = len(fallback_prompt(messages))
            return result

        gen_result = self.generator.answer(question, hits)
        result.update(gen_result)
        return result


# ── 候选挑选 ────────────────────────────────────────────────────────
def _pick_candidates(results: list[dict], refs: list[str]) -> list[dict]:
    """支持用列表序号或 arXiv ID 指定要入库的论文。"""
    picked: list[dict] = []
    seen: set[str] = set()
    for ref in refs:
        ref = str(ref).strip()
        hit = None
        if ref.isdigit():
            idx = int(ref) - 1
            if 0 <= idx < len(results):
                hit = results[idx]
        if hit is None:
            for item in results:
                if item.get("arxiv_id") == ref:
                    hit = item
                    break
        if hit and hit.get("arxiv_id") not in seen:
            seen.add(hit.get("arxiv_id"))
            picked.append(hit)
    return picked


# ── 命令行 ──────────────────────────────────────────────────────────
def _print_papers(agent: PaperAgent) -> None:
    papers = agent.list_papers()
    if not papers:
        print("论文库为空。先运行： python src/agent.py search \"关键词\"")
        return
    print(f"{'ID':5s} {'年':5s} {'块':>4s} {'页':>4s} {'语':3s} 标题")
    print("-" * 78)
    for p in papers:
        print(
            f"{p['paper_id']:5s} {str(p.get('year') or '-'):5s} "
            f"{str(p.get('n_chunks') or '-'):>4s} {str(p.get('n_pages') or '-'):>4s} "
            f"{p.get('lang') or '-':3s} {(p.get('title') or '')[:44]}"
        )
    print("-" * 78)
    print(f"共 {len(papers)} 篇；" + str(agent.library.stats()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent", description="论文智能体：关键词检索 -> 选择性入库 -> 多论文综合问答"
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="数据根目录")
    parser.add_argument("--embedder", default="bert", choices=["bert", "hashing"])
    parser.add_argument("--dry-run", action="store_true", help="不加载生成模型，只做检索与 prompt 预览")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search", help="按关键词检索 arXiv")
    p_search.add_argument("keywords")
    p_search.add_argument("--limit", type=int, default=10)

    p_load = sub.add_parser("load", help="把候选论文下载入库")
    p_load.add_argument("refs", nargs="+", help="候选序号或 arXiv ID")
    p_load.add_argument("--from", dest="candidate_file", default=None)

    sub.add_parser("list", help="查看论文库")
    sub.add_parser("candidates", help="查看最近一次检索结果")
    sub.add_parser("index", help="重建向量索引")

    p_add = sub.add_parser("add", help="手工添加本地 PDF")
    p_add.add_argument("pdf")
    p_add.add_argument("--id", dest="paper_id", default=None)

    p_rm = sub.add_parser("remove", help="从论文库删除")
    p_rm.add_argument("paper_id")

    p_ask = sub.add_parser("ask", help="提问")
    p_ask.add_argument("question")
    p_ask.add_argument("--k", type=int, default=DEFAULT_K)

    args = parser.parse_args(argv)
    agent = PaperAgent(
        data_dir=args.data_dir, embedder=args.embedder, dry_run=args.dry_run
    )

    try:
        if args.cmd == "search":
            out = agent.search(args.keywords, limit=args.limit)
            print(format_candidate_list(out["results"]))
            print(f"\n候选已缓存：{out['path']}")
            print(f"入库示例： python src/agent.py load 1 3 5")

        elif args.cmd == "candidates":
            payload = agent.last_candidates()
            if not payload:
                print("还没有检索缓存")
            else:
                print(f"查询：{payload.get('query')}  时间：{payload.get('created_at')}")
                print(format_candidate_list(payload.get("results", [])))

        elif args.cmd == "load":
            loaded = agent.load(args.refs, args.candidate_file)
            for item in loaded:
                st = item["stats"]
                print(
                    f"  ✓ {item['record']['paper_id']}  "
                    f"{st['n_pages']} 页 / {st['n_sections']} 章 / {st['n_chunks']} 块  "
                    f"{item['record']['title'][:40]}"
                )
            if not loaded:
                print("没有新论文入库")

        elif args.cmd == "add":
            item = agent.add_local(args.pdf, args.paper_id)
            st = item["stats"]
            print(f"  ✓ {item['record']['paper_id']}  {st['n_pages']} 页 / {st['n_chunks']} 块")

        elif args.cmd == "remove":
            print("已删除" if agent.remove_paper(args.paper_id) else "未找到该论文")

        elif args.cmd == "list":
            _print_papers(agent)

        elif args.cmd == "index":
            meta = agent.build_index()
            print(json.dumps(meta, ensure_ascii=False, indent=2))

        elif args.cmd == "ask":
            result = agent.ask(args.question, k=args.k)
            print(f"\n【问题】{result['question']}")
            print(f"\n检索：命中 {result['n_hits']} 条，覆盖 {result['coverage']['n_papers']} 篇论文"
                  f"（{result['retrieve_seconds']}s）\n")
            print(result["evidence_table"])
            if result.get("answer"):
                print(f"\n【回答】\n{result['answer']}")
                timing = {k: result.get(k) for k in ("prompt_tokens", "out_tokens", "ttft_s", "tok_per_s")}
                print(f"\n生成：{timing}")
            else:
                print("\n[dry-run] 未加载生成模型。发给模型的 prompt 长度："
                      f"{result.get('prompt_chars')} 字符")
                print("-" * 60)
                print(result.get("prompt_preview", ""))
            print(format_citations(result["citations"]))
        return 0
    except AgentError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
