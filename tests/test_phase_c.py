"""阶段 C/D 验证：arXiv 检索解析、多论文综合 prompt、智能体工作流（dry-run）。

运行：
    python tests/test_phase_c.py

全部不需要 torch / pymupdf：arXiv 解析用离线 XML 片段，检索用 hashing 编码器，
生成环节走 `--dry-run`（只构造 prompt、不加载模型）。
真实联网检索作为"信息性"检查，网络不通不影响结论。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import PaperAgent, _pick_candidates  # noqa: E402
from corpus import build_chunks, load_abstracts  # noqa: E402
from discovery import (  # noqa: E402
    build_search_query,
    format_candidate_list,
    load_candidates,
    parse_atom,
    save_candidates,
)
from generator import (  # noqa: E402
    SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    build_citations,
    build_context,
    build_messages,
    format_citations,
)
from library import Library  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


ATOM_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2402.03216v5</id>
    <title>M3-Embedding:  Multi-Linguality,
      Multi-Functionality</title>
    <summary>In this paper,  we introduce a new
      embedding model for retrieval.</summary>
    <published>2024-02-05T17:26:49Z</published>
    <author><name>Jianlv Chen</name></author>
    <author><name>Shitao Xiao</name></author>
    <category term="cs.CL"/>
    <category term="cs.AI"/>
    <arxiv:comment>Accepted at ACL 2024</arxiv:comment>
    <link href="http://arxiv.org/abs/2402.03216v5" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2402.03216v5" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2307.03172v3</id>
    <title>Lost in the Middle</title>
    <summary>Models degrade when relevant text sits in the middle.</summary>
    <published>2023-07-06T17:54:11Z</published>
    <author><name>Nelson F. Liu</name></author>
  </entry>
</feed>
"""

HITS = [
    {"rank": 1, "paper_id": "P06", "title": "树结构 RAG", "section": "2 Method", "page": 3,
     "score": 0.81, "chunk_id": "P06#s2c0",
     "text": "固定文本长度分块会丢失段落层级之间的结构关系，导致输出质量下降。"},
    {"rank": 2, "paper_id": "P06", "title": "树结构 RAG", "section": "3 Experiments", "page": 5,
     "score": 0.74, "chunk_id": "P06#s3c0",
     "text": "树结构 RAG 相比朴素 RAG，ACC 平均提升 3.81%。"},
    {"rank": 3, "paper_id": "P03", "title": "Lost in the Middle", "section": "Abstract", "page": 1,
     "score": 0.69, "chunk_id": "P03#s1c0",
     "text": "Performance degrades when relevant information is in the middle of long contexts."},
]


# ── 1. arXiv 检索 ───────────────────────────────────────────────────
def test_discovery() -> None:
    section("1. arXiv 关键词检索")
    check("单词查询", build_search_query("rag") == "all:rag", build_search_query("rag"))
    multi = build_search_query("retrieval augmented generation")
    check("多词用 +OR+ 连接",
          multi == "all:retrieval+OR+all:augmented+OR+all:generation", multi)
    check("支持指定字段", build_search_query("rag", field="ti").startswith("ti:"))
    quoted = build_search_query('"graph neural network"')
    check("引号按短语处理", quoted.startswith('all:"') and "%20" in quoted, quoted)
    try:
        build_search_query("   ")
        check("空关键词应报错", False)
    except ValueError:
        check("空关键词应报错", True)

    results = parse_atom(ATOM_FIXTURE)
    check("解析出 2 条候选", len(results) == 2, len(results))
    first = results[0]
    check("arXiv ID 去掉版本号", first["arxiv_id"] == "2402.03216", first["arxiv_id"])
    check("标题空白被折叠", first["title"] == "M3-Embedding: Multi-Linguality, Multi-Functionality",
          first["title"])
    check("摘要空白被折叠", "  " not in first["abstract"])
    check("年份从 published 解析", first["year"] == 2024, first["year"])
    check("作者列表正确", first["authors"] == ["Jianlv Chen", "Shitao Xiao"], first["authors"])
    check("分类列表正确", first["categories"] == ["cs.CL", "cs.AI"], first["categories"])
    check("comment 被读取", "ACL 2024" in first["comment"], first["comment"])
    check("pdf 链接来自 link 标签", first["pdf_url"].endswith(".pdf") or "pdf/" in first["pdf_url"],
          first["pdf_url"])
    check("缺失 pdf 链接时按 ID 兜底", results[1]["pdf_url"] == "https://arxiv.org/pdf/2307.03172",
          results[1]["pdf_url"])

    out_dir = ROOT / "results" / "_tmp_cand"
    shutil.rmtree(out_dir, ignore_errors=True)
    try:
        path = save_candidates(results, out_dir, "retrieval augmented generation")
        check("候选缓存已写入", path.exists() and path.suffix == ".json")
        payload = load_candidates(path)
        check("缓存含查询与结果", payload["query"] and len(payload["results"]) == 2)
        text = format_candidate_list(results)
        check("候选列表含 arXiv ID", "2402.03216" in text)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# ── 2. 多论文综合 Prompt ────────────────────────────────────────────
def test_prompt() -> None:
    section("2. 多论文综合 Prompt")
    ctx = build_context(HITS)
    check("按论文分组（出现论文标题行）", "■ 论文 P06" in ctx and "■ 论文 P03" in ctx)
    check("统计论文数与证据数", "2 篇论文" in ctx and "3 条证据" in ctx)
    check("证据含章节与页码", "§2 Method p.3" in ctx, ctx[:200])
    check("证据编号连续", "[1]" in ctx and "[3]" in ctx)

    msgs = build_messages("固定分块有什么问题？", HITS)
    check("跨论文时切换到综合 system prompt",
          msgs[0]["content"] == SYNTHESIS_SYSTEM_PROMPT)
    check("综合 prompt 要求说明论文关系", "一致" in msgs[0]["content"] or "互补" in msgs[0]["content"])
    check("单一论文时使用基础 prompt",
          build_messages("q", HITS[:2])[0]["content"] == SYSTEM_PROMPT)
    user = msgs[1]["content"]
    check("user 含作答要求", "【要求】" in user and "根据提供的资料无法回答" in user)
    check("user 含引用指令", "[编号]" in user)

    cites = build_citations(HITS, excerpt_chars=20)
    check("引用条数与命中数一致", len(cites) == len(HITS))
    check("引用含论文与章节", cites[0]["paper_id"] == "P06" and cites[0]["section"] == "2 Method")
    check("摘录被截断并加省略号", cites[0]["excerpt"].endswith("…"), cites[0]["excerpt"])
    table = format_citations(cites)
    check("证据表含页码范围", "p.3" in table)
    check("空命中时给出占位", format_citations([]) == "（无引用）")


# ── 3. 候选挑选 ─────────────────────────────────────────────────────
def test_pick() -> None:
    section("3. 候选挑选")
    results = parse_atom(ATOM_FIXTURE)
    check("按序号挑选", [r["arxiv_id"] for r in _pick_candidates(results, ["1"])] == ["2402.03216"])
    check("按 arXiv ID 挑选",
          [r["arxiv_id"] for r in _pick_candidates(results, ["2307.03172"])] == ["2307.03172"])
    check("越界序号被忽略", _pick_candidates(results, ["99"]) == [])
    check("重复去重", len(_pick_candidates(results, ["1", "2402.03216"])) == 1)


# ── 4. 智能体工作流（dry-run）───────────────────────────────────────
def test_agent_dry_run() -> None:
    section("4. 智能体工作流（dry-run）")
    data_dir = ROOT / "results" / "_tmp_agent"
    shutil.rmtree(data_dir, ignore_errors=True)
    try:
        # 造一个含 3 篇论文的库（用摘要语料切块，避免依赖 PDF）
        lib = Library(data_dir / "library")
        papers = load_abstracts(ROOT / "data" / "abstracts.jsonl")
        for paper in papers[:3]:
            chunks = build_chunks([paper])
            for c in chunks:
                c["section"] = "Abstract"
                c["page"] = 1
            lib.add(paper["paper_id"], {
                "title": paper["title"], "lang": paper["lang"], "year": paper["year"],
                "n_chunks": len(chunks), "n_pages": 1, "arxiv_id": "test-" + paper["paper_id"],
            })
            lib.save_chunks(paper["paper_id"], chunks)
        lib.save()

        agent = PaperAgent(data_dir=data_dir, embedder="hashing", dry_run=True)
        check("论文库读到 3 篇", len(agent.list_papers()) == 3, len(agent.list_papers()))

        meta = agent.build_index()
        print(f"  索引：{meta['n_chunks']} 块 / dim={meta['dim']} / backend={meta['backend']}"
              f" / {meta['index_bytes']} 字节 / {meta['embed_seconds']}s")
        check("索引记录论文数", meta["n_papers"] == 3, meta["n_papers"])
        check("索引记录了嵌入耗时", meta["embed_seconds"] >= 0)
        check("索引文件已落盘",
              (data_dir / "index" / "vectors.npz").exists()
              and (data_dir / "index" / "chunks.jsonl").exists()
              and (data_dir / "index" / "meta.json").exists())

        result = agent.ask("固定文本长度分块会带来什么问题？", k=4)
        print(f"  问答：命中 {result['n_hits']} 条，覆盖 {result['coverage']['n_papers']} 篇"
              f"（{result['retrieve_seconds']}s）")
        check("dry-run 不加载生成模型", result["answer"] is None)
        check("返回了命中", result["n_hits"] >= 1, result["n_hits"])
        check("覆盖至少 2 篇（多篇连结）", result["coverage"]["n_papers"] >= 2, result["coverage"])
        check("生成了 prompt 预览", bool(result.get("prompt_preview")))
        check("prompt 长度已统计", result["prompt_chars"] > 100, result.get("prompt_chars"))
        check("证据表含论文编号",
              any(p["paper_id"] in result["evidence_table"] for p in agent.list_papers()))
        check("引用结构完整",
              all({"rank", "paper_id", "section", "page"} <= set(c) for c in result["citations"]))

        # 索引过期自动重建：删掉一篇后 chunk 数变化
        agent.remove_paper("P03")
        check("删除后库中只剩 2 篇", len(agent.list_papers()) == 2)
        agent._chunks = None
        agent._vectors = None
        agent._ensure_index()
        check("索引随库变化重建", len(agent._chunks) < meta["n_chunks"],
              (len(agent._chunks), meta["n_chunks"]))
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


# ── 5. 联网检索（信息性，不判失败）──────────────────────────────────
def test_live_optional() -> None:
    section("5. 联网检索（信息性）")
    try:
        from discovery import search_arxiv

        results = search_arxiv("retrieval augmented generation", limit=3)
        print(f"  联网成功，返回 {len(results)} 条")
        for item in results[:3]:
            print(f"   - {item['arxiv_id']}  {item['title'][:56]}")
        check("联网检索返回结果", len(results) > 0, len(results))
    except Exception as exc:
        print(f"  跳过联网检查（{type(exc).__name__}: {exc}）")


# ── main ────────────────────────────────────────────────────────────
def main() -> int:
    print("阶段 C/D 验证：arXiv 检索 + 多论文综合 prompt + 智能体 dry-run")
    print(f"python={sys.version.split()[0]}  numpy={np.__version__}")
    test_discovery()
    test_prompt()
    test_pick()
    test_agent_dry_run()
    test_live_optional()

    print("\n" + "=" * 60)
    if FAILS:
        print(f"FAILED: {len(FAILS)} 项")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
