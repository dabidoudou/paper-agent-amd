# 语料说明（10 篇 AI 方向论文 · 中英各 5 篇）

## 为什么这里没有 PDF

论文 PDF 受版权保护，且单个文件体积较大（1–10 MB），因此**不纳入版本控制**（见根目录 `.gitignore`）。
本仓库改为提交**抽取后的摘要文本** `data/abstracts.jsonl`（约 30 KB），它足以完整复现 RAG 问答流程。

- 想跑 **RAG 主流程** → 直接 `git clone`，无需下载任何 PDF
- 想跑 **摘要抽取模块**（`pdf_extract.py`）→ 按下表下载 PDF 并重命名放回本目录

## 复现步骤

```bash
# 1. 从下表"来源"列下载 PDF
# 2. 重命名为 file 列所示名称，放入 data/papers/
ls -1 data/papers/*.pdf | wc -l        # 应为 10
# 3. 运行抽取
python scripts/02_build_abstracts.py --papers data/papers/
```

> ⚠️ 文件名前三位即 `paper_id`，请勿改名，否则摘要与 PDF 将无法对齐。

## 论文清单

| ID | 文件名 | 标题 | 语言 | 年份 | 来源 | 许可 |
|---|---|---|---|---|---|---|
| P01 | `P01_rag-lewis-2020.pdf` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | EN | 2020 | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) | arXiv 非独占许可 |
| P02 | `P02_dpr-karpukhin-2020.pdf` | Dense Passage Retrieval for Open-Domain Question Answering | EN | 2020 | [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) | arXiv 非独占许可 |
| P03 | `P03_lost-in-the-middle-2023.pdf` | Lost in the Middle: How Language Models Use Long Contexts | EN | 2023 | [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) | arXiv 非独占许可 |
| P04 | `P04_rag-survey-2023.pdf` | Retrieval-Augmented Generation for Large Language Models: A Survey | EN | 2023 | [arXiv:2312.10997](https://arxiv.org/abs/2312.10997) | arXiv 非独占许可 |
| P05 | `P05_m3-embedding-2024.pdf` | M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings | EN | 2024 | [arXiv:2402.03216](https://arxiv.org/abs/2402.03216) | arXiv 非独占许可 |
| P06 | `P06_ccl2025-tree-rag.pdf` | 面向工艺规范的树结构检索增强生成方法研究 | ZH | 2025 | [ACL:2025.ccl-1.12](https://aclanthology.org/2025.ccl-1.12/) | CC BY 4.0 |
| P07 | `P07_ccl2025-two-stage-commonsense-rag.pdf` | 基于检索增强生成的两阶段常识推理方法 | ZH | 2025 | [ACL:2025.ccl-1.13](https://aclanthology.org/2025.ccl-1.13/) | CC BY 4.0 |
| P08 | `P08_ccl2025-frame-semantic-cot.pdf` | 基于检索增强思维提示的汉语框架语义解析方法 | ZH | 2025 | [ACL:2025.ccl-1.15](https://aclanthology.org/2025.ccl-1.15/) | CC BY 4.0 |
| P09 | `P09_ccl2024-chinese-char-word-rag.pdf` | 基于汉语字词资源的检索增强生成与应用评估 | ZH | 2024 | [ACL:2024.ccl-1.3](https://aclanthology.org/2024.ccl-1.3/) | CC BY 4.0 |
| P10 | `P10_ccl2025-topic-steering-mt.pdf` | 多领域翻译中语义消歧的话题方向盘方法 | ZH | 2025 | [ACL:2025.ccl-1.53](https://aclanthology.org/2025.ccl-1.53/) | CC BY 4.0 |

## 选型理由

1. **主题内聚**：10 篇中有 9 篇围绕检索增强生成（RAG）展开，便于验证细粒度语义检索。
2. **自指效应**：`P10` 的最优基线使用 **Qwen2.5-1.5B**，与本地生成模型同源，可作为演示亮点。
3. **许可明确**：全部来自 arXiv / ACL Anthology，可合法公开引用摘要。
4. **中英均衡**：5:5，用于验证**多语嵌入模型**的必要性（单语模型会显著掉点，见 `docs/experiment.md` 的 E3 实验）。
5. **含天然干扰项**：`P03`（长上下文位置偏差）与 `P06`（分块策略）语义高度相近，可用于检验 Top-1 / Top-3 排序质量。

## 引用与版权

本仓库仅收录各论文的**摘要原文**，用于课程技术演示，摘要版权归原作者所有。
如需在论文中引用，请使用上表"来源"列的正式出处。
