# 基于本地小参数模型的论文摘要 RAG 问答 Demo

> 在 **AMD AUP Learning Cloud**（Radeon 8060S / ROCm）上，用纯 Python 手写一条极简 RAG 流水线：
> **论文 PDF → 自动抽取摘要 → 稠密向量检索 → 本地 Qwen2.5-1.5B 生成带引用的回答**。
> 不依赖 LangChain，每一步都可计时，用于获取端到端的算力消耗数据。

## 当前状态

| 模块 | 状态 |
|---|---|
| 数据协议（`.gitignore` / `.gitattributes` / `manifest.csv`） | ✅ 已完成 |
| 语料：10 篇 AI 方向论文摘要（中英各 5 篇） | ✅ 已完成 `data/abstracts.jsonl` |
| 评测问题集（6 条 + 1 条附加） | ✅ 已完成 `data/queries.txt` |
| 抽取模块 `pdf_extract.py` / `abstract_parser.py` | ⏳ 待实现 |
| 检索与生成模块 `embedder.py` / `vector_store.py` / `generator.py` | ⏳ 待实现 |
| 云端预演与算力数据采集 | ⏳ 待执行 |

## 语料

10 篇论文全部来自 **arXiv** 与 **ACL Anthology**（开放获取），PDF 原件因版权与体积原因**不入库**，
仓库仅提交抽取后的摘要文本。

- 论文清单与下载链接 → [`data/papers/README.md`](data/papers/README.md)
- 摘要数据（可直接使用）→ [`data/abstracts.jsonl`](data/abstracts.jsonl)
- 评测问题 → [`data/queries.txt`](data/queries.txt)

> 其中 `P10` 恰好使用 **Qwen-2.5-1.5B**（与本 demo 的生成模型同源），可用于自指演示。

## 目录结构

```
.
├── data/
│   ├── papers/README.md      # 论文清单 + 官方链接 + 版权说明（PDF 自行下载）
│   ├── manifest.csv          # 机器可读的语料元数据
│   ├── abstracts.jsonl       # 10 篇摘要（RAG 主流程的输入）
│   └── queries.txt           # 评测问题集
├── .gitignore                # PDF / 缓存 / 运行时产物 不入库
└── .gitattributes            # 统一 LF 换行 + 二进制标记
```

## 快速开始

```bash
git clone <this-repo>
cd <repo>
# RAG 主流程不需要 PDF，开箱即跑（代码待补充）
```

## 待办

- [ ] 实现摘要抽取（`pymupdf` + 规则引擎，LLM 兜底可选）
- [ ] 实现向量化（手写 `transformers` mean-pooling，不依赖 sentence-transformers）
- [ ] 实现 FAISS 检索（含 numpy 兜底）
- [ ] 实现 Qwen2.5-1.5B 生成与引用标注
- [ ] 补充计时埋点与算力规划文档
- [ ] AMD AUP Learning Cloud 云端预演 + 截图

## 许可

本仓库代码采用 MIT 许可；所收录的论文摘要版权归原作者所有，仅用于课程技术演示。
