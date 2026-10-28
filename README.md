# paper-agent-amd

> 运行在 **AMD AUP Learning Cloud**（Radeon 8060S / ROCm）上的论文智能体：
> **关键词检索论文 → 选择性入库 → 全文分节检索 → 带章节与页码溯源的多论文综合问答**。
> 本地 Qwen2.5-1.5B + FAISS，手写 RAG 全链路，不依赖 LangChain。

## 工作流

```
① search  关键词 ──► arXiv API ──► 候选列表（标题/年份/摘要/PDF 链接）
② load    挑选候选 ──► 下载 PDF ──► 抽全文 + 识别章节 + 裁掉参考文献 ──► 写入论文库
③ index   全库 chunk 向量化 ──► FAISS 索引（每条带 论文/章节/页码 溯源）
④ ask     问题 ──► 多论文多样化检索 ──► 按论文分组 ──► 生成回答 + 证据表
```

设计上刻意选择**确定性编排（workflow agent）**而非工具调用：1.5B 模型在 function calling 上
不可靠，把"该调哪个工具"交给代码、"该说什么话"交给模型，稳定性和可复现性都更高。

## 当前状态

| 模块 | 文件 | 状态 |
|---|---|---|
| 文本切分（中英混合按 token 预算） | `src/chunker.py` | ✅ |
| 语料解析 | `src/corpus.py` | ✅ |
| PDF 清洗 / 分节 / 裁参考文献 / 带溯源切块 | `src/fulltext.py` | ✅ |
| 论文库注册表（论文单独存储） | `src/library.py` | ✅ |
| 多论文多样化检索（配额 + MMR） | `src/retrieval.py` | ✅ |
| 手写 mean-pooling 编码器 + 免依赖哈希编码器 | `src/embedder.py` | ✅ |
| Prompt（多论文综合）+ Qwen2.5 流式生成 | `src/generator.py` | ✅ |
| FAISS + numpy 兜底 | `src/vector_store.py` | ✅ |
| 计时与指标落盘 | `src/timing.py` | ✅ |
| arXiv 关键词检索 | `src/discovery.py` | ✅ |
| PDF 下载与入库 | `src/fetcher.py` | ✅ |
| 工作流编排 + CLI | `src/agent.py` | ✅ |
| 本地测试（4 套，无 torch 依赖） | `tests/` | ✅ 全通过 |
| 云端预演与算力数据 | `docs/` | ⏳ 待执行 |

## 目录结构

```
.
├── data/
│   ├── abstracts.jsonl        # 10 篇论文摘要（零下载的降级路径）
│   ├── manifest.csv           # 语料元数据清单
│   ├── queries.txt            # 评测问题集
│   ├── papers/README.md       # 论文清单 + 官方链接 + 版权说明
│   ├── candidates/            # 关键词搜索结果缓存（纯元数据）
│   └── library/               # 论文库
│       ├── library.json       # ✅ 入库：注册表（不含正文）
│       └── P01/ U01/ ...      # 🚫 不入库：PDF、全文、章节、chunk（版权）
├── src/                       # 全部源码
├── tests/                     # 本地验证（不需要 GPU）
├── .gitignore                 # PDF / 全文 / 派生索引 一律不入库
└── .gitattributes             # 统一 LF + 二进制标记
```

> **版权说明**：仓库只提交摘要与元数据。全文是受版权保护的完整表达，
> 由 `agent.py load` 从 arXiv / ACL Anthology 现场重建，可通过 `library.json` 里的 sha1 校验。

## 快速开始

```bash
git clone https://github.com/dabidoudou/paper-agent-amd.git
cd paper-agent-amd

# AMD 云端：只装两个轻量依赖（不要装 sentence-transformers，会顶掉 ROCm 版 torch）
pip install --only-binary=:all: faiss-cpu pymupdf

# ① 按关键词找论文
python src/agent.py search "tree structured chunking retrieval augmented generation" --limit 10
# ② 挑选入库（序号或 arXiv ID 均可）
python src/agent.py load 1 3 5
# ③ 建索引
python src/agent.py index
# ④ 提问（带证据溯源）
python src/agent.py ask "固定文本长度分块会带来什么问题？"
```

其他命令：`list`（查看论文库）、`candidates`（查看最近检索结果）、
`add <pdf>`（手工添加本地论文）、`remove <paper_id>`、`--dry-run`（不加载模型，只验证管道）。

## 依赖

| 包 | 用途 | 说明 |
|---|---|---|
| `numpy` | 向量运算、numpy 兜底检索 | 通常已预装 |
| `faiss-cpu` | 向量索引（可选，缺失自动回退 numpy） | 纯 CPU，无 CUDA 依赖 |
| `pymupdf` | PDF 文本抽取（可选回退 `pypdf`） | 单轮子、CJK 友好 |
| `torch` / `transformers` | 嵌入与生成 | **由 AMD 平台镜像提供，不要自行升级** |

`sentence-transformers` **故意不使用**：它会声明 `torch>=...` 依赖，可能把 ROCm 定制构建的
torch 换成 CUDA/CPU 轮子，导致 `torch.cuda.is_available()` 变 False。本项目手写
mean-pooling 编码器（约 20 行）替代它，依赖更少且每步可计时。

## 本地验证（无需 GPU）

```bash
python tests/test_batch1.py    # 切分 / 语料 / 检索链路
python tests/test_batch2.py    # Prompt / 编码器 / 计时 / 惰性加载
python tests/test_phase_a.py   # 全文分节 / 论文库 / 多样化检索
python tests/test_phase_c.py   # arXiv 检索 / 多论文 prompt / 工作流 dry-run
```

## 许可

代码 MIT；所收录论文摘要与元数据版权归原作者所有，仅用于课程技术演示。
