"""集中配置：路径、模型名、切分与检索超参。

设计意图
--------
所有可调参数集中在顶部，做消融实验（E1 改变 topk、E2 换 dtype、E3 换嵌入模型）
时只需改这里，不动业务代码。
"""

from __future__ import annotations

from pathlib import Path

# ── 路径 ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

ABSTRACTS_PATH = DATA_DIR / "abstracts.jsonl"
QUERIES_PATH = DATA_DIR / "queries.txt"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
INDEX_VECTORS_PATH = DATA_DIR / "index_vectors.npz"   # 只持久化向量，不持久化索引

# ── 嵌入模型 ────────────────────────────────────────────────────────
# 语料为中英混合（5:5），必须使用多语模型；单语模型（bge-small-zh / MiniLM-L6）
# 在跨语言检索上会明显掉点，见 docs/experiment.md 的 E3 对照实验。
EMBED_MODEL = "intfloat/multilingual-e5-small"
EMBED_BATCH_SIZE = 16
EMBED_MAX_LENGTH = 512

# E5 系列要求：索引用 passage 前缀，查询用 query 前缀
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# ── 文本切分 ────────────────────────────────────────────────────────
# 按 token 预算切分而非固定字符数：中文约 1 字 ≈ 1 token，英文约 4 字符 ≈ 1 token，
# 固定字符数会让中英语料得到完全不同的切分粒度。
CHUNK_TARGET_TOKENS = 384
CHUNK_OVERLAP_TOKENS = 48

# ── 检索 ────────────────────────────────────────────────────────────
TOPK = 3

# ── 生成 ────────────────────────────────────────────────────────────
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GEN_MAX_NEW_TOKENS = 256
GEN_DTYPE = "bfloat16"          # ROCm 上比 float16 更稳，溢出风险低
GEN_DO_SAMPLE = False           # 贪心解码，保证实验可复现
