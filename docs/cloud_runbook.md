# AMD AUP Learning Cloud 运行手册

面向 60 分钟实例的操作手册。核心原则：**先用零依赖路径验证管道，再逐层引入模型**，
这样任何失败都能立刻定位是"管道问题"还是"模型/GPU 问题"。

---

## 0. 上云前的三件事

### ① 确认代码已推送

```powershell
# 本地（Windows）
git push
git log --oneline -3
```

### ② 找到持久化目录（5 分钟，但决定后面要不要重下 3.6GB 模型）

```bash
df -h
ls -la ~
echo "persist-test $(date)" > ~/work/persist_test.txt
cat ~/work/persist_test.txt
# 关掉实例 → 重新开一个 → 再执行：
cat ~/work/persist_test.txt   # 还在 = 该目录持久
```

### ③ 设定环境变量（每个新 shell 都要重新执行）

```bash
export WORK_ROOT=$HOME/work        # 换成上一步确认的持久目录
export HF_HOME=$WORK_ROOT/cache/hf
export PIP_CACHE_DIR=$WORK_ROOT/cache/pip
mkdir -p "$WORK_ROOT" "$HF_HOME" "$PIP_CACHE_DIR"

# 先存环境基线，出事能回滚
pip list --format=freeze > "$WORK_ROOT/env_before.txt"
python -c "import torch,transformers;print('torch',torch.__version__,'hip',torch.version.hip,'cuda',torch.cuda.is_available(),'tf',transformers.__version__)"
```

> 把上面那行 `torch ...` 的输出**复制到记事本**。万一 torch 被 pip 换掉，这就是恢复依据。

---

## 1. 六步验收阶梯

### 第 1 步：clone + 零依赖空跑（约 1 分钟）

```bash
git clone https://github.com/dabidoudou/paper-agent-amd.git
cd paper-agent-amd
python src/agent.py --dry-run list
```

预期：`论文库为空。先运行： python src/agent.py search "关键词"`

失败则说明 Python 或仓库有问题，**不要往下走**。

### 第 2 步：预检（约 2 分钟）

```bash
python scripts/01_preflight_amd.py
```

重点看四项：
- `真实矩阵乘` fp32 / bf16 / fp16 是否都 PASS（**只看 `is_available()` 不算数**）
- `相对误差` 是否在预算内 → 决定 `GEN_DTYPE`
- `带宽实测` 与 `解码速度上限` → 写进报告的算力数据
- `transformers 认识 Qwen2 架构` 是否 PASS

### 第 3 步：装依赖（约 2 分钟）

```bash
pip install --only-binary=:all: -r requirements.txt
pip list --format=freeze > "$WORK_ROOT/env_after.txt"
diff "$WORK_ROOT/env_before.txt" "$WORK_ROOT/env_after.txt" || true
```

**diff 里只应出现 faiss-cpu / pymupdf / pypdf。若出现 torch 或 transformers，立即停止**并回滚 torch。

### 第 4 步：检索 + 入库（约 5 分钟）

```bash
python src/agent.py search "tree structured chunking retrieval augmented generation" --limit 8
python src/agent.py --dry-run load 1 2 3
python src/agent.py list
```

预期每篇显示 `xx 页 / yy 章 / zz 块`。

- **页数 = 1** → PDF 抽取失败（可能是扫描版，或 pymupdf 没装上）
- **章节数 = 1** → 章节标题没识别出来；功能正常，只是溯源粒度变粗。把这批 PDF 留下来调 `is_heading`

### 第 5 步：免模型建索引（约 1 分钟，不碰 GPU）

```bash
python src/agent.py --embedder hashing index
```

预期 `n_chunks` 几十到几百、`dim=512`、`backend=faiss`（若第 3 步 faiss 装成功）。

这一步验证了「切块 → 向量 → 建索引 → 落盘 → 加载」整条链路，**完全不依赖 torch**。

### 第 6 步：真模型建索引 + 问答（约 10 分钟，含下载）

```bash
# 6a 嵌入模型（约 470MB）
python src/agent.py --embedder bert index

# 6b 生成模型（约 3.08GB）
python src/agent.py ask "固定文本长度分块会带来什么问题？有哪些改进方案？"
```

第 6a 步是 `embedder.py` 的 torch 路径首次真跑，第 6b 步是 `generator.py` 首次真跑。

---

## 2. 截图清单（报告与 README 都用得上）

| 序号 | 内容 | 怎么来 |
|---|---|---|
| 01 | 平台实例规格页 | 平台控制台截图 |
| 02 | ROCm 设备信息 | `rocm-smi` |
| 03 | 预检报告（矩阵乘 + 带宽） | `python scripts/01_preflight_amd.py` 输出 |
| 04 | 关键词检索候选列表 | `agent.py search` 输出 |
| 05 | 入库结果（页/章/块数） | `agent.py load` + `agent.py list` |
| 06 | 检索命中 + 证据表 | `agent.py ask` 前半段 |
| 07 | 最终回答（带 [1][2] 引用） | `agent.py ask` 后半段 |
| 08 | 消融结果 | 见 `docs/experiment.md` |

---

## 3. 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `torch.cuda.is_available()` 变 False | torch 被换成 CUDA/CPU 轮子 | 按下面「torch 恢复」处理 |
| 模型下载卡住 | 直连 HF 不稳 | `export HF_ENDPOINT=https://hf-mirror.com` |
| `CERTIFICATE_VERIFY_FAILED` | CA 路径不可用 | `pip install certifi`（`discovery.py` 会自动用它） |
| `TypeError: ... 'torch_dtype'` | transformers ≥ 4.56 改名为 `dtype` | `pip install -U transformers` 或改代码 |
| `ImportError: libgomp.so.1` | faiss 缺 OpenMP | `sudo apt-get install -y libgomp1` |
| 抽取结果只有 1 页 | 扫描版 PDF | 换论文，或手工 `agent.py add` 一个文本版 |
| 回答里出现编造内容 | 拒答指令未生效 | 降低 `--k`，或检查 `SYSTEM_PROMPT` 是否被改动 |
| 检索全是同一篇论文 | 覆盖策略异常 | 已在 `retrieval.py` 中保证；若出现请记录 query 与 top-k 复现 |

### torch 恢复（被 pip 换掉时）

```bash
# 用之前记下的版本号，例如 2.6.0+rocm6.2
pip install --force-reinstall --no-deps torch==<原版本> \
  --index-url https://download.pytorch.org/whl/rocm6.2
python -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

---

## 4. 时间预算（单次 60 分钟）

| 时段 | 动作 |
|---|---|
| 0–5 | clone + 零依赖空跑 + 持久化验证 |
| 5–8 | **后台启动模型下载**（`nohup python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-1.5B-Instruct'); s('intfloat/multilingual-e5-small')" &`） |
| 8–11 | 装依赖 + diff 校验 |
| 11–13 | 跑预检 + 截图 |
| 13–20 | search + load + list + 截图 |
| 20–23 | hashing 建索引（管道自检） |
| 23–32 | bert 建索引 + ask + 截图 |
| 32–45 | 消融实验（见 experiment.md） |
| 45–55 | 补截图 + 保存 results |
| 55–60 | 结果打包下载 |

> 若只给一次 60 分钟：**优先保证 1→6b 全通并截图**，消融实验可砍到只做 E1（topk）。

---

## 5. 结果回传

云端实例会销毁，所以每次会话结束前必须把结果带出来：

```bash
# 方式 A（推荐）：用 git 带出来（注意 PDF / 索引已被 .gitignore 挡住）
git add results/ data/candidates/ data/library/library.json
git commit -m "exp: 云端预演 run1（含 timing 与预检数据）"
git push
```

```powershell
# 本地拉回
git pull
```

方式 B：JupyterLab 里右键 `results/timing.csv` → Download。

**不要把 PDF、`data/index/`、`data/library/P*/` 提交上去**——它们已在 `.gitignore` 里；提交前用
`git status --porcelain | Select-String '\.pdf$'` 确认无输出。
