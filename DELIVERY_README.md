# ANN Marketplace v2 — 交付说明

面向同事的快速上手指南。完整架构对比见 **`docs/ARCHITECTURE_COMPARISON.md`**。

---

## 1. 包内容

| 包含 | 说明 |
|------|------|
| `src/` | 全部源代码（含新增 `pricing/`、`buyer_versions/`、负载生成器） |
| `scripts/` | 实验与训练脚本 |
| `configs/` | 数据集与实验配置 |
| `tests/` | 单元测试（含 hetero / tiered / must-serve） |
| `models/` | ONNX 难度模型 + Q-Net 权重（小文件） |
| `reports/` | 已跑实验结果 CSV |
| `docs/` | 架构对比、买家模型问题说明 |

| **不包含**（需自行下载） | 原因 |
|--------------------------|------|
| `data/` | SIFT1M 等 HDF5 + FAISS 索引约 **7+ GB** |
| `logs/` | 运行日志，体积大且可重现 |

---

## 2. 环境安装

```bash
cd ann-marketplace-full
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 3. 数据准备（SIFT1M）

```bash
# 下载数据并建索引（需网络，约 600MB+）
bash scripts/download_sift1m.sh data/sift1m   # 若无此脚本，用 download_data.py
python scripts/build_index.py --config configs/sift1m.yaml
python scripts/pretrain_difficulty.py --config configs/sift1m.yaml
```

数据就绪后应有：
- `data/sift1m/sift-128-euclidean.hdf5`
- `data/sift1m/index_ivfpq.faiss`

---

## 4. 买家版本切换

```bash
python scripts/switch_buyer_version.py show
python scripts/switch_buyer_version.py must_serve   # 永不拒单（场景实验默认）
python scripts/switch_buyer_version.py hetero       # 异质市场
python scripts/switch_buyer_version.py soft         # 软约束（推荐通用）
```

或通过环境变量（无需改文件）：

```bash
export BUYER_VERSION=must_serve
```

---

## 5. 快速复现关键实验

### 5.1 异质市场（Fixed vs HeteroLinUCB，+11.5%）

```bash
BUYER_VERSION=hetero python scripts/run_hetero_experiment.py \
  --config configs/sift1m_hetero.yaml \
  --n-queries 10000 \
  --seeds 42,123,456 \
  --output reports/sift1m_hetero_real.csv
```

### 5.2 场景 must-serve（S1–S6，+48~50%，多进程）

```bash
BUYER_VERSION=must_serve python scripts/run_scenario_experiment.py \
  --config configs/sift1m_scenario.yaml \
  --scenarios all \
  --seeds 42,123,456 \
  --n-queries 5000 \
  --workers 32 \
  --faiss-threads 1 \
  --output reports/sift1m_scenario_must_serve.csv
```

### 5.3 三档分层负载

```bash
python scripts/run_tiered_experiment.py \
  --config configs/sift1m.yaml \
  --tiers all \
  --seeds 42 \
  --n-queries 5000
```

### 5.4 运行测试

```bash
pytest tests/ -q
# 或仅新版测试
pytest tests/test_must_serve_scenario.py tests/test_hetero_marketplace.py tests/test_tiered_workload.py -q
```

---

## 6. 已有实验结果（reports/）

| 文件 | 内容 |
|------|------|
| `main_results.csv` | 原始 baseline 条件（~$47，勿与异质实验混比） |
| `sift1m_hetero_real.csv` | 异质市场 3 seeds |
| `sift1m_scenario_must_serve.csv` | S1–S6 must-serve 42 jobs |
| `tiered_experiment.csv` | Tier1/2/3 分层 |
| `sift1m_problem1_fixed_fixed_vs_qnet.csv` | 问题1修复后 Fixed vs Q-Net |

---

## 7. v2 相对原始代码的核心变化（一句话）

**补全买家合同语义 + 结构化异质负载 + 可行域约束 + HeteroLinUCB**，使动态定价能在同负载下稳定击败强 Fixed 基线。

详细对比图见：`docs/ARCHITECTURE_COMPARISON.md`

---

## 8. 联系 / 待办

- [ ] 在 10 维状态下重训 Q-Net
- [ ] hetero / tiered 实验脚本加 `--workers` 并行
- [ ] 根目录 `SYSTEM_DOCUMENTATION.md`（仓库外）可与此文档合并

交付日期：2026-07-06
