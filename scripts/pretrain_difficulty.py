"""
  Pretrain Difficulty MLP — Week 4.

  Generates ground-truth difficulty labels for SIFT1M queries,
  trains a 3-layer MLP to predict difficulty from query features,
  and exports to ONNX for low-latency inference.

  Usage:
      python scripts/pretrain_difficulty.py --config configs/base.yaml

  Outputs:
      models/difficulty_v1.pt   — PyTorch weights
      models/difficulty_v1.onnx — ONNX model for inference
"""
import argparse
from src.system.types import Query
import os
import time
import yaml
import numpy as np
import faiss
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def extract_features(query,sample_vectors,global_mean_norm):
    v=query.v_t
    #特征1：范数偏离全局均值的程度
    norm=float(np.linalg.norm(v))
    norm_ratio=min(norm/max(global_mean_norm,1e-6),3.0)/3.0
    # 范围 [0, 1]，偏离中心越远越接近 1

    # 特征 2：局部密度——到最近参考向量的距离
    dists=np.linalg.norm(sample_vectors[:1000]-v, axis=1)
    nearest=float(np.min(dists))
    density=min(nearest/max(global_mean_norm,1e-6),1.0)
    # 范围 [0, 1]，越稀疏越接近 1

    #特征3：filter复杂度
    n_filter_keys=len(query.filter_t) if query.filter_t else 0
    filter_diff=min(n_filter_keys*0.2,0.6)
    # 范围 [0, 0.6]，SIFT1M 无 filter 所以始终 0

    # 特征 4：请求的 k（归一化）
    k_norm=min(query.k_t/100.0,1.0)
    # 范围 [0.1, 1.0]

    # 特征 5：SLA 紧密程度（SLA 越紧，搜索越难——不能慢慢翻）
    sla_norm=1.0-min(query.sla_t/0.1,1.0)
    # 范围 [0, 1]，SLA 越紧越接近 1

    # 特征 6：偏置项
    bias=1.0

    return np.array(
        [norm_ratio,density,filter_diff,k_norm,sla_norm,bias],
        dtype=np.float32,
    )

def generate_labels(xq,gt,index,n_queries=10000):
    """
          对每条查询跑 FAISS nprobe=16，和 ground truth 比对得 recall。
          true_difficulty = 1.0 - recall@k
    """
    index.nprobe=16
    labels=np.zeros(n_queries,dtype=np.float32)

    for i in tqdm(range(n_queries),desc="Generating labels for query"):
        v=xq[i].reshape(1,-1).astype(np.float32)
        k_t=10 # SIFT1M 默认 k=10

        #ANN搜索
        k_search=min(k_t*4,200)
        D,I=index.search(v, k_search)
        aprox_ids=I[0][:k_t]

        # 和 ground truth 比对
        true_ids=gt[i][:k_t]
        approx_set=set(int(x) for x in aprox_ids)
        true_set=set(int(x) for x in true_ids)
        recall=len(approx_set & true_set)/len(true_set)

        labels[i]=1.0-recall

    return labels

class DifficultyMLP(nn.Module):
    """3-layer MLP: 6 → 64 → 64 → 1 (sigmoid output)."""
    def __init__(self,input_dim=6,hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,hidden),  # 6×64 + 64 = 448 参数
            nn.ReLU(),
            nn.Linear(hidden,hidden), # 64×64 + 64 = 4160 参数
            nn.ReLU(),
            nn.Linear(hidden,1), # 64×1 + 1 = 65 参数
            nn.Sigmoid(), # 输出压缩到 [0, 1]
        )# 总参数：4673

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch, 1) → (batch,)

def train(model,X_train,Y_train,X_val,Y_val,epochs=200,lr=0.001):
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    loss_fn=nn.MSELoss()
    best_val_loss=float("inf")

    x_train_t=torch.tensor(X_train)
    y_train_t=torch.tensor(Y_train)
    x_val_t=torch.tensor(X_val)
    y_val_t=torch.tensor(Y_val)

    for epoch in range(epochs):
        # ── 训练 ──
        model.train()
        optimizer.zero_grad()
        pred=model(x_train_t)
        loss=loss_fn(pred,y_train_t)
        loss.backward()
        optimizer.step()

        # —— 验证 ——
        model.eval()
        with torch.no_grad():
            val_pred=model(x_val_t)
            val_loss=loss_fn(val_pred,y_val_t).item()

        # —— 存最优模型 ——
        if val_loss<best_val_loss:
            best_val_loss=val_loss
            os.makedirs("models",exist_ok=True)
            torch.save(model.state_dict(),"models/difficulty_v1.pt")

        if epoch%20==0:
            print(f"  epoch {epoch:3d}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}")

    return best_val_loss

def export_onnx(model,onnx_path="models/difficulty_v1.onnx"):
    model.eval()
    dummy_input=torch.randn(1,6)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["features"],
        output_names=["difficulty"],
        dynamic_axes={
            "features": {0:"batch_size"},
            "difficulty": {0:"batch_size"},
        },
        opset_version=14,
    )
    print(f"ONNX model saved to {onnx_path}")

def validate_onnx(onnx_path="models/difficulty_v1.onnx",n_warmup=100,n_test=1000):
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)

    # 预热（第一次推理有 JIT 编译开销）
    for _ in range(n_warmup):
        x=np.random.randn(1,6).astype(np.float32)
        session.run(None,{"features":x})

    # 正式测时
    latencies=[]
    for _ in range(n_test):
        x=np.random.randn(1,6).astype(np.float32)
        t0=time.perf_counter()
        session.run(None,{"features":x})
        latencies.append(time.perf_counter()-t0)

    latencies = np.array(latencies) * 1000  # 转 ms
    print(f"\nONNX inference latency (n={n_test}):")
    print(f"  mean: {latencies.mean():.3f} ms")
    print(f"  p50:  {np.percentile(latencies, 50):.3f} ms")
    print(f"  p99:  {np.percentile(latencies, 99):.3f} ms")
    print(f"  {'PASS' if np.percentile(latencies, 99) < 0.5 else 'FAIL'} (threshold: p99 < 0.5ms)")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args=ap.parse_args()

    cfg=yaml.safe_load(open(args.config))
    data_dir=cfg["dataset"]["path"]
    seed=cfg["experiment"]["seed"]

    # ── 1. 加载数据 ──
    from src.data.datasets import load_sift1m, load_hdf5
    name = cfg["dataset"]["name"]
    print(f"Loading {name}...")
    if name == "ag_news":
        filepath = os.path.join(data_dir, cfg["dataset"]["file"])
        xb, xq, xt, gt = load_hdf5(filepath)
    else:
        xb, xq, xt, gt = load_sift1m(data_dir)

    index_path = os.path.join(data_dir, "index_ivfpq.faiss")
    index = faiss.read_index(index_path)

    # 全局范数均值（特征提取用）
    global_mean_norm = float(np.linalg.norm(xt[:5000], axis=1).mean())
    sample_vecs = xt[:5000]

    # ── 2. 生成标签 ──
    print("\nGenerating difficulty labels (nprobe=16)...")
    labels = generate_labels(xq, gt, index, n_queries=len(xq))

    # ── 3. 提取特征 ──
    print("Extracting features...")
    X = np.array([
        extract_features(
            Query(id=f"q_{i:06d}", v_t=xq[i], k_t=10, filter_t={},
                  sla_t=0.05, budget_t=0.01),
            sample_vecs, global_mean_norm
        )
        for i in range(len(xq))
    ], dtype=np.float32)
    y = labels

    # ── 4. 80/20 split ──
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # ── 5. 训练 ──
    print(f"\nTraining MLP...")
    model = DifficultyMLP(input_dim=6, hidden=64)
    val_loss = train(model, X_train, y_train, X_val, y_val, epochs=200)

    print(f"\nFinal val MSE: {val_loss:.4f}")
    print(f"  {'PASS' if val_loss < 0.05 else 'FAIL'} (threshold: val MSE < 0.05)")

    # ── 6. 导出 ONNX ──
    print("\nExporting ONNX...")
    model.load_state_dict(torch.load("models/difficulty_v1.pt"))
    export_onnx(model)

    # ── 7. 验证 ONNX 推理延迟 ──
    validate_onnx()


if __name__ == "__main__":
    main()