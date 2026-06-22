import os
import h5py
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def main():
    out_dir = "data/ag_news"
    out_file = os.path.join(out_dir, "ag_news-384-euclidean.hdf5")
    os.makedirs(out_dir, exist_ok=True)

    print("1. 正在从 Hugging Face 下载 AG News 文本数据集...")
    dataset = load_dataset("mteb/ag_news")
    train_texts = dataset['train']['text']
    test_texts = dataset['test']['text']

    print(f"   获取到 {len(train_texts)} 条训练集，{len(test_texts)} 条测试集。")

    print("\n2. 正在加载 384 维向量编码模型 (all-MiniLM-L6-v2)...")
    # 这是一个速度极快、专门用于召回的轻量级模型，维度刚好是 384
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("\n3. 开始将文本编码为 384 维稠密向量 (这可能需要几分钟，取决于你的 CPU/GPU)...")
    train_embeddings = model.encode(train_texts, show_progress_bar=True, convert_to_numpy=True)
    test_embeddings = model.encode(test_texts, show_progress_bar=True, convert_to_numpy=True)

    print("\n4. 正在使用 FAISS 暴力计算精确的 Top-100 Ground Truth (计算精确召回率必须用)...")
    d = train_embeddings.shape[1] # 384 维
    index = faiss.IndexFlatL2(d)  # 对应文件名里的 euclidean (L2 距离)
    index.add(train_embeddings)
    
    k = 100 # 预存前 100 个最近邻
    distances, neighbors = index.search(test_embeddings, k)

    print("\n5. 正在将数据打包写入 HDF5 文件...")
    with h5py.File(out_file, 'w') as f:
        f.create_dataset('train', data=train_embeddings)
        f.create_dataset('test', data=test_embeddings)
        f.create_dataset('distances', data=distances)
        f.create_dataset('neighbors', data=neighbors)

    print(f"\n✅ 成功！文件已保存至: {out_file}")

if __name__ == "__main__":
    main()
