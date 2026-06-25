# Week 1 系统理解笔记


## 系统整体架构

这个系统在模拟一个**向量检索市场**。我们假设自己是一家卖 ANN 搜索服务的公司 ，客户每秒发几万条查询过来。每条查询我们要做两个决定：**搜多仔细**（nprobe 从 8 到 128 共 5 档）和**收多少钱**（$0.001 到 $0.020 共 5 档）。搜得越仔细成本越高但召回越好，客户对召回满意就更可能下次还来，我们可以通过长期日志的学习来逐渐适应——这是个典型的长期收益优化问题。

系统由 5 个主要的 agent 组成，Orchestrator 是我们的调度中心。客户每来一条查询，在系统里都按固定 8 步走：

```
Query 进来
  │
  ├─ 1. DifficultyEstimator   → U_t        （难度估计：这个查询有多难？仅看 query 本身，不能偷看结果）
  ├─ 2. ContextCache          → h_t        （上下文缓存：最近 100 条的平均接受率/延迟/收入）
  ├─ 3. PolicyAgent           → z_t, p_t, propensity  （决策中心：选什么参数、报什么价、这个决策的概率）
  ├─ 4. ExecutionAgent        → results, L_t, C_t     （执行中心：FAISS 搜索 + 计时 + 算成本）
  ├─ 5. ShadowSampler         → （异步精准计算：2% 概率抽中时跑精确搜索，不阻塞主流程）
  ├─ 6. Buyer                 → A_t, S_t              （买方模拟：客户接受/拒绝 + 满意度）
  ├─ 7. 结算 R_t              = (p_t - C_t) 如果接受，否则白亏 C_t
  └─ 8. LogWriter + ContextCache 更新
```

其中**步骤 5 不阻塞主路径。而是异步运行** ShadowSampler 用 `ThreadPoolExecutor.submit()` fire-and-forget，`Q_t`（真实 recall）先填 None，shadow 线程算完了通过 `LogWriter.record_recall()` 回调补上。这意味着即使精确搜索要跑 100ms（比 ANN 搜索慢 50-100 倍），主请求的延迟完全不受影响。这就是文档里说的"systems contribution"——把一个不可观测的量（真实 recall）变成了可采样的信号，而且不影响在线性能。



