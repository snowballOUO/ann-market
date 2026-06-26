import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.distill import SmallQNet, distill_q_net
from src.models.q_net import LargeQNet, ips_bellman_loss, stable_bellman_loss


PRICE_TIERS = [0.001, 0.002, 0.005, 0.01, 0.02]


def policy_diagnostics(model, states_t, temperature: float) -> dict:
    model.eval()
    with torch.no_grad():
        q_values = model(states_t)
        probs = F.softmax(q_values / temperature, dim=1).cpu().numpy()

    price_probs = {
        str(price): float(probs[:, idx:: len(PRICE_TIERS)].sum(axis=1).mean())
        for idx, price in enumerate(PRICE_TIERS)
    }
    return {
        "q_mean": float(q_values.mean().item()),
        "q_std": float(q_values.std().item()),
        "diagnostic_temperature": temperature,
        "price_probs": price_probs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=None, help="Specific log dir to train on")
    ap.add_argument("--output-model", "--output", dest="output_model", default="models/qnet_distilled_v1.pt")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--target-sync-epochs", type=int, default=1)
    ap.add_argument("--metrics-out", default=None)
    ap.add_argument("--use-u-t", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-ut", dest="use_u_t", action="store_false", help="Alias for --no-use-u-t")
    ap.add_argument("--reward-mode", default="dr", choices=["dr", "ips"])
    ap.add_argument("--distill", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--distill-epochs", type=int, default=5)
    ap.add_argument("--diagnostic-temperature", type=float, default=0.03)
    args = ap.parse_args()

    from src.causal.dr_estimator import build_action_index, build_qnet_state, load_logs

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("1. 正在寻找最新的历史日志...")
    if args.log_dir is None:
        log_dirs = sorted(glob.glob("logs/run_*"))
        if not log_dirs:
            raise FileNotFoundError("No log directories found under logs/")
        latest_log_dir = log_dirs[-1]
    else:
        latest_log_dir = args.log_dir

    df = load_logs(latest_log_dir)
    print(f"成功加载日志: {latest_log_dir}，共 {len(df)} 条轨迹。")

    # Merge valid shadow recalls that arrived after a shard had already flushed.
    orphan_files = sorted(glob.glob(os.path.join(latest_log_dir, "*orphan*.parquet")))
    if orphan_files:
        df_orphan = pd.concat([pd.read_parquet(f) for f in orphan_files], ignore_index=True)
        df_orphan = df_orphan.dropna(subset=["Q_t"])
        recall_map = dict(zip(df_orphan["query_id"], df_orphan["Q_t"]))
        matched = df["query_id"].isin(recall_map)
        df.loc[matched, "Q_t"] = df.loc[matched, "query_id"].map(recall_map)
        print(f"  Merged {len(df_orphan)} orphan shadow recalls, matched {matched.sum()} trajectories")

    print("2. 正在提取 Causal 状态特征 (Deconfounding)...")
    states = build_qnet_state(df, use_u_t=args.use_u_t).astype(np.float32)
    if not args.use_u_t:
        print("  (U_t zeroed out)")
    actions = build_action_index(df).astype(np.int64)
    rewards = df["R_t"].values.astype(np.float32)

    # Q_t quality feedback: shadow-sampled queries use true recall to adjust reward.
    Q_t = df["Q_t"].values.astype(float)
    has_quality = ~np.isnan(Q_t)
    mean_q = float(np.nanmean(Q_t)) if np.any(has_quality) else 0.5
    target_recall = max(mean_q, 0.3)
    quality_lambda = 0.05 / max(mean_q, 0.3)
    print(f"  Auto-calibrated: mean_recall={mean_q:.3f}, target={target_recall:.2f}, lambda={quality_lambda:.4f}")
    n_quality = int(has_quality.sum())
    if n_quality > 0:
        quality_bonus = quality_lambda * (Q_t[has_quality] - target_recall)
        rewards = rewards.copy()
        rewards[has_quality] += quality_bonus
        print(
            f"  Quality reward adjustment: {n_quality} queries "
            f"(avg Δ={quality_bonus.mean():+.6f}, range=[{quality_bonus.min():+.4f}, {quality_bonus.max():+.4f}])"
        )

    propensities = np.clip(df["propensity"].values, 1e-4, 1.0)

    n_actions = 25
    state_dim = states.shape[1]

    # next_state: i -> i+1; final row is terminal.
    next_states = np.zeros_like(states, dtype=np.float32)
    next_states[:-1] = states[1:]

    states_t = torch.tensor(states, dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    props_t = torch.tensor(propensities, dtype=torch.float32)
    next_states_t = torch.tensor(next_states, dtype=torch.float32)

    print(f"3. 正在训练 LargeQNet ({args.reward_mode.upper()} Bellman)...")
    large_q = LargeQNet(state_dim, n_actions)
    target_q = LargeQNet(state_dim, n_actions)
    target_q.load_state_dict(large_q.state_dict())
    target_q.eval()
    optimizer = optim.Adam(large_q.parameters(), lr=args.lr)

    dataset = TensorDataset(states_t, actions_t, rewards_t, next_states_t, props_t)
    torch_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=torch_generator)

    losses = []
    for epoch in range(args.epochs):
        large_q.train()
        total_loss = 0.0
        for s, a, r, s_next, p in loader:
            optimizer.zero_grad()
            if args.reward_mode == "dr":
                loss = stable_bellman_loss(large_q, target_q, s, a, r, s_next, p, gamma=args.gamma)
            else:
                loss = ips_bellman_loss(large_q, s, a, r, s_next, p, gamma=args.gamma)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(large_q.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % max(1, args.target_sync_epochs) == 0:
            target_q.load_state_dict(large_q.state_dict())
            target_q.eval()

        epoch_loss = total_loss / max(1, len(loader))
        losses.append(float(epoch_loss))
        print(f"  Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss:.6f}")

    if args.distill:
        print("4. 正在蒸馏为 SmallQNet (Hidden=32) 并导出 TorchScript...")
        export_model = SmallQNet(state_dim, n_actions)
        distill_q_net(
            large_q,
            export_model,
            DataLoader(states_t, batch_size=args.batch_size),
            state_dim=state_dim,
            epochs=args.distill_epochs,
        )
    else:
        print("4. 跳过蒸馏，直接导出 LargeQNet TorchScript...")
        export_model = large_q

    os.makedirs(os.path.dirname(args.output_model) or ".", exist_ok=True)
    dummy_input = torch.randn(1, state_dim)
    export_model.eval()
    traced_model = torch.jit.trace(export_model, dummy_input)
    traced_model.save(args.output_model)
    print(f"✅ 模型导出成功：{args.output_model}")
    diagnostics = policy_diagnostics(export_model, states_t, args.diagnostic_temperature)
    print(f"  Diagnostic price probs: {diagnostics['price_probs']}")

    if args.metrics_out is not None:
        os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
        metrics = {
            "log_dir": latest_log_dir,
            "output_model": args.output_model,
            "gamma": args.gamma,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "target_sync_epochs": args.target_sync_epochs,
            "use_u_t": args.use_u_t,
            "reward_mode": args.reward_mode,
            "distill": args.distill,
            "losses": losses,
            "loss_final": losses[-1] if losses else None,
            "loss_min": min(losses) if losses else None,
            **diagnostics,
        }
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
