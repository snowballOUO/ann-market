"""
Week 8 ablation runner.

Runs one ablation variant for one or more seeds:
  1. run LinUCB logging
  2. train a Q-Net from that log
  3. evaluate the trained Q-Net
  4. append a row to reports/ablation_results.csv
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.agents.learner_agent import LearnerAgent


CSV_COLUMNS = [
    "variant_id",
    "variant_name",
    "seed",
    "shadow_rate",
    "use_u_t",
    "reward_mode",
    "gamma",
    "distill",
    "epochs",
    "qnet_temperature",
    "train_n_shadow_sampled",
    "train_mean_recall_when_sampled",
    "train_accept_rate",
    "train_mean_propensity",
    "train_min_propensity",
    "qnet_loss_final",
    "qnet_q_mean",
    "qnet_q_std",
    "qnet_price_probs",
    "eval_accept_rate",
    "eval_mean_revenue_per_query",
    "eval_mean_recall_when_sampled",
    "eval_mean_latency_ms",
    "eval_p99_latency_ms",
    "eval_sla_violation_rate",
    "eval_qps",
    "eval_mean_propensity",
    "eval_min_propensity",
    "linucb_total_revenue",
    "qnet_total_revenue",
]


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def run_slug(variant_id: str, variant_name: str, seed: int) -> str:
    return f"{slugify(variant_id)}-seed{seed}"


@dataclass(frozen=True)
class RunPaths:
    train_log_dir: Path
    eval_log_dir: Path
    model_path: Path
    train_metrics_path: Path


def build_run_paths(out_dir: Path, logs_dir: Path, models_dir: Path, slug: str, distill: bool = True) -> RunPaths:
    model_kind = "distilled" if distill else "large"
    return RunPaths(
        train_log_dir=logs_dir / slug / "train_linucb",
        eval_log_dir=logs_dir / slug / "eval_qnet",
        model_path=models_dir / f"{slug}_qnet_{model_kind}.pt",
        train_metrics_path=out_dir / "train_metrics" / f"{slug}.json",
    )


def write_variant_config(
    base_config: Path,
    out_dir: Path,
    slug: str,
    seed: int,
    shadow_rate: float,
) -> Path:
    cfg = yaml.safe_load(base_config.read_text())
    cfg.setdefault("shadow", {})["sample_rate"] = shadow_rate
    cfg.setdefault("experiment", {})["seed"] = seed

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_path


def append_result(result_path: Path, row: dict[str, Any]) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists() and result_path.stat().st_size > 0:
        migrate_results_csv(result_path)
    exists = result_path.exists() and result_path.stat().st_size > 0
    with result_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


def migrate_results_csv(result_path: Path) -> None:
    with result_path.open(newline="") as f:
        reader = csv.DictReader(f)
        old_fieldnames = reader.fieldnames or []
        if old_fieldnames == CSV_COLUMNS:
            return
        rows = list(reader)

    migrated = []
    for row in rows:
        clean = {col: row.get(col, "") for col in CSV_COLUMNS}
        if not clean["linucb_total_revenue"]:
            clean["linucb_total_revenue"] = row.get("train_total_revenue", "")
        if not clean["qnet_total_revenue"]:
            clean["qnet_total_revenue"] = row.get("eval_total_revenue", "")
        migrated.append(clean)

    with result_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(migrated)


def run_command(cmd: list[str]) -> str:
    print("$ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=1,
        env=env,
    )

    output_parts: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        output_parts.append(line)

    returncode = proc.wait()
    output = "".join(output_parts)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd, output=output)
    return output


def learner_summary(log_dir: Path) -> dict[str, Any]:
    summary = LearnerAgent(str(log_dir)).summary()
    if summary is None:
        raise RuntimeError(f"No trajectories found in {log_dir}")
    return summary


def qps_from_run_output(output: str) -> float:
    match = re.search(r"\(([\d.]+)\s+qps\)", output)
    return float(match.group(1)) if match else float("nan")


def sla_violation_rate(log_dir: Path, sla_ms: float = 2.0) -> float:
    df = LearnerAgent(str(log_dir)).read_trajectories()
    if df.empty or "L_t" not in df:
        return float("nan")
    df = df.dropna(subset=["L_t"])
    if df.empty:
        return float("nan")
    return float((df["L_t"] * 1000.0 > sla_ms).mean())


def load_train_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def results_csv_path(args: argparse.Namespace, out_dir: Path) -> Path:
    return Path(args.results_csv) if args.results_csv else out_dir / "ablation_results.csv"


def regenerate_plot(results_csv: Path, fig_path: Path) -> None:
    if not results_csv.exists():
        return

    df = pd.read_csv(results_csv)
    if df.empty:
        return

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    plot_df["label"] = plot_df["variant_id"].astype(str) + "\n" + plot_df["variant_name"].astype(str)

    metrics = [
        ("qnet_total_revenue", "Q-Net Revenue"),
        ("eval_accept_rate", "Accept Rate"),
        ("eval_p99_latency_ms", "P99 Latency (ms)"),
        ("eval_mean_recall_when_sampled", "Mean Recall"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    axes = axes.ravel()

    grouped = plot_df.groupby("label", sort=False)
    labels = list(grouped.groups.keys())
    x = range(len(labels))

    for ax, (col, title) in zip(axes, metrics):
        means = grouped[col].mean()
        stds = grouped[col].std().fillna(0.0)
        ax.bar(x, [means[label] for label in labels], yerr=[stds[label] for label in labels], capsize=4)
        if col == "eval_p99_latency_ms":
            ax.axhline(2.0, color="red", linestyle="--", linewidth=1.2)
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def write_report_stub(out_dir: Path, results_csv: Path, fig_path: Path) -> None:
    report_path = out_dir / "week8_ablation.md"
    df = pd.read_csv(results_csv) if results_csv.exists() else pd.DataFrame()
    lines = [
        "# Week 8 Ablation Results",
        "",
        "## Setup",
        "",
        f"- Results CSV: `{results_csv}`",
        f"- Figure: `{fig_path}`",
        "",
        "## Results",
        "",
    ]
    if not df.empty:
        cols = [
            "variant_id",
            "variant_name",
            "seed",
            "qnet_total_revenue",
            "eval_accept_rate",
            "eval_p99_latency_ms",
            "eval_mean_recall_when_sampled",
        ]
        table = df[cols].fillna("")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in table.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    else:
        lines.append("No results yet.")
    lines.extend(["", "## Notes", "", "- Add interpretation for each variant after reviewing the metrics."])
    report_path.write_text("\n".join(lines) + "\n")


def run_one_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    slug = run_slug(args.variant_id, args.variant_name, seed)
    out_dir = Path(args.out_dir)
    logs_dir = Path(args.logs_dir)
    models_dir = Path(args.models_dir)
    paths = build_run_paths(out_dir, logs_dir, models_dir, slug, distill=args.distill)

    config_path = write_variant_config(
        base_config=Path(args.config),
        out_dir=out_dir / "configs",
        slug=slug,
        seed=seed,
        shadow_rate=args.shadow_rate,
    )
    py = sys.executable
    run_command([
        py,
        "scripts/run_experiment.py",
        "--config",
        str(config_path),
        "--policy",
        "linucb",
        "--n-queries",
        str(args.n_queries),
        "--seed",
        str(seed),
        "--log-dir",
        str(paths.train_log_dir),
        "--use-u-t" if args.use_u_t else "--no-use-u-t",
    ])

    run_command([
        py,
        "scripts/train_qnet.py",
        "--log-dir",
        str(paths.train_log_dir),
        "--output-model",
        str(paths.model_path),
        "--gamma",
        str(args.gamma),
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--reward-mode",
        args.reward_mode,
        "--metrics-out",
        str(paths.train_metrics_path),
        "--distill" if args.distill else "--no-distill",
        "--use-u-t" if args.use_u_t else "--no-use-u-t",
        "--diagnostic-temperature",
        str(args.qnet_temperature),
    ])

    eval_output = run_command([
        py,
        "scripts/run_experiment.py",
        "--config",
        str(config_path),
        "--policy",
        "qnet",
        "--n-queries",
        str(args.n_queries),
        "--seed",
        str(seed),
        "--log-dir",
        str(paths.eval_log_dir),
        "--qnet-model",
        str(paths.model_path),
        "--qnet-temperature",
        str(args.qnet_temperature),
        "--use-u-t" if args.use_u_t else "--no-use-u-t",
    ])

    train_summary = learner_summary(paths.train_log_dir)
    eval_summary = learner_summary(paths.eval_log_dir)
    train_metrics = load_train_metrics(paths.train_metrics_path)
    eval_total_revenue = float(eval_summary["total_revenue"])
    n_queries = int(args.n_queries)

    return {
        "variant_id": args.variant_id,
        "variant_name": args.variant_name,
        "seed": seed,
        "shadow_rate": args.shadow_rate,
        "use_u_t": args.use_u_t,
        "reward_mode": args.reward_mode,
        "gamma": args.gamma,
        "distill": args.distill,
        "epochs": args.epochs,
        "qnet_temperature": args.qnet_temperature,
        "train_n_shadow_sampled": int(train_summary["n_shadow_sampled"]),
        "train_mean_recall_when_sampled": train_summary["mean_recall_when_sampled"],
        "train_accept_rate": train_summary["accept_rate"],
        "train_mean_propensity": train_summary["mean_propensity"],
        "train_min_propensity": train_summary["min_propensity"],
        "qnet_loss_final": train_metrics.get("loss_final", ""),
        "qnet_q_mean": train_metrics.get("q_mean", ""),
        "qnet_q_std": train_metrics.get("q_std", ""),
        "qnet_price_probs": json.dumps(train_metrics.get("price_probs", {}), sort_keys=True),
        "eval_accept_rate": eval_summary["accept_rate"],
        "eval_mean_revenue_per_query": eval_total_revenue / max(n_queries, 1),
        "eval_mean_recall_when_sampled": eval_summary["mean_recall_when_sampled"],
        "eval_mean_latency_ms": eval_summary["mean_latency_ms"],
        "eval_p99_latency_ms": eval_summary["p99_latency_ms"],
        "eval_sla_violation_rate": sla_violation_rate(paths.eval_log_dir, sla_ms=args.sla_ms),
        "eval_qps": qps_from_run_output(eval_output),
        "eval_mean_propensity": eval_summary["mean_propensity"],
        "eval_min_propensity": eval_summary["min_propensity"],
        "linucb_total_revenue": train_summary["total_revenue"],
        "qnet_total_revenue": eval_total_revenue,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run one Week 8 ablation variant.")
    ap.add_argument("--variant-id", default=None, help="e.g. ours, a1, a2")
    ap.add_argument("--variant-name", default=None, help="e.g. no_shadow_sampling")
    ap.add_argument("--config", default="configs/sift1m.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-queries", type=int, default=10000)
    ap.add_argument("--shadow-rate", type=float, default=0.02)
    ap.add_argument("--use-u-t", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reward-mode", default="dr", choices=["dr", "ips"])
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--distill", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--qnet-temperature", type=float, default=0.03)
    ap.add_argument("--sla-ms", type=float, default=2.0)
    ap.add_argument("--out-dir", default="reports/week8")
    ap.add_argument("--results-csv", default=None)
    ap.add_argument("--logs-dir", default="logs/week8")
    ap.add_argument("--models-dir", default="models/week8")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--write-plot", action="store_true")
    ap.add_argument("--write-report", action="store_true")
    args = ap.parse_args(argv)
    if not args.plot_only and (not args.variant_id or not args.variant_name):
        ap.error("--variant-id and --variant-name are required unless --plot-only is used")
    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    results_csv = results_csv_path(args, out_dir)
    fig_path = out_dir / "figs" / "ablation.png"

    if args.plot_only:
        regenerate_plot(results_csv, fig_path)
        print(f"Wrote {fig_path}")
        return

    for seed in args.seeds:
        row = run_one_seed(args, seed)
        append_result(results_csv, row)

    if args.write_plot:
        regenerate_plot(results_csv, fig_path)
    if args.write_report:
        write_report_stub(out_dir, results_csv, fig_path)
    print(f"Wrote {results_csv}")
    if args.write_plot:
        print(f"Wrote {fig_path}")
    if args.write_report:
        print(f"Wrote {out_dir / 'week8_ablation.md'}")


if __name__ == "__main__":
    main()
