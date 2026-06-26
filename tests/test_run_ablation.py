import csv
import os
import subprocess
from pathlib import Path

import yaml

from scripts import run_ablation


def test_csv_columns_match_week8_schema():
    assert run_ablation.CSV_COLUMNS == [
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


def test_run_names_include_variant_and_seed():
    assert run_ablation.run_slug("a1", "no shadow sampling", 42) == "a1-seed42"
    assert run_ablation.run_slug("ours", "full system", 0) == "ours-seed0"


def test_write_variant_config_overrides_shadow_rate_and_seed(tmp_path):
    base = {
        "dataset": {"name": "sift1m", "path": "data/sift1m"},
        "shadow": {"sample_rate": 0.02},
        "experiment": {"seed": 42, "n_queries": 1000},
    }
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))

    out_path = run_ablation.write_variant_config(
        base_config=base_path,
        out_dir=tmp_path,
        slug="a1-seed7",
        seed=7,
        shadow_rate=0.0,
    )

    cfg = yaml.safe_load(out_path.read_text())
    assert cfg["shadow"]["sample_rate"] == 0.0
    assert cfg["experiment"]["seed"] == 7


def test_run_paths_use_short_variant_seed_prefix(tmp_path):
    paths = run_ablation.build_run_paths(
        out_dir=tmp_path / "reports",
        logs_dir=tmp_path / "logs",
        models_dir=tmp_path / "models",
        slug="a1-seed7",
        distill=True,
    )

    assert paths.train_log_dir == tmp_path / "logs" / "a1-seed7" / "train_linucb"
    assert paths.eval_log_dir == tmp_path / "logs" / "a1-seed7" / "eval_qnet"
    assert paths.model_path == tmp_path / "models" / "a1-seed7_qnet_distilled.pt"
    assert paths.train_metrics_path == tmp_path / "reports" / "train_metrics" / "a1-seed7.json"


def test_run_paths_use_large_model_suffix_without_distill(tmp_path):
    paths = run_ablation.build_run_paths(
        out_dir=tmp_path / "reports",
        logs_dir=tmp_path / "logs",
        models_dir=tmp_path / "models",
        slug="a5-seed7",
        distill=False,
    )

    assert paths.model_path == tmp_path / "models" / "a5-seed7_qnet_large.pt"


def test_append_result_writes_header_once(tmp_path):
    result_path = tmp_path / "ablation_results.csv"
    row = {col: "" for col in run_ablation.CSV_COLUMNS}
    row["variant_id"] = "ours"
    row["seed"] = 0

    run_ablation.append_result(result_path, row)
    run_ablation.append_result(result_path, row)

    lines = result_path.read_text().splitlines()
    assert lines[0].split(",") == run_ablation.CSV_COLUMNS
    assert len(lines) == 3

    with result_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["variant_id"] == "ours"


def test_append_result_migrates_stale_revenue_columns(tmp_path):
    result_path = tmp_path / "ablation_results.csv"
    result_path.write_text(
        "variant_id,seed,train_total_revenue,eval_total_revenue\n"
        "ours,0,10.5,12.75\n"
    )
    row = {col: "" for col in run_ablation.CSV_COLUMNS}
    row["variant_id"] = "ours"
    row["seed"] = 1
    row["linucb_total_revenue"] = 20.5
    row["qnet_total_revenue"] = 22.75

    run_ablation.append_result(result_path, row)

    with result_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0].keys() == set(run_ablation.CSV_COLUMNS)
    assert list(rows[0].keys())[-2:] == ["linucb_total_revenue", "qnet_total_revenue"]
    assert rows[0]["linucb_total_revenue"] == "10.5"
    assert rows[0]["qnet_total_revenue"] == "12.75"
    assert rows[1]["linucb_total_revenue"] == "20.5"
    assert rows[1]["qnet_total_revenue"] == "22.75"


def test_run_one_seed_passes_seed_to_qnet_training(tmp_path, monkeypatch):
    commands = []

    def fake_run_command(cmd):
        commands.append(cmd)
        return "Done in 1.0s (123 qps)\n"

    def fake_summary(log_dir):
        return {
            "n_shadow_sampled": 1,
            "mean_recall_when_sampled": 0.5,
            "accept_rate": 0.8,
            "total_revenue": 10.0,
            "mean_propensity": 0.04,
            "min_propensity": 0.01,
            "mean_latency_ms": 1.0,
            "p99_latency_ms": 2.0,
        }

    base = {
        "dataset": {"name": "sift1m", "path": "data/sift1m"},
        "shadow": {"sample_rate": 0.02},
        "experiment": {"seed": 42, "n_queries": 1000},
    }
    config_path = tmp_path / "sift1m.yaml"
    config_path.write_text(yaml.safe_dump(base))

    args = run_ablation.parse_args([
        "--variant-id",
        "ours",
        "--variant-name",
        "full_system",
        "--reward-mode",
        "ips",
        "--no-distill",
        "--config",
        str(config_path),
        "--seeds",
        "7",
        "--out-dir",
        str(tmp_path / "reports"),
        "--logs-dir",
        str(tmp_path / "logs"),
        "--models-dir",
        str(tmp_path / "models"),
    ])

    monkeypatch.setattr(run_ablation, "run_command", fake_run_command)
    monkeypatch.setattr(run_ablation, "learner_summary", fake_summary)
    monkeypatch.setattr(run_ablation, "load_train_metrics", lambda _: {"loss_final": 0.1})
    monkeypatch.setattr(run_ablation, "sla_violation_rate", lambda *_args, **_kwargs: 0.0)

    run_ablation.run_one_seed(args, seed=7)

    train_qnet_cmd = commands[1]
    assert "scripts/train_qnet.py" in train_qnet_cmd
    seed_idx = train_qnet_cmd.index("--seed")
    assert train_qnet_cmd[seed_idx + 1] == "7"
    reward_idx = train_qnet_cmd.index("--reward-mode")
    assert train_qnet_cmd[reward_idx + 1] == "ips"
    epochs_idx = train_qnet_cmd.index("--epochs")
    assert train_qnet_cmd[epochs_idx + 1] == "20"
    diagnostic_temp_idx = train_qnet_cmd.index("--diagnostic-temperature")
    assert train_qnet_cmd[diagnostic_temp_idx + 1] == "0.03"
    assert "--no-distill" in train_qnet_cmd

    eval_qnet_cmd = commands[2]
    qnet_temp_idx = eval_qnet_cmd.index("--qnet-temperature")
    assert eval_qnet_cmd[qnet_temp_idx + 1] == "0.03"


def test_report_generation_is_opt_in():
    args = run_ablation.parse_args([
        "--variant-id",
        "ours",
        "--variant-name",
        "full_system",
    ])

    assert args.write_report is False

    args = run_ablation.parse_args([
        "--variant-id",
        "ours",
        "--variant-name",
        "full_system",
        "--write-report",
    ])

    assert args.write_report is True


def test_plot_generation_is_opt_in():
    args = run_ablation.parse_args([
        "--variant-id",
        "ours",
        "--variant-name",
        "full_system",
    ])

    assert args.write_plot is False

    args = run_ablation.parse_args([
        "--variant-id",
        "ours",
        "--variant-name",
        "full_system",
        "--write-plot",
    ])

    assert args.write_plot is True


def test_plot_only_does_not_require_variant_args():
    args = run_ablation.parse_args(["--plot-only"])

    assert args.plot_only is True
    assert args.variant_id is None
    assert args.variant_name is None


def test_results_csv_path_defaults_to_out_dir(tmp_path):
    args = run_ablation.parse_args([
        "--variant-id",
        "ours",
        "--variant-name",
        "full_system",
        "--out-dir",
        str(tmp_path / "reports"),
    ])

    assert run_ablation.results_csv_path(args, Path(args.out_dir)) == tmp_path / "reports" / "ablation_results.csv"


def test_results_csv_path_can_be_overridden(tmp_path):
    custom_csv = tmp_path / "reports" / "week8" / "real_ablation_result.csv"
    args = run_ablation.parse_args([
        "--variant-id",
        "ours",
        "--variant-name",
        "full_system",
        "--results-csv",
        str(custom_csv),
    ])

    assert run_ablation.results_csv_path(args, Path(args.out_dir)) == custom_csv


def test_report_uses_current_revenue_columns(tmp_path):
    results_csv = tmp_path / "ablation_results.csv"
    row = {col: "" for col in run_ablation.CSV_COLUMNS}
    row.update({
        "variant_id": "ours",
        "variant_name": "full_system",
        "seed": 0,
        "qnet_total_revenue": 12.5,
        "eval_accept_rate": 0.8,
        "eval_p99_latency_ms": 1.2,
        "eval_mean_recall_when_sampled": 0.5,
    })
    run_ablation.append_result(results_csv, row)

    run_ablation.write_report_stub(tmp_path, results_csv, tmp_path / "figs" / "ablation.png")

    report = (tmp_path / "week8_ablation.md").read_text()
    assert "qnet_total_revenue" in report
    assert "eval_total_revenue" not in report


def test_plot_uses_current_revenue_columns(tmp_path):
    results_csv = tmp_path / "ablation_results.csv"
    row = {col: "" for col in run_ablation.CSV_COLUMNS}
    row.update({
        "variant_id": "ours",
        "variant_name": "full_system",
        "seed": 0,
        "qnet_total_revenue": 12.5,
        "eval_accept_rate": 0.8,
        "eval_p99_latency_ms": 1.2,
        "eval_mean_recall_when_sampled": 0.5,
    })
    run_ablation.append_result(results_csv, row)
    fig_path = tmp_path / "figs" / "ablation.png"

    run_ablation.regenerate_plot(results_csv, fig_path)

    assert fig_path.exists()


def test_normal_run_requires_variant_args():
    try:
        run_ablation.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should require variant args unless --plot-only is used")


def test_run_command_streams_stderr_and_collects_stdout(monkeypatch, capsys):
    calls = {}

    class FakeProc:
        def __init__(self):
            self.stdout = ["Done in 1.0s (123 qps)\n"]
            self.returncode = 0

        def wait(self):
            return self.returncode

    def fake_popen(cmd, **kwargs):
        calls["cmd"] = cmd
        calls.update(kwargs)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    output = run_ablation.run_command(["python", "child.py"])

    assert output == "Done in 1.0s (123 qps)\n"
    assert calls["stdout"] == subprocess.PIPE
    assert calls["stderr"] is None
    assert calls["text"] is True
    assert calls["bufsize"] == 1
    assert calls["env"]["PYTHONUNBUFFERED"] == "1"
    assert calls["env"] is not os.environ
    printed = capsys.readouterr().out
    assert "$ python child.py" in printed
    assert "Done in 1.0s (123 qps)" in printed
