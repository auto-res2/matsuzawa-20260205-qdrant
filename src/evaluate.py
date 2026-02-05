import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import yaml
from scipy import stats


PRIMARY_METRIC = "worst_group_accuracy"


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def plot_learning_curve(history: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    if "best_val_score" in history.columns:
        plt.plot(history["iteration"], history["best_val_score"], label="best_val_score")
    if "label_budget_used" in history.columns:
        plt.plot(history["iteration"], history["label_budget_used"], label="label_budget_used")
    plt.xlabel("Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(str(out_path))


def plot_group_bars(summary: Dict, out_path: Path) -> None:
    plt.figure(figsize=(4, 4))
    vals = [summary.get("group_acc_0", 0.0), summary.get("group_acc_1", 0.0)]
    plt.bar(["g0", "g1"], vals, color=["#4c72b0", "#dd8452"])
    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.title("Group accuracies")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(str(out_path))


def plot_confusion_like(summary: Dict, out_path: Path) -> None:
    mat = np.array(
        [
            [summary.get("group_acc_0", 0.0), 1.0 - summary.get("group_acc_0", 0.0)],
            [summary.get("group_acc_1", 0.0), 1.0 - summary.get("group_acc_1", 0.0)],
        ]
    )
    plt.figure(figsize=(4, 3))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Correct", "Error"], yticklabels=["g0", "g1"])
    plt.title("Group accuracy matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(str(out_path))


def plot_label_efficiency(history: pd.DataFrame, out_path: Path) -> None:
    if "label_budget_used" not in history.columns or "best_val_score" not in history.columns:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(history["label_budget_used"], history["best_val_score"], marker="o")
    plt.xlabel("Cumulative labels")
    plt.ylabel("Best val score")
    plt.title("Label-efficiency curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(str(out_path))


def aggregate_metrics(run_metrics: Dict[str, Dict]) -> Dict:
    metrics = {}
    for run_id, summary in run_metrics.items():
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                metrics.setdefault(k, {})[run_id] = v
    proposed = {k: v for k, v in metrics.get(PRIMARY_METRIC, {}).items() if "proposed" in k}
    baseline = {
        k: v
        for k, v in metrics.get(PRIMARY_METRIC, {}).items()
        if "comparative" in k or "baseline" in k
    }
    best_proposed = max(proposed.items(), key=lambda x: x[1]) if proposed else (None, float("nan"))
    best_baseline = max(baseline.items(), key=lambda x: x[1]) if baseline else (None, float("nan"))

    if best_baseline[1] == 0 or math.isnan(best_baseline[1]):
        gap = float("nan")
    else:
        gap = (best_proposed[1] - best_baseline[1]) / best_baseline[1] * 100.0
    return {
        "primary_metric": PRIMARY_METRIC,
        "metrics": metrics,
        "best_proposed": {"run_id": best_proposed[0], "value": best_proposed[1]},
        "best_baseline": {"run_id": best_baseline[0], "value": best_baseline[1]},
        "gap": gap,
    }


def comparison_plots(metrics: Dict[str, Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, values in metrics.items():
        runs = list(values.keys())
        vals = list(values.values())
        plt.figure(figsize=(8, 4))
        sns.barplot(x=runs, y=vals)
        plt.xticks(rotation=45, ha="right")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
        plt.title(f"Comparison: {metric}")
        plt.tight_layout()
        out_path = out_dir / f"comparison_{metric}_bar_chart.pdf"
        plt.savefig(out_path)
        plt.close()
        print(str(out_path))


def run_stats_test(run_metrics: Dict[str, Dict], out_dir: Path) -> None:
    primary = PRIMARY_METRIC
    proposed = []
    baseline = []
    for run_id, summary in run_metrics.items():
        if primary in summary:
            if "proposed" in run_id:
                proposed.append(summary[primary])
            if "comparative" in run_id or "baseline" in run_id:
                baseline.append(summary[primary])
    if len(proposed) >= 2 and len(baseline) >= 2:
        t_stat, p_val = stats.ttest_ind(proposed, baseline, equal_var=False)
        out = {"t_stat": float(t_stat), "p_value": float(p_val)}
        out_path = out_dir / "comparison_stat_tests.json"
        save_json(out_path, out)
        print(str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str)
    args = parser.parse_args()

    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    entity = cfg["wandb"]["entity"]
    project = cfg["wandb"]["project"]

    api = wandb.Api()
    run_ids = json.loads(args.run_ids)
    results_dir = Path(args.results_dir)

    run_metrics = {}

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history()
        summary = run.summary._json_dict
        config = dict(run.config)

        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = run_dir / "metrics.json"
        save_json(metrics_path, {"summary": summary, "config": config, "history": history.to_dict()})
        print(str(metrics_path))

        if not history.empty:
            plot_learning_curve(history, run_dir / f"{run_id}_learning_curve.pdf")
            plot_label_efficiency(history, run_dir / f"{run_id}_label_efficiency.pdf")
        plot_group_bars(summary, run_dir / f"{run_id}_group_accuracy_bar.pdf")
        plot_confusion_like(summary, run_dir / f"{run_id}_group_confusion_matrix.pdf")

        run_metrics[run_id] = summary

    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    aggregated = aggregate_metrics(run_metrics)
    aggregated_path = comparison_dir / "aggregated_metrics.json"
    save_json(aggregated_path, aggregated)
    print(str(aggregated_path))

    comparison_plots(aggregated["metrics"], comparison_dir)
    run_stats_test(run_metrics, comparison_dir)


if __name__ == "__main__":
    main()
