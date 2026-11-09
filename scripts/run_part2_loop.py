#!/usr/bin/env python3
"""Part 2: run the modified agent loop on the Part 1 output and save its evaluation.

Usage (from the repo root, after the Airflow DAG has produced fraud_clean_latest.csv):
    python scripts/run_part2_loop.py --max-cycles 4

Writes:
    reports/part2/loop_result.json   full result (audit, best spec, holdout metrics, steps)
    reports/part2/loop_cycles.md     one row per cycle + the final holdout row
    reports/part2/f1_by_cycle.png    validation F1 per cycle (before / after threshold tuning)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")  # sklearn convergence noise

from solution.langgraph.graph import run_solution_loop  # noqa: E402

OUT_DIR = REPO / "reports" / "part2"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the modified agent loop on Part 1 data")
    parser.add_argument("--dataset", default=str(REPO / "datasets" / "airflow" / "fraud_clean_latest.csv"))
    parser.add_argument("--target", default="is_fraud")
    parser.add_argument("--target-score", type=float, default=0.8)
    parser.add_argument("--max-cycles", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not Path(args.dataset).is_file():
        print(f"dataset not found: {args.dataset} -> run the fraud_etl_pipeline DAG first")
        return 1

    result = run_solution_loop(
        dataset_path=args.dataset,
        target_column=args.target,
        target_score=args.target_score,
        max_cycles=args.max_cycles,
        seed=args.seed,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "loop_result.json").write_text(json.dumps(result, indent=2, default=str))

    # One row per cycle: ML engineer score, then the tuner's score (if it ran)
    rows, cycle = [], 0
    for step in result["steps"]:
        if step["node"] == "ml_engineer":
            cycle += 1
            rows.append({
                "cycle": cycle,
                "model": step.get("model_family"),
                "hyperparameters": (step.get("spec") or {}).get("hyperparameters"),
                "f1_default": step.get("f1"),
                "f1_tuned": None,
                "threshold": 0.5,
            })
        elif step["node"] == "threshold_tuner" and rows:
            out = step.get("output") or {}
            rows[-1]["f1_tuned"] = step.get("f1")
            rows[-1]["threshold"] = out.get("chosen_threshold")
            rows[-1]["val_roc_auc"] = (out.get("validation_metrics") or {}).get("roc_auc")

    hold = result.get("holdout_metrics") or {}
    lines = [
        "| cycle | model | hyperparameters | val F1 @0.5 | val F1 tuned | threshold |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        tuned = f"{r['f1_tuned']:.3f}" if r["f1_tuned"] is not None else "-"
        f1d = f"{r['f1_default']:.3f}" if r["f1_default"] is not None else "-"
        lines.append(f"| {r['cycle']} | {r['model']} | `{r['hyperparameters']}` | {f1d} | {tuned} | {r['threshold']} |")
    lines.append("")
    lines.append(
        f"**Sealed holdout (best spec):** F1 = {hold.get('f1', 0):.3f}, "
        f"ROC-AUC = {hold.get('roc_auc') or 0:.3f}, PR-AUC = {hold.get('pr_auc') or 0:.3f} "
        f"(positive rate = {hold.get('positive_rate', 0):.3f}), accuracy = {hold.get('accuracy', 0):.3f}, "
        f"threshold = {hold.get('decision_threshold')}"
    )
    (OUT_DIR / "loop_cycles.md").write_text("\n".join(lines) + "\n")

    # Chart: validation F1 per cycle, before and after threshold tuning
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [r["cycle"] for r in rows]
    ax.plot(xs, [r["f1_default"] or 0 for r in rows], marker="o", label="ML engineer (threshold 0.5)")
    if any(r["f1_tuned"] is not None for r in rows):
        ax.plot(xs, [r["f1_tuned"] or 0 for r in rows], marker="s", label="after Threshold Tuner")
    ax.axhline(hold.get("f1", 0), color="gray", linestyle="--", label="holdout F1 (best spec)")
    ax.set_xticks(xs)
    ax.set_xlabel("cycle")
    ax.set_ylabel("F1 (is_fraud)")
    ax.set_title("Agent loop: validation F1 by cycle")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "f1_by_cycle.png", dpi=120)

    print("\n".join(lines))
    print("\naudit:", json.dumps(result.get("data_audit"), indent=2))
    print(f"\nsaved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
