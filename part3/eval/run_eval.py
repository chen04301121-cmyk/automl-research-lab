#!/usr/bin/env python3
"""Run every evaluation case against all three systems, score them, write results + dashboard.

Usage (repo root, after the Part 1 DAG has produced datasets/airflow/fraud_clean_latest.csv):
    python -m part3.eval.run_eval                 # all systems, all cases
    python -m part3.eval.run_eval --cases C5_dirty_numeric --systems adaptive

Outputs:
    part3/eval/results.json   raw results + per-criterion scores
    part3/eval/results.md     summary tables (also pasted into the README)
    part3/dashboard.html      self-contained dashboard (open in a browser)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

from part3.adaptive_system import run_adaptive  # noqa: E402
from part3.baselines import run_class_loop, run_fixed_workflow  # noqa: E402
from part3.eval import make_datasets  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DATA = EVAL_DIR / "data"
SYSTEMS = {
    "adaptive": run_adaptive,
    "class_agent_loop": run_class_loop,
    "fixed_airflow": run_fixed_workflow,
}
LABELS = {"adaptive": "Adaptive multi-agent (Part 3)", "class_agent_loop": "Class agent loop",
          "fixed_airflow": "Fixed Airflow workflow"}


# ---------------------------------------------------------------------------
# Scoring: each check returns (passed: bool, observed: str)
# ---------------------------------------------------------------------------
def _f(x):
    return "-" if x is None else f"{x:.3f}"


def check(name: str, args: dict, r: dict, case: dict) -> tuple[bool, str]:
    hold = r.get("holdout") or {}
    done = r["status"] == "completed"
    if name == "completed":
        return done, r["status"] if done else f"failed: {str(r.get('error'))[:60]}"
    if name == "column_dropped":
        ok = args["column"] in (r.get("dropped_columns") or [])
        return ok, "dropped" if ok else "kept"
    if name == "no_leak_in_final":
        pr = hold.get("pr_auc")
        return (done and pr is not None and pr < args["max_pr_auc"]), f"PR-AUC {_f(pr)}"
    if name == "min_holdout_f1":
        f1 = hold.get("f1")
        return (done and f1 is not None and f1 >= args["value"]), f"F1 {_f(f1)}"
    if name == "tuner_used":
        return r.get("tuner_used", False) == args["expected"], f"tuner_used={r.get('tuner_used', False)}"
    if name == "issue_detected":
        ok = args["type"] in (r.get("validation_issues") or [])
        return ok, "detected" if ok else "not detected"
    if name == "repair_applied":
        acts = [a.get("action") for a in r.get("repairs") or []]
        ok = args["action"] in acts
        return ok, "applied" if ok else "not applied"
    if name == "retried":
        return r.get("retries", 0) >= args["min"], f"retries={r.get('retries', 0)}"
    if name == "early_stop":
        budget = case["input"].get("budget", 6)
        decided = r.get("stop_reason") not in (None, "", "budget spent")
        ok = done and decided and r.get("experiments", 0) < budget
        return ok, f"{r.get('experiments', 0)}/{budget} exps · stop={r.get('stop_reason')}"
    raise ValueError(f"unknown check {name}")


def score_case(case: dict, r: dict) -> dict:
    rows, total, got = [], 0.0, 0.0
    for c in case["criteria"]:
        ok, observed = check(c["check"], c.get("args", {}), r, case)
        w = float(c.get("weight", 1))
        total += w
        got += w if ok else 0
        rows.append({"check": c["check"], "args": c.get("args", {}), "group": c["group"],
                     "weight": w, "passed": ok, "observed": observed})
    score = got / total if total else 0.0
    return {"score": round(score, 3), "passed": score >= case["pass_threshold"], "criteria": rows}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", help="case ids (default: all)")
    ap.add_argument("--systems", nargs="*", default=list(SYSTEMS), choices=list(SYSTEMS))
    ap.add_argument("--rebuild-data", action="store_true")
    args = ap.parse_args()

    if args.rebuild_data or not (DATA / "signal_clean.csv").exists():
        make_datasets.main()
    spec = json.loads((EVAL_DIR / "cases.json").read_text())
    cases = [c for c in spec["cases"] if not args.cases or c["id"] in args.cases]

    results = []
    for case in cases:
        inp = case["input"]
        for sys_name in args.systems:
            work = EVAL_DIR / "data" / "_work" / f"{case['id']}_{sys_name}"
            shutil.rmtree(work, ignore_errors=True)
            kwargs = {"budget": inp.get("budget", 6), "trainer_fail_times": inp.get("trainer_fail_times", 0)}
            if sys_name == "adaptive":
                kwargs["workdir"] = str(work)
            t0 = time.time()
            try:
                r = SYSTEMS[sys_name](str(DATA / inp["dataset"]), inp["target"], **kwargs)
            except Exception as exc:  # noqa: BLE001 - a crash is a failed case, not a crashed eval
                r = {"system": sys_name, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                     "runtime_s": round(time.time() - t0, 2)}
            r.pop("events", None) if sys_name != "adaptive" else None
            scored = score_case(case, r)
            results.append({"case": case["id"], "title": case["title"], "system": sys_name,
                            "result": r, **scored})
            mark = "PASS" if scored["passed"] else "fail"
            print(f"{case['id']:22s} {sys_name:17s} {mark}  score={scored['score']:.2f}  "
                  f"F1={_f((r.get('holdout') or {}).get('f1'))}  calls={r.get('agent_calls', 0)}  "
                  f"{r.get('runtime_s', 0)}s")

    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": cases, "results": results}
    (EVAL_DIR / "results.json").write_text(json.dumps(out, indent=2, default=str))

    from part3.eval.report import write_markdown, write_dashboard

    write_markdown(out, EVAL_DIR / "results.md")
    write_dashboard(out, REPO / "part3" / "dashboard.html")
    print("\nwrote part3/eval/results.json, part3/eval/results.md, part3/dashboard.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
