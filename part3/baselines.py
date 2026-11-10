"""Part 3 baselines, wrapped to return the same flat result dict as run_adaptive().

Baseline 1 — fixed Airflow-style workflow
    The exact predefined steps of the Part 1 `train_models_from_db` task, in order:
    read -> drop hard-coded id / timestamp / known-leaky columns -> one-hot -> scaled
    logistic regression (balanced) -> threshold 0.5 -> holdout. No validation, no
    adaptation. Like the DAG (default_args retries=2) a failed step is re-run up to
    2 more times with the SAME code, then the run fails.

Baseline 2 — the simple agentic system from class
    The unmodified in-class LangGraph loop (part3/class_loop_original.py):
    data_scientist -> hypothesis -> ml_engineer -> judge -> revise ... -> holdout.
"""

from __future__ import annotations

import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from solution.langgraph.graph import _fit_predict_proba

# Same hard-coded lists as lab/airflow/dags/fraud_etl_pipeline.py
ID_COLUMNS = ["transaction_id", "customer_id", "merchant_id", "PassengerId", "Name", "Ticket"]
LEAKY_COLUMNS = ["fraud_probability"]
AIRFLOW_RETRIES = 2


def _empty(system: str) -> dict:
    return {"system": system, "status": "failed", "error": None, "holdout": {}, "validation_issues": [],
            "repairs": [], "audit": {}, "dropped_columns": [], "tuner_used": False, "retries": 0,
            "experiments": 0, "failed_experiments": 0, "stop_reason": None, "agent_calls": 0,
            "route": [], "events": [], "runtime_s": 0.0}


# ---------------------------------------------------------------------------
# Baseline 1: fixed workflow
# ---------------------------------------------------------------------------
def run_fixed_workflow(dataset_path: str, target_column: str, *, seed: int = 42, holdout_frac: float = 0.2,
                       trainer_fail_times: int = 0, **_ignored) -> dict:
    res = _empty("fixed_airflow")
    res["route"] = ["read", "drop_known_columns", "one_hot", "train_logreg", "holdout_eval"]
    t0 = time.time()
    fails_left = trainer_fail_times
    for attempt in range(1 + AIRFLOW_RETRIES):
        res["agent_calls"] += len(res["route"])
        try:
            df = pd.read_csv(dataset_path, low_memory=False)
            drop = [c for c in [*ID_COLUMNS, *LEAKY_COLUMNS, "timestamp", target_column] if c in df.columns]
            X = pd.get_dummies(df.drop(columns=drop), drop_first=True, dtype=float).fillna(0)
            y = df[target_column]
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=holdout_frac, stratify=y, random_state=seed)
            if fails_left > 0:
                fails_left -= 1
                raise RuntimeError("transient worker failure (injected)")
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            res["holdout"] = {
                "f1": float(f1_score(y_te, pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_te, pred)),
                "roc_auc": float(roc_auc_score(y_te, proba)),
                "pr_auc": float(average_precision_score(y_te, proba)),
                "threshold": 0.5, "model_family": "logistic_regression", "n_holdout": int(len(y_te)),
            }
            res.update(status="completed", experiments=1, dropped_columns=drop[:-1], retries=attempt)
            break
        except Exception as exc:  # noqa: BLE001
            res["error"] = f"{type(exc).__name__}: {exc}"
            res["retries"] = attempt
    res["runtime_s"] = round(time.time() - t0, 2)
    return res


# ---------------------------------------------------------------------------
# Baseline 2: in-class agent loop
# ---------------------------------------------------------------------------
def run_class_loop(dataset_path: str, target_column: str, *, budget: int = 6, target_score: float = 0.8,
                   seed: int = 42, holdout_frac: float = 0.2, trainer_fail_times: int = 0, **_ignored) -> dict:
    from part3 import class_loop_original as cls

    res = _empty("class_agent_loop")
    t0 = time.time()
    original_fit = cls._fit_score
    counter = {"left": trainer_fail_times}

    def flaky_fit(*args, **kwargs):  # same fault injection as the other two systems
        if counter["left"] > 0:
            counter["left"] -= 1
            raise RuntimeError("transient worker failure (injected)")
        return original_fit(*args, **kwargs)

    cls._fit_score = flaky_fit
    try:
        out = cls.run_solution_loop(dataset_path=dataset_path, target_column=target_column,
                                    target_score=target_score, max_cycles=budget, seed=seed,
                                    holdout_frac=holdout_frac)
        steps = out.get("steps") or []
        ml_steps = [s for s in steps if s.get("node") == "ml_engineer"]
        spec = out.get("best_spec") or {}
        res["route"] = [s.get("node") for s in steps]
        res["agent_calls"] = len(steps)
        res["experiments"] = len(ml_steps)
        res["failed_experiments"] = sum(1 for s in ml_steps if s.get("status") == "failed")
        res["retries"] = sum(1 for s in ml_steps if s.get("repaired"))
        res["dropped_columns"] = list(spec.get("drop_columns") or [])
        res["stop_reason"] = out.get("stop_reason") or (out.get("verdict") or {}).get("decision")
        hold = out.get("holdout_metrics") or {}
        if spec and hold:
            # Re-score the chosen spec with probabilities so PR-AUC / ROC-AUC are comparable
            df = pd.read_csv(dataset_path, low_memory=False)
            y, p = _fit_predict_proba(df, target_column, spec, seed, holdout_frac, on_holdout=True)
            res["holdout"] = {"f1": float(hold.get("f1", 0)), "accuracy": float(hold.get("accuracy", 0)),
                              "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
                              "threshold": 0.5, "model_family": spec.get("model_family"),
                              "n_holdout": int(len(y))}
            res["status"] = "completed"
        else:
            res["error"] = "no holdout result"
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cls._fit_score = original_fit
    res["runtime_s"] = round(time.time() - t0, 2)
    return res
