"""Part 3 — adaptive multi-agent ML system (LangGraph).

A *supervisor* agent looks at the shared state after every step and decides which
specialist agent runs next. Every specialist returns control to the supervisor, so
the path through the graph is chosen at run time from intermediate results:

    START -> supervisor -> {validator | data_repair | auditor | planner | trainer |
                            trainer_repair | threshold_tuner | evaluator | finalize}
                 ^______________________________|                         |
                                                                           v
                                                                          END

Specialists (each one has a single responsibility):
  validator        schema + data-quality checks (labels, numeric contamination, duplicates)
  data_repair      fixes what the validator found (quarantine / coerce / dedupe), then re-validate
  auditor          leakage scan (single-feature ROC-AUC), id columns, class imbalance (reused from Part 2)
  planner          picks the next experiment: explore model families, then refine the best one
  trainer          fits the experiment on the train split, scores on validation
  trainer_repair   inspects a training failure and retries (same spec / simpler spec / other family)
  threshold_tuner  only for imbalanced targets: F1-optimal decision threshold
  evaluator        keeps the best model, detects plateaus, decides continue vs stop
  finalize         scores the best model once on the sealed holdout

Adaptive behaviour (examples): dirty data triggers repair + re-validation; a crashed
fit triggers a retry with a repaired spec; balanced data skips the tuner; a plateau
switches family or stops early to save budget.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Annotated, Any, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

from solution.langgraph.graph import _fit_predict_proba, node_data_quality_auditor
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

MAX_REPAIR_ROUNDS = 2       # validator <-> data_repair loops
MAX_FIT_RETRIES = 2         # trainer_repair retries per experiment
PLATEAU_DELTA = 0.005       # "no real improvement" margin
PLATEAU_PATIENCE = 2        # stop after this many non-improving experiments in a row
MIN_EXPERIMENTS = 3         # explore at least this many before stopping on a plateau
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)

# Exploration order, then refinements of the best family
EXPLORE = [
    {"model_family": "logistic_regression", "hyperparameters": {"C": 1.0, "max_iter": 500}},
    {"model_family": "hist_gradient_boosting", "hyperparameters": {"learning_rate": 0.1, "max_iter": 200}},
    {"model_family": "random_forest", "hyperparameters": {"n_estimators": 300, "max_depth": 12}},
]
REFINE = {
    "logistic_regression": [{"C": 0.2, "max_iter": 500}, {"C": 5.0, "max_iter": 500}],
    "hist_gradient_boosting": [{"learning_rate": 0.05, "max_iter": 400}, {"learning_rate": 0.2, "max_iter": 150}],
    "random_forest": [{"n_estimators": 500, "max_depth": None}, {"n_estimators": 300, "max_depth": 6}],
}


def _append(left: list, right: list) -> list:
    return (left or []) + (right or [])


class AdaptiveState(TypedDict, total=False):
    # input
    dataset_path: str
    target_column: str
    budget: int
    target_score: float
    seed: int
    holdout_frac: float
    workdir: str
    # data
    work_path: str            # current (possibly repaired) CSV
    validation: dict
    validated: bool
    repair_rounds: int
    repairs: list
    audit: dict
    audited: bool
    imbalanced: bool
    drops: list
    # experiments
    spec: dict | None
    exp: dict | None          # current experiment result
    retries_current: int
    experiments: Annotated[list, _append]
    best: dict | None
    plateau: int
    stop_reason: str | None
    # bookkeeping
    route: Annotated[list, _append]
    events: Annotated[list, _append]
    retries_total: int
    final: dict | None
    status: str
    error: str | None


# ===========================================================================
# Supervisor (router)
# ===========================================================================
def supervisor(state: AdaptiveState) -> dict:
    """Pick the next agent from the current state; the choice is logged in `route`."""
    nxt = _decide(state)
    return {"route": [nxt]}


def _decide(s: AdaptiveState) -> str:
    if s.get("final") is not None or s.get("status") == "failed":
        return "end"
    if not s.get("validated"):
        return "validator"
    v = s.get("validation") or {}
    if v.get("fatal"):
        return "finalize"
    if v.get("fixable") and s.get("repair_rounds", 0) < MAX_REPAIR_ROUNDS:
        return "data_repair"
    if not s.get("audited"):
        return "auditor"
    if s.get("stop_reason"):
        return "finalize"
    exp = s.get("exp")
    if exp is None:
        return "planner" if s.get("spec") is None else "trainer"
    if exp["status"] == "failed":
        return "trainer_repair" if s.get("retries_current", 0) < MAX_FIT_RETRIES else "evaluator"
    if s.get("imbalanced") and not exp.get("tuned"):
        return "threshold_tuner"
    return "evaluator"


ROUTES = {
    "validator": "validator", "data_repair": "data_repair", "auditor": "auditor",
    "planner": "planner", "trainer": "trainer", "trainer_repair": "trainer_repair",
    "threshold_tuner": "threshold_tuner", "evaluator": "evaluator",
    "finalize": "finalize", "end": END,
}


def route_from_supervisor(state: AdaptiveState) -> str:
    return state["route"][-1]


# ===========================================================================
# Specialist agents
# ===========================================================================
def validator(state: AdaptiveState) -> dict:
    """Detect problems; classify them as fatal (stop) or fixable (send to data_repair)."""
    df = pd.read_csv(state["work_path"], low_memory=False)
    target = state["target_column"]
    issues, fixable, fatal = [], [], []

    if target not in df.columns:
        fatal.append(f"target column '{target}' missing")
    else:
        y = pd.to_numeric(df[target], errors="coerce")
        bad_labels = int((~y.isin([0, 1])).sum())
        if bad_labels:
            fixable.append({"type": "invalid_labels", "rows": bad_labels})
        if y[y.isin([0, 1])].nunique() < 2:
            fatal.append("target has a single class")

    dup = int(df.duplicated().sum())
    if dup:
        fixable.append({"type": "duplicate_rows", "rows": dup})

    # Numeric contamination: an object column that is almost entirely numbers
    for col in df.columns:
        # text-typed column (object in pandas 2, "str" in pandas 3)
        if col == target or pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        nonnull = df[col].dropna()
        if nonnull.empty:
            continue
        parsed = pd.to_numeric(nonnull, errors="coerce")
        share_numeric = float(parsed.notna().mean())
        if 0.9 <= share_numeric < 1.0:
            fixable.append({"type": "numeric_contamination", "column": col,
                            "rows": int(parsed.isna().sum())})

    if len(df) < 50:
        fatal.append(f"only {len(df)} rows")

    issues = [*fixable, *({"type": "fatal", "detail": f} for f in fatal)]
    validation = {"n_rows": int(len(df)), "issues": issues, "fixable": fixable, "fatal": fatal}
    note = f"{len(df)} rows · {len(fixable)} fixable · {len(fatal)} fatal"
    return {"validation": validation, "validated": True,
            "events": [{"agent": "validator", "note": note, "issues": issues}]}


def data_repair(state: AdaptiveState) -> dict:
    """Apply one repair per detected issue, write a repaired CSV, then re-validate."""
    df = pd.read_csv(state["work_path"], low_memory=False)
    target = state["target_column"]
    actions = []
    for issue in state["validation"]["fixable"]:
        kind = issue["type"]
        if kind == "invalid_labels":
            y = pd.to_numeric(df[target], errors="coerce")
            bad = ~y.isin([0, 1])
            quarantine = df[bad]
            quarantine.to_csv(Path(state["workdir"]) / "quarantine_labels.csv", index=False)
            df = df[~bad].copy()
            df[target] = y[~bad].astype(int)
            actions.append({"action": "quarantine_invalid_labels", "rows": int(bad.sum())})
        elif kind == "duplicate_rows":
            n = len(df)
            df = df.drop_duplicates()
            actions.append({"action": "drop_duplicates", "rows": n - len(df)})
        elif kind == "numeric_contamination":
            col = issue["column"]
            values = pd.to_numeric(df[col], errors="coerce")
            n_bad = int(values.isna().sum() - df[col].isna().sum())
            df[col] = values  # bad strings -> NaN, imputed later by the model pipeline
            actions.append({"action": "coerce_numeric", "column": col, "rows": n_bad})

    out = Path(state["workdir"]) / f"repaired_round{state.get('repair_rounds', 0) + 1}.csv"
    df.to_csv(out, index=False)
    return {
        "work_path": str(out),
        "validated": False,                    # force a re-validation (retry the gate)
        "repair_rounds": state.get("repair_rounds", 0) + 1,
        "repairs": (state.get("repairs") or []) + actions,
        "events": [{"agent": "data_repair", "note": f"{len(actions)} actions", "actions": actions}],
    }


def auditor(state: AdaptiveState) -> dict:
    """Reuse the Part 2 auditor: leakage AUC scan, id columns, imbalance."""
    patch = node_data_quality_auditor({
        "dataset_path": state["work_path"],
        "target_column": state["target_column"],
        "profile": {},
    })
    audit = patch["data_audit"]
    return {
        "audit": audit,
        "audited": True,
        "imbalanced": bool(audit["imbalanced"]),
        "drops": audit["final_drops"],
        "events": [{"agent": "auditor", "note": patch["steps"][0]["note"], "audit": audit}],
    }


def planner(state: AdaptiveState) -> dict:
    """Explore one spec per family first, then refine the best family."""
    tried = [(e["spec"]["model_family"], str(e["spec"]["hyperparameters"])) for e in state.get("experiments") or []]
    candidates = list(EXPLORE)
    best = state.get("best")
    if best is not None:
        fam = best["spec"]["model_family"]
        candidates += [{"model_family": fam, "hyperparameters": hp} for hp in REFINE[fam]]
    for cand in candidates:
        if (cand["model_family"], str(cand["hyperparameters"])) not in tried:
            spec = {**cand, "drop_columns": list(state.get("drops") or []),
                    "experiment_id": f"exp-{len(tried) + 1}"}
            phase = "explore" if cand in EXPLORE else "refine best family"
            return {"spec": spec, "exp": None, "retries_current": 0,
                    "events": [{"agent": "planner", "note": f"{phase}: {spec['model_family']} {spec['hyperparameters']}"}]}
    return {"stop_reason": "search space exhausted",
            "events": [{"agent": "planner", "note": "nothing left to try"}]}


# Optional fault injection used by the evaluation suite (simulates a flaky worker)
_FAULTS = {"trainer_fail_times": 0}


def trainer(state: AdaptiveState) -> dict:
    spec = state["spec"]
    t0 = time.time()
    try:
        if _FAULTS["trainer_fail_times"] > 0:
            _FAULTS["trainer_fail_times"] -= 1
            raise RuntimeError("transient worker failure (injected)")
        df = pd.read_csv(state["work_path"], low_memory=False)
        y_val, proba = _fit_predict_proba(df, state["target_column"], spec,
                                          state.get("seed", 42), state.get("holdout_frac", 0.2), on_holdout=False)
        f1 = float(f1_score(y_val, (proba >= 0.5).astype(int), zero_division=0))
        exp = {"status": "succeeded", "spec": spec, "f1_default": f1, "score": f1, "threshold": 0.5,
               "roc_auc": float(roc_auc_score(y_val, proba)), "pr_auc": float(average_precision_score(y_val, proba)),
               "y_val": y_val.tolist(), "proba": proba.tolist(), "seconds": round(time.time() - t0, 2)}
        note = f"{spec['model_family']} val F1@0.5 {f1:.3f}"
    except Exception as exc:  # noqa: BLE001 - every failure goes to trainer_repair
        exp = {"status": "failed", "spec": spec, "error": f"{type(exc).__name__}: {exc}"}
        note = f"FAILED: {exp['error'][:80]}"
    return {"exp": exp, "events": [{"agent": "trainer", "note": note}]}


def trainer_repair(state: AdaptiveState) -> dict:
    """Read the error and choose how to retry."""
    exp = state["exp"]
    spec = dict(state["spec"])
    err = exp.get("error", "").lower()
    if "transient" in err or "timeout" in err:
        fix = "retry same spec (transient error)"
    elif "memory" in err or "converge" in err:
        hp = dict(spec["hyperparameters"])
        hp = {k: (max(1, v // 2) if isinstance(v, int) else v) for k, v in hp.items()}
        spec["hyperparameters"] = hp
        fix = "retry with a smaller model"
    else:
        spec = {**spec, "model_family": "logistic_regression", "hyperparameters": {"C": 1.0, "max_iter": 500}}
        fix = "retry with logistic regression (safe fallback)"
    return {"spec": spec, "exp": None,
            "retries_current": state.get("retries_current", 0) + 1,
            "retries_total": state.get("retries_total", 0) + 1,
            "events": [{"agent": "trainer_repair", "note": fix}]}


def threshold_tuner(state: AdaptiveState) -> dict:
    exp = dict(state["exp"])
    y, p = np.array(exp["y_val"]), np.array(exp["proba"])
    scores = {float(t): float(f1_score(y, (p >= t).astype(int), zero_division=0)) for t in THRESHOLDS}
    t_best = max(scores, key=scores.get)
    if scores[t_best] > exp["f1_default"]:
        exp.update(score=scores[t_best], threshold=t_best)
    exp["tuned"] = True
    return {"exp": exp, "events": [{"agent": "threshold_tuner",
                                    "note": f"threshold {exp['threshold']} · F1 {exp['f1_default']:.3f} -> {exp['score']:.3f}"}]}


def evaluator(state: AdaptiveState) -> dict:
    """Record the experiment, update the best model, decide whether to continue."""
    exp = dict(state["exp"])
    record = {k: v for k, v in exp.items() if k not in ("y_val", "proba")}
    experiments = (state.get("experiments") or []) + [record]
    best = state.get("best")
    plateau = state.get("plateau", 0)

    if exp["status"] == "succeeded":
        if best is None or exp["score"] > best["score"] + PLATEAU_DELTA:
            best = {"spec": exp["spec"], "score": exp["score"], "threshold": exp["threshold"]}
            plateau = 0
        else:
            plateau += 1

    stop = None
    n = len(experiments)
    if best is not None and best["score"] >= state.get("target_score", 0.8):
        stop = "target score reached"
    elif n >= state.get("budget", 6):
        stop = "budget spent"
    elif plateau >= PLATEAU_PATIENCE and n >= MIN_EXPERIMENTS:
        stop = "plateau (early stop)"

    note = f"exp {n}: {exp['status']} score {exp.get('score', 0):.3f} · best {best['score'] if best else 0:.3f}"
    if stop:
        note += f" · STOP: {stop}"
    return {"experiments": [record], "best": best, "plateau": plateau, "stop_reason": stop,
            "spec": None, "exp": None, "events": [{"agent": "evaluator", "note": note}]}


def finalize(state: AdaptiveState) -> dict:
    v = state.get("validation") or {}
    if v.get("fatal"):
        return {"status": "failed", "error": "; ".join(v["fatal"]), "final": {},
                "events": [{"agent": "finalize", "note": "stopped: unfixable data"}]}
    best = state.get("best")
    if best is None:
        return {"status": "failed", "error": "no successful experiment", "final": {},
                "events": [{"agent": "finalize", "note": "no model"}]}
    df = pd.read_csv(state["work_path"], low_memory=False)
    y, p = _fit_predict_proba(df, state["target_column"], best["spec"], state.get("seed", 42),
                              state.get("holdout_frac", 0.2), on_holdout=True)
    pred = (p >= best["threshold"]).astype(int)
    final = {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "threshold": best["threshold"],
        "model_family": best["spec"]["model_family"],
        "n_holdout": int(len(y)),
    }
    return {"final": final, "status": "completed",
            "events": [{"agent": "finalize", "note": f"holdout F1 {final['f1']:.3f} · PR-AUC {final['pr_auc']:.3f}"}]}


# ===========================================================================
# Graph
# ===========================================================================
AGENTS = {
    "validator": validator, "data_repair": data_repair, "auditor": auditor, "planner": planner,
    "trainer": trainer, "trainer_repair": trainer_repair, "threshold_tuner": threshold_tuner,
    "evaluator": evaluator, "finalize": finalize,
}


def build_graph():
    g = StateGraph(AdaptiveState)
    g.add_node("supervisor", supervisor)
    for name, fn in AGENTS.items():
        g.add_node(name, fn)
        g.add_edge(name, "supervisor")   # every specialist reports back
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_from_supervisor, ROUTES)
    return g.compile()


def run_adaptive(dataset_path: str, target_column: str, *, budget: int = 6, target_score: float = 0.8,
                 seed: int = 42, holdout_frac: float = 0.2, workdir: str | None = None,
                 trainer_fail_times: int = 0) -> dict[str, Any]:
    """Run the system and return a flat result dict (used by the evaluation suite)."""
    workdir = workdir or str(Path(dataset_path).parent / "_work")
    Path(workdir).mkdir(parents=True, exist_ok=True)
    _FAULTS["trainer_fail_times"] = trainer_fail_times
    t0 = time.time()
    state = build_graph().invoke(
        {"dataset_path": dataset_path, "target_column": target_column, "budget": budget,
         "target_score": target_score, "seed": seed, "holdout_frac": holdout_frac, "workdir": workdir,
         "work_path": dataset_path, "repair_rounds": 0, "retries_total": 0, "plateau": 0},
        {"recursion_limit": 200},
    )
    route = [r for r in state.get("route", []) if r != "end"]
    return {
        "system": "adaptive",
        "status": state.get("status", "failed"),
        "error": state.get("error"),
        "holdout": state.get("final") or {},
        # issues found by the FIRST validation pass (before any repair)
        "validation_issues": next(
            ([i["type"] for i in e.get("issues", [])] for e in state.get("events", []) if e["agent"] == "validator"),
            [],
        ),
        "repairs": state.get("repairs") or [],
        "audit": state.get("audit") or {},
        "dropped_columns": state.get("drops") or [],
        "tuner_used": "threshold_tuner" in route,
        "retries": state.get("retries_total", 0),
        "experiments": len(state.get("experiments") or []),
        "failed_experiments": sum(e["status"] == "failed" for e in state.get("experiments") or []),
        "stop_reason": state.get("stop_reason"),
        "agent_calls": len(route),
        "route": route,
        "events": state.get("events", []),
        "runtime_s": round(time.time() - t0, 2),
    }


def mermaid() -> str:
    """Mermaid text of the compiled graph (rendered in the dashboard)."""
    return build_graph().get_graph().draw_mermaid()
