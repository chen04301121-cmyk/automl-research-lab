"""Simple research loop — one self-contained LangGraph file.

Flow (modified for the IDS706 Week 11 assignment):
  START → data_scientist → data_quality_auditor → hypothesis → ml_engineer
            → (imbalanced? threshold_tuner) → judge
                         ↑__________________________________________|
                         |   (revise: memory → context → hypothesis again)
                         └── accept/stop → holdout_eval → END

New agents (vs. the class version):
  * data_quality_auditor — code-only checks with evidence: single-feature ROC-AUC
    leakage scan, id/constant columns, duplicate rows, class imbalance.
  * threshold_tuner — runs only when the auditor flags class imbalance; picks the
    decision threshold that maximizes validation F1 (holdout then uses that threshold).

LLM / train helpers live in loop_runtime.py (Cursor SDK + OpenRouter + heuristics).
"""

from __future__ import annotations

import json
import operator
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
# Repo root locally; /app in the gateway container (loop_runtime.py is mounted there).
for _p in (REPO, Path("/app")):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from loop_runtime import (  # noqa: E402
    DEFAULT_MODEL,
    FAMILIES,
    _SPEC_SCHEMA_HINT,
    _build_model,
    _coerce_experiment_spec,
    _fit_score,
    _heuristic_plan,
    _heuristic_profile,
    _heuristic_spec,
    _heuristic_verdict,
    _repair_heuristic,
    _resolve_model,
    _validate_spec,
    extract_json_object,
    llm_configured,
    openrouter_chat,
)

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
class SimpleLoopState(TypedDict, total=False):
    dataset_path: str
    target_column: str
    target_score: float
    max_cycles: int
    cycle: int
    seed: int
    holdout_frac: float
    model: str  # OpenRouter / Cursor model id from the UI / env
    profile: dict
    data_audit: dict  # evidence from data_quality_auditor
    imbalanced: bool  # set by the auditor; decides whether threshold_tuner runs
    brief: dict
    hypothesis: dict  # conceptual plan from hypothesis_generator (not HPT numbers)
    experiment_spec: dict  # concrete HPT config from ml_engineer
    execution_result: dict
    verdict: dict
    memory: dict
    best_spec: dict
    best_val_f1: float
    holdout_metrics: dict
    stop_reason: str
    steps: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def node_data_scientist(state: SimpleLoopState) -> dict:
    """Profile CSV via OpenRouter; heuristic fallback if key/parse fails."""
    path = Path(state["dataset_path"])
    target = state["target_column"]
    df = pd.read_csv(path)
    if target not in df.columns:
        return {
            "profile": {},
            "steps": [{"node": "data_scientist", "note": f"target {target!r} missing", "status": "error"}],
        }

    profile = _heuristic_profile(df, target)
    used_llm = False
    note = "heuristic"
    model = _resolve_model(state)

    if llm_configured(model):
        missing = {c: float(df[c].isna().mean()) for c in df.columns}
        nunique = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
        sample = df.head(6).astype(str).replace("nan", "").to_dict(orient="records")
        system = (
            "You are the Data Scientist only — not the hypothesizer and not the ML engineer. "
            "Your job is dataset understanding: profile the CSV and flag risk. "
            "Do NOT propose model families, hyperparameters, or experiment configs. "
            "Return ONLY one JSON object with keys: "
            "n_rows (int), n_columns (int), target (string), target_rate (number|null), "
            "numeric_features (string[]), categorical_features (string[]), "
            "leakage_suspects (string[]), recommended_drops (string[]), "
            "notes (string[], short bullets about data risks — leakage, imbalance, sparsity). "
            "Flag PassengerId/Name/Ticket/Cabin-style identifiers and near-unique text as leakage. "
            "Do not invent columns that are not present."
        )
        user = (
            f"dataset={path.name}\ntarget={target}\nn_rows={len(df)}\n"
            f"columns={list(df.columns)}\n"
            f"dtypes={{{', '.join(f'{c}:{df[c].dtype}' for c in df.columns)}}}\n"
            f"missing_rate={missing}\nnunique={nunique}\n"
            f"sample_rows={sample}\n"
        )
        content, err = openrouter_chat(system, user, model=model, max_tokens=900, temperature=0.15)
        if content:
            parsed = extract_json_object(content)
            if parsed and (
                "recommended_drops" in parsed
                or "leakage_suspects" in parsed
                or "numeric_features" in parsed
            ):
                profile = parsed
                profile.setdefault("n_rows", int(len(df)))
                profile.setdefault("n_columns", int(len(df.columns)))
                profile.setdefault("target", target)
                profile.setdefault("recommended_drops", profile.get("leakage_suspects") or [])
                used_llm = True
                note = f"llm profile · {model}"
            else:
                note = f"llm unparsable; heuristic ({err or 'ok'})"
        else:
            note = f"{err or 'no llm'}; heuristic"

    return {
        "profile": profile,
        "steps": [
            {
                "node": "data_scientist",
                "note": note,
                "used_llm": used_llm,
                "model": model if used_llm else None,
                "output": profile,
            }
        ],
    }


# ---------------------------------------------------------------------------
# NEW AGENT 1: data_quality_auditor (pure code, no LLM)
# ---------------------------------------------------------------------------
LEAKAGE_AUC = 0.95        # a single column this predictive is almost surely leaking the label
IMBALANCE_RATE = 0.20     # minority class below 20% -> treat the task as imbalanced


def node_data_quality_auditor(state: SimpleLoopState) -> dict:
    """Verify the Data Scientist's profile with measured evidence.

    The LLM profile can miss leaky columns (it only sees names and a few rows).
    This agent measures them: each numeric column is scored alone with ROC-AUC
    against the target. It also flags id-like/constant columns, duplicate rows and
    class imbalance, then merges every finding into profile.recommended_drops so
    the hypothesis + ML engineer agents never train on them.
    """
    df = pd.read_csv(state["dataset_path"])
    target = state["target_column"]
    profile = dict(state.get("profile") or {})
    y = df[target]

    # Class balance
    rate = float(y.mean()) if pd.api.types.is_numeric_dtype(y) else None
    minority = min(rate, 1 - rate) if rate is not None else None
    imbalanced = bool(minority is not None and minority < IMBALANCE_RATE)

    # Leakage scan: single-feature ROC-AUC (only meaningful for a binary target)
    leaky: dict[str, float] = {}
    if y.nunique() == 2:
        for col in df.columns:
            if col == target or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            x = df[col].fillna(df[col].median())
            if x.nunique() < 2:
                continue
            auc = float(roc_auc_score(y, x))
            auc = max(auc, 1 - auc)  # direction does not matter
            if auc >= LEAKAGE_AUC:
                leaky[col] = round(auc, 4)

    # Structural problems: ids / near-unique text, constant columns, duplicated rows
    n = len(df)
    id_like = [
        c for c in df.columns
        if c != target
        and (c.lower().endswith("_id") or (not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 0.9 * n))
    ]
    constant = [c for c in df.columns if c != target and df[c].nunique(dropna=False) <= 1]
    dup_rows = int(df.duplicated().sum())

    # Merge findings into the profile used by the downstream agents
    drops = list(dict.fromkeys(
        list(profile.get("recommended_drops") or []) + list(leaky) + id_like + constant
    ))
    suspects = list(dict.fromkeys(list(profile.get("leakage_suspects") or []) + list(leaky)))
    profile["recommended_drops"] = [c for c in drops if c in df.columns and c != target]
    profile["leakage_suspects"] = suspects
    notes = list(profile.get("notes") or [])
    if leaky:
        notes.append(f"auditor: leaky columns by single-feature AUC >= {LEAKAGE_AUC}: {leaky}")
    if imbalanced:
        notes.append(f"auditor: imbalanced target (minority rate {minority:.3f}) -> tune threshold, judge by F1")
    profile["notes"] = notes

    audit = {
        "target_rate": round(rate, 4) if rate is not None else None,
        "imbalanced": imbalanced,
        "leaky_columns_auc": leaky,
        "id_like_columns": id_like,
        "constant_columns": constant,
        "duplicate_rows": dup_rows,
        "final_drops": profile["recommended_drops"],
    }
    note = f"leaky={list(leaky) or 'none'} · imbalanced={imbalanced} · drops={len(profile['recommended_drops'])}"
    return {
        "profile": profile,
        "data_audit": audit,
        "imbalanced": imbalanced,
        "steps": [{"node": "data_quality_auditor", "note": note, "output": audit}],
    }


# ---------------------------------------------------------------------------
# Probability-level fit helper (same splits as loop_runtime._fit_score)
# ---------------------------------------------------------------------------
def _fit_predict_proba(df, target, spec, seed, holdout_frac, *, on_holdout):
    """Fit the spec's pipeline and return (y_eval, positive-class probabilities)."""
    y = df[target]
    drops = [c for c in (spec.get("drop_columns") or []) if c in df.columns and c != target]
    X = df.drop(columns=[target, *drops], errors="ignore")
    strat = y if y.nunique() > 1 else None
    X_tmp, X_hold, y_tmp, y_hold = train_test_split(
        X, y, test_size=holdout_frac, random_state=seed, stratify=strat
    )
    if on_holdout:
        X_train, y_train, X_eval, y_eval = X_tmp, y_tmp, X_hold, y_hold
    else:
        X_train, X_eval, y_train, y_eval = train_test_split(
            X_tmp, y_tmp, test_size=0.25, random_state=seed,
            stratify=y_tmp if y_tmp.nunique() > 1 else None,
        )
    num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    family = str(spec.get("model_family") or "logistic_regression")
    pipe = Pipeline([("pre", pre), ("clf", _build_model(family, dict(spec.get("hyperparameters") or {})))])
    pipe.fit(X_train, y_train)
    return y_eval, pipe.predict_proba(X_eval)[:, 1]


# ---------------------------------------------------------------------------
# NEW AGENT 2: threshold_tuner (pure code; only reached when data is imbalanced)
# ---------------------------------------------------------------------------
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def node_threshold_tuner(state: SimpleLoopState) -> dict:
    """With rare positives the default 0.5 cut-off is rarely F1-optimal.

    Refit the ML engineer's spec, score every threshold on the validation split,
    keep the best one in the spec (holdout reuses it) and update best_val_f1.
    """
    result = dict(state.get("execution_result") or {})
    spec = dict(state.get("experiment_spec") or {})
    df = pd.read_csv(state["dataset_path"])
    target = state["target_column"]
    seed = int(state.get("seed") or 42)
    holdout_frac = float(state.get("holdout_frac") or 0.2)

    y_val, proba = _fit_predict_proba(df, target, spec, seed, holdout_frac, on_holdout=False)
    scores = {float(t): float(f1_score(y_val, (proba >= t).astype(int), zero_division=0)) for t in THRESHOLD_GRID}
    best_t = max(scores, key=scores.get)
    default_f1 = float((result.get("validation_metrics") or {}).get("f1") or 0)
    tuned_f1 = scores[best_t]

    metrics = dict(result.get("validation_metrics") or {})
    metrics.update({
        "f1_at_0.5": round(default_f1, 4),
        "roc_auc": round(float(roc_auc_score(y_val, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_val, proba)), 4),
    })
    if tuned_f1 > default_f1:
        spec["decision_threshold"] = best_t
        metrics["f1"] = tuned_f1
        metrics["decision_threshold"] = best_t
    result["validation_metrics"] = metrics

    patch: dict = {"execution_result": result, "experiment_spec": spec}
    # Keep best_* consistent: ml_engineer may already have stored this spec with its untuned score
    best = state.get("best_spec") or {}
    same_spec = best.get("experiment_id") == spec.get("experiment_id")
    if same_spec or metrics["f1"] > float(state.get("best_val_f1") or -1):
        patch["best_val_f1"] = float(metrics["f1"])
        patch["best_spec"] = spec

    note = (f"threshold {spec.get('decision_threshold', 0.5)} · f1 {default_f1:.3f} → {metrics['f1']:.3f}"
            f" · roc_auc {metrics['roc_auc']:.3f}")
    patch["steps"] = [{
        "node": "threshold_tuner", "note": note, "f1": float(metrics["f1"]),
        "output": {"threshold_f1_curve": scores, "chosen_threshold": spec.get("decision_threshold", 0.5),
                   "validation_metrics": metrics},
    }]
    return patch


def route_after_ml_engineer(state: SimpleLoopState) -> str:
    """Adaptive routing: only imbalanced data with a successful run goes to the tuner."""
    ok = (state.get("execution_result") or {}).get("status") == "succeeded"
    return "tune" if ok and state.get("imbalanced") else "judge"


def node_hypothesis(state: SimpleLoopState) -> dict:
    """Conceptual next experiment from DS profile + Judge feedback — no HPT numbers."""
    profile = state.get("profile") or {}
    brief = state.get("brief") or {}
    verdict = state.get("verdict") or {}
    history = list(brief.get("history") or (state.get("memory") or {}).get("history") or [])
    model = _resolve_model(state)

    plan = _heuristic_plan(profile, history, verdict)
    used_llm = False
    note = "heuristic plan"

    # MODIFIED: stagnation rule. If the last two cycles used the same model family and
    # F1 moved by less than 0.01, stop tuning it and switch family (avoids an endless C sweep).
    recent = [h for h in history if isinstance(h, dict)][-2:]
    if len(recent) == 2:
        fams = {h.get("model_family") for h in recent}
        f1s = [float(h.get("f1") or 0) for h in recent]
        if len(fams) == 1 and abs(f1s[1] - f1s[0]) < 0.01:
            tried = list(dict.fromkeys(h.get("model_family") for h in history if isinstance(h, dict)))
            plan["focus"] = "model_family_switch"
            plan["direction"] = f"F1 plateaued on {recent[-1].get('model_family')}; switch model family."
            plan["avoid_repeats"] = tried
            note = "heuristic plan · stagnation -> family switch"

    if llm_configured(model):
        system = (
            "You are the Hypothesis Generator only — not the Data Scientist and not the ML Engineer. "
            "Read the data-scientist profile and the latest judge verdict / lessons. "
            "Propose ONE conceptual research direction for the next trial. "
            "Do NOT invent concrete hyperparameter values, C/learning_rate/max_depth numbers, "
            "or a full ExperimentSpec — the ML Engineer owns HPT config. "
            "Return ONLY one JSON object with keys: "
            "direction (1-2 sentences: what should change conceptually), "
            "why (cite profile signals and/or judge observation), "
            "focus (one of: baseline_then_iterate|capacity_or_regularization|model_family_switch|feature_drops), "
            "respect_drops (string[] of columns to keep dropped — copy from profile when relevant), "
            "avoid_repeats (string[] of model families or ideas already tried). "
            "Stay conceptual: e.g. 'try a tree ensemble with less regularization' — never 'C=0.5'."
        )
        user = (
            f"profile={json.dumps(profile, default=str)}\n"
            f"latest_verdict={json.dumps(verdict, default=str)}\n"
            f"brief={json.dumps(brief, default=str)}\n"
            f"recent_history={json.dumps(history[-5:], default=str)}\n"
        )
        content, err = openrouter_chat(system, user, model=model, max_tokens=500, temperature=0.4)
        if content:
            parsed = extract_json_object(content)
            if parsed and parsed.get("direction"):
                plan = {
                    "direction": parsed.get("direction"),
                    "why": parsed.get("why") or plan.get("why"),
                    "focus": parsed.get("focus") or plan.get("focus"),
                    "respect_drops": parsed.get("respect_drops")
                    if isinstance(parsed.get("respect_drops"), list)
                    else plan.get("respect_drops"),
                    "avoid_repeats": parsed.get("avoid_repeats")
                    if isinstance(parsed.get("avoid_repeats"), list)
                    else plan.get("avoid_repeats"),
                }
                used_llm = True
                note = f"live conceptual hypothesis · {model}"
            else:
                note = f"llm unparsable; heuristic ({err or 'ok'})"
        else:
            note = f"{err or 'no llm'}; heuristic"

    return {
        "hypothesis": plan,
        "steps": [
            {
                "node": "hypothesis_generator",
                "note": note,
                "hypothesis": plan,
                "output": plan,
                "used_llm": used_llm,
                "model": model if used_llm else None,
            }
        ],
    }


def node_ml_engineer(state: SimpleLoopState) -> dict:
    """Turn conceptual hypothesis into concrete HPT ExperimentSpec, then train/val."""
    df = pd.read_csv(state["dataset_path"])
    target = state["target_column"]
    profile = state.get("profile") or {}
    plan = dict(state.get("hypothesis") or {})
    brief = state.get("brief") or {}
    history = list(brief.get("history") or (state.get("memory") or {}).get("history") or [])
    prev_spec = dict(state.get("experiment_spec") or state.get("best_spec") or {})
    seed = int(state.get("seed") or 42)
    holdout_frac = float(state.get("holdout_frac") or 0.2)
    model_id = _resolve_model(state)

    spec = _heuristic_spec(profile, history, plan)
    used_llm = False
    repaired = False
    note = "heuristic HPT"

    default_drops = (
        plan.get("respect_drops")
        or profile.get("recommended_drops")
        or profile.get("leakage_suspects")
        or []
    )
    next_eid = f"exp-{len(history) + 1}"
    raw_llm: str | None = None

    if llm_configured(model_id):
        system = (
            "You are the ML Engineer only — not the Hypothesis Generator and not the Judge. "
            "The hypothesis agent already decided the conceptual direction. "
            "Your job: translate that plan into a concrete hyperparameter / training config. "
            "Return ONLY one JSON object matching this schema exactly: "
            f"{_SPEC_SCHEMA_HINT} "
            "model_family MUST be one of logistic_regression|random_forest|hist_gradient_boosting. "
            "hyperparameters MUST be a JSON object with concrete numbers for that family "
            "(C/max_iter OR n_estimators/max_depth OR learning_rate/max_iter). "
            "drop_columns is a string array. rationale is one sentence. "
            "No markdown fences, no commentary."
        )
        user = (
            f"hypothesis_plan={json.dumps(plan, default=str)}\n"
            f"profile={json.dumps({k: profile.get(k) for k in ('recommended_drops', 'leakage_suspects', 'n_rows', 'target_rate')}, default=str)}\n"
            f"prior_spec={json.dumps(prev_spec, default=str)}\n"
            f"recent_history={json.dumps(history[-3:], default=str)}\n"
            f"allowed_families={list(FAMILIES)}\n"
            f"next_experiment_id={next_eid}\n"
        )
        content, err = openrouter_chat(
            system, user, model=model_id, max_tokens=700, temperature=0.2, json_object=True
        )
        raw_llm = content
        coerced = _coerce_experiment_spec(
            extract_json_object(content or ""),
            experiment_id=next_eid,
            default_drops=default_drops,
        )
        if coerced is None and content and llm_configured(model_id):
            # One strict repair pass — common with Cursor prose wrapping JSON.
            fix_system = (
                "Convert the text into ONE valid ExperimentSpec JSON object. "
                f"Schema example: {_SPEC_SCHEMA_HINT} "
                f"model_family must be one of {list(FAMILIES)}. "
                "Output JSON only."
            )
            fix_user = (
                f"raw_model_output={content[:3500]}\n"
                f"plan={json.dumps(plan, default=str)}\n"
                f"next_experiment_id={next_eid}\n"
            )
            content2, err2 = openrouter_chat(
                fix_system,
                fix_user,
                model=model_id,
                max_tokens=500,
                temperature=0.0,
                json_object=True,
            )
            raw_llm = content2 or content
            coerced = _coerce_experiment_spec(
                extract_json_object(content2 or ""),
                experiment_id=next_eid,
                default_drops=default_drops,
            )
            err = err2 or err
        if coerced:
            spec = coerced
            used_llm = True
            note = f"live HPT config · {model_id}"
        else:
            note = f"llm unparsable HPT; heuristic ({err or 'ok'})"

    err = _validate_spec(spec)
    if err and llm_configured(model_id):
        system = (
            "You are the ML Engineer repairing an invalid HPT ExperimentSpec. "
            f"Return ONLY JSON matching: {_SPEC_SCHEMA_HINT} "
            f"model_family must be one of {list(FAMILIES)}."
        )
        user = (
            f"error={err}\n"
            f"spec={json.dumps(spec, default=str)}\n"
            f"plan={json.dumps(plan, default=str)}\n"
            f"raw_model_output={(raw_llm or '')[:2000]}\n"
        )
        content, _llm_err = openrouter_chat(
            system, user, model=model_id, max_tokens=500, temperature=0.0, json_object=True
        )
        coerced = _coerce_experiment_spec(
            extract_json_object(content or ""),
            experiment_id=next_eid,
            default_drops=default_drops,
        )
        if coerced:
            spec = coerced
            used_llm = True
            repaired = True
            note = f"repaired HPT after: {err}"
            err = _validate_spec(spec)
    if err:
        spec = _repair_heuristic(spec, err)
        repaired = True
        note = f"heuristic repair after: {err}"
        err = _validate_spec(spec)

    cycle = int(state.get("cycle") or 0) + 1
    if err:
        result = {"status": "failed", "error": err, "validation_metrics": {}}
        return {
            "execution_result": result,
            "experiment_spec": spec,
            "cycle": cycle,
            "steps": [
                {
                    "node": "ml_engineer",
                    "note": note,
                    "status": "failed",
                    "repaired": repaired,
                    "used_llm": used_llm,
                    "experiment_id": spec.get("experiment_id"),
                    "model_family": spec.get("model_family"),
                    "spec": spec,
                    "output": {"experiment_spec": spec, "error": err},
                }
            ],
        }

    try:
        result = _fit_score(df, target, spec, seed, holdout_frac, on_holdout=False)
    except Exception as exc:  # noqa: BLE001
        if not repaired:
            spec = _repair_heuristic(spec, str(exc))
            repaired = True
            note = f"heuristic repair after run error: {exc}"
            try:
                result = _fit_score(df, target, spec, seed, holdout_frac, on_holdout=False)
            except Exception as exc2:  # noqa: BLE001
                result = {"status": "failed", "error": str(exc2), "validation_metrics": {}}
        else:
            result = {"status": "failed", "error": str(exc), "validation_metrics": {}}

    f1 = float((result.get("validation_metrics") or {}).get("f1") or 0)
    patch: dict = {
        "execution_result": result,
        "experiment_spec": spec,
        "cycle": cycle,
        "steps": [
            {
                "node": "ml_engineer",
                "note": note if result.get("status") != "succeeded" else f"{note} · f1={f1:.3f}",
                "f1": f1 if result.get("status") == "succeeded" else None,
                "status": result.get("status"),
                "repaired": repaired,
                "used_llm": used_llm,
                "experiment_id": spec.get("experiment_id"),
                "model_family": spec.get("model_family"),
                "validation_metrics": result.get("validation_metrics") or {},
                "duration_seconds": result.get("duration_seconds"),
                "spec": spec,
                "output": {
                    "experiment_spec": spec,
                    "execution_result": {
                        "status": result.get("status"),
                        "validation_metrics": result.get("validation_metrics"),
                        "error": result.get("error"),
                    },
                },
            }
        ],
    }
    if result.get("status") == "succeeded" and f1 > float(state.get("best_val_f1") or -1):
        patch["best_val_f1"] = f1
        patch["best_spec"] = spec
    return patch


def node_judge(state: SimpleLoopState) -> dict:
    """Verdict via OpenRouter; heuristic fallback."""
    result = state.get("execution_result") or {}
    spec = state.get("experiment_spec") or {}
    plan = state.get("hypothesis") or {}
    brief = state.get("brief") or {}
    target_score = float(state.get("target_score") or 0.8)
    model = _resolve_model(state)

    verdict = _heuristic_verdict(result, target_score)
    used_llm = False
    note = "heuristic"

    if llm_configured(model):
        system = (
            "You are the Judge only — not the Hypothesis Generator and not the ML Engineer. "
            "Decide from VALIDATION evidence only — never invent holdout scores, "
            "never rewrite hyperparameters. "
            "Return ONLY one JSON object with keys: "
            "action (accept|revise|reject), observation (1-2 sentences with the val F1), "
            "next_action (short conceptual instruction for the hypothesis agent — no HPT numbers). "
            "accept only if validation F1 meets or exceeds target_score; "
            "revise when under target but recoverable; reject for hopeless/broken runs."
        )
        user = (
            f"target_score={target_score}\n"
            f"hypothesis_plan={json.dumps(plan, default=str)}\n"
            f"spec={json.dumps(spec, default=str)}\n"
            f"result={json.dumps(result, default=str)}\n"
            f"brief={json.dumps(brief, default=str)}\n"
        )
        content, err = openrouter_chat(system, user, model=model, max_tokens=500, temperature=0.1)
        if content:
            parsed = extract_json_object(content)
            if parsed and parsed.get("action") in {"accept", "revise", "reject"}:
                verdict = {
                    "action": parsed["action"],
                    "observation": parsed.get("observation") or verdict.get("observation"),
                    "next_action": parsed.get("next_action") or verdict.get("next_action"),
                }
                used_llm = True
                note = f"llm verdict · {model}"
            else:
                note = f"llm unparsable; heuristic ({err or 'ok'})"
        else:
            note = f"{err or 'no llm'}; heuristic"

    return {
        "verdict": verdict,
        "steps": [
            {
                "node": "judge",
                "note": note,
                "verdict": verdict,
                "used_llm": used_llm,
                "model": model if used_llm else None,
            }
        ],
    }


def node_memory_writer(state: SimpleLoopState) -> dict:
    """Append this cycle's lesson so the next hypothesis can see history."""
    mem = dict(state.get("memory") or {"history": [], "findings": [], "lessons": []})
    hist = list(mem.get("history") or [])
    hist.append(
        {
            "cycle": state.get("cycle"),
            "hypothesis": state.get("hypothesis"),
            "spec": state.get("experiment_spec"),
            "model_family": (state.get("experiment_spec") or {}).get("model_family"),
            "f1": (state.get("execution_result") or {}).get("validation_metrics", {}).get("f1"),
            "verdict": state.get("verdict"),
        }
    )
    mem["history"] = hist
    lessons = list(mem.get("lessons") or [])
    lessons.append(str((state.get("verdict") or {}).get("observation") or ""))
    mem["lessons"] = lessons
    return {"memory": mem, "steps": [{"node": "memory_writer", "note": "filed lesson"}]}


def node_context_builder(state: SimpleLoopState) -> dict:
    """Build a short brief for the next hypothesis from recent memory."""
    mem = state.get("memory") or {}
    brief = {
        "history": mem.get("history") or [],
        "lessons": (mem.get("lessons") or [])[-3:],
    }
    return {"brief": brief, "steps": [{"node": "context_builder", "note": "next brief"}]}


def node_holdout(state: SimpleLoopState) -> dict:
    """Score the best spec ONCE on the sealed holdout (same seed as ml_engineer)."""
    df = pd.read_csv(state["dataset_path"])
    target = state["target_column"]
    spec = state.get("best_spec") or state.get("experiment_spec") or {}
    seed = int(state.get("seed") or 42)
    holdout_frac = float(state.get("holdout_frac") or 0.2)
    if _validate_spec(spec):
        spec = _repair_heuristic(dict(spec), "invalid best_spec")
    # Score with probabilities so the tuned threshold (if any) is applied, and
    # report threshold-free metrics (ROC-AUC, PR-AUC) that suit imbalanced data.
    threshold = float(spec.get("decision_threshold", 0.5))
    y_hold, proba = _fit_predict_proba(df, target, spec, seed, holdout_frac, on_holdout=True)
    pred = (proba >= threshold).astype(int)
    metrics = {
        "f1": float(f1_score(y_hold, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_hold, pred)),
        "roc_auc": float(roc_auc_score(y_hold, proba)) if y_hold.nunique() > 1 else None,
        "pr_auc": float(average_precision_score(y_hold, proba)) if y_hold.nunique() > 1 else None,
        "decision_threshold": threshold,
        "positive_rate": float(y_hold.mean()),
        "n_holdout": int(len(y_hold)),
    }
    return {
        "holdout_metrics": metrics,
        "stop_reason": "finalize",
        "steps": [{"node": "holdout_eval", "note": "sealed holdout", "metrics": metrics}],
    }


def route_after_judge(state: SimpleLoopState) -> str:
    """Router = plain code (not an LLM)."""
    action = str((state.get("verdict") or {}).get("action") or "stop")
    cycle = int(state.get("cycle") or 0)
    max_cycles = int(state.get("max_cycles") or 3)
    if action == "accept":
        return "holdout"
    if action == "revise" and cycle < max_cycles:
        return "memory"
    return "holdout"


# ---------------------------------------------------------------------------
# GRAPH WIRING
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(SimpleLoopState)

    g.add_node("data_scientist", node_data_scientist)
    g.add_node("data_quality_auditor", node_data_quality_auditor)   # NEW
    g.add_node("hypothesis_generator", node_hypothesis)
    g.add_node("ml_engineer", node_ml_engineer)
    g.add_node("threshold_tuner", node_threshold_tuner)             # NEW
    g.add_node("judge", node_judge)
    g.add_node("memory_writer", node_memory_writer)
    g.add_node("context_builder", node_context_builder)
    g.add_node("holdout_eval", node_holdout)

    g.add_edge(START, "data_scientist")
    g.add_edge("data_scientist", "data_quality_auditor")
    g.add_edge("data_quality_auditor", "hypothesis_generator")
    g.add_edge("hypothesis_generator", "ml_engineer")
    # NEW conditional edge: imbalanced data -> threshold_tuner, otherwise straight to judge
    g.add_conditional_edges(
        "ml_engineer",
        route_after_ml_engineer,
        {"tune": "threshold_tuner", "judge": "judge"},
    )
    g.add_edge("threshold_tuner", "judge")
    g.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "memory": "memory_writer",
            "holdout": "holdout_eval",
        },
    )
    g.add_edge("memory_writer", "context_builder")
    g.add_edge("context_builder", "hypothesis_generator")  # loop back
    g.add_edge("holdout_eval", END)
    return g.compile()


REQUIRED_NODES = frozenset(
    {
        "data_scientist",
        "data_quality_auditor",
        "threshold_tuner",
        "hypothesis_generator",
        "ml_engineer",
        "judge",
        "memory_writer",
        "context_builder",
        "holdout_eval",
    }
)


def assess_graph_ready(app) -> dict:
    """Return readiness dict from a compiled LangGraph app."""
    raw = app.get_graph().nodes
    present = {n for n in raw if n not in {START, END, "__start__", "__end__"}}
    missing = sorted(REQUIRED_NODES - present)
    return {
        "ready": not missing,
        "nodes": sorted(present),
        "missing_nodes": missing,
        "error": None if not missing else f"missing nodes: {', '.join(missing)}",
    }


def get_status() -> dict:
    """Import-safe status for the gateway / UI."""
    try:
        app = build_graph()
    except NotImplementedError as exc:
        return {
            "ready": False,
            "nodes": [],
            "missing_nodes": sorted(REQUIRED_NODES),
            "error": str(exc) or "build_graph not implemented",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ready": False,
            "nodes": [],
            "missing_nodes": sorted(REQUIRED_NODES),
            "error": str(exc),
        }
    return assess_graph_ready(app)


def _initial_state(
    *,
    dataset_path: str,
    target_column: str,
    target_score: float,
    max_cycles: int,
    seed: int,
    holdout_frac: float,
    model: str | None = None,
) -> SimpleLoopState:
    state: SimpleLoopState = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "target_column": target_column,
        "target_score": target_score,
        "max_cycles": max_cycles,
        "cycle": 0,
        "seed": seed,
        "holdout_frac": holdout_frac,
        "best_val_f1": -1.0,
        "steps": [],
        "memory": {"history": [], "findings": [], "lessons": []},
        "brief": {},
    }
    chosen = (model or "").strip() or DEFAULT_MODEL
    state["model"] = chosen
    return state


def _finalize_payload(final: dict) -> dict:
    return {
        "dataset_path": final.get("dataset_path"),
        "target_column": final.get("target_column"),
        "cycles": final.get("cycle"),
        "best_val_f1": final.get("best_val_f1"),
        "best_spec": final.get("best_spec"),
        "holdout_metrics": final.get("holdout_metrics"),
        "verdict": final.get("verdict"),
        "stop_reason": final.get("stop_reason"),
        "profile": final.get("profile"),
        "data_audit": final.get("data_audit"),
        "memory": final.get("memory"),
        "model": final.get("model"),
        "steps": final.get("steps") or [],
    }


def run_solution_loop(
    *,
    dataset_path: str,
    target_column: str = "Survived",
    target_score: float = 0.8,
    max_cycles: int = 3,
    seed: int = 42,
    holdout_frac: float = 0.2,
    model: str | None = None,
) -> dict:
    """Run the whole graph to completion; return a summary dict."""
    app = build_graph()
    initial = _initial_state(
        dataset_path=dataset_path,
        target_column=target_column,
        target_score=target_score,
        max_cycles=max_cycles,
        seed=seed,
        holdout_frac=holdout_frac,
        model=model,
    )
    final = app.invoke(initial, config={"recursion_limit": 10 + max_cycles * 8})
    return _finalize_payload(final)


def stream_solution_loop(
    *,
    dataset_path: str,
    target_column: str = "Survived",
    target_score: float = 0.8,
    max_cycles: int = 3,
    seed: int = 42,
    holdout_frac: float = 0.2,
    model: str | None = None,
):
    """Yield UI events as each node starts/finishes (used by the gateway SSE)."""
    app = build_graph()
    initial = _initial_state(
        dataset_path=dataset_path,
        target_column=target_column,
        target_score=target_score,
        max_cycles=max_cycles,
        seed=seed,
        holdout_frac=holdout_frac,
        model=model,
    )
    yield {
        "type": "run_started",
        "dataset_path": str(Path(dataset_path).resolve()),
        "target_column": target_column,
        "model": initial.get("model"),
    }

    merged: dict = dict(initial)
    last_node: str | None = None
    try:
        for update in app.stream(
            initial, stream_mode="updates", config={"recursion_limit": 10 + max_cycles * 8}
        ):
            if not isinstance(update, dict):
                continue
            for node_name, patch in update.items():
                if last_node == "judge" and node_name in {"memory_writer", "holdout_eval"}:
                    action = str((merged.get("verdict") or {}).get("action") or "stop")
                    yield {
                        "type": "router",
                        "node": "router",
                        "status": "completed",
                        "action": action,
                        "route": "memory" if node_name == "memory_writer" else "holdout",
                    }
                    if node_name == "memory_writer":
                        yield {
                            "type": "node_completed",
                            "node": "research_memory",
                            "status": "completed",
                            "note": "history + lessons filed",
                        }

                yield {"type": "node_started", "node": node_name, "status": "running"}
                if isinstance(patch, dict):
                    steps = list(patch.get("steps") or [])
                    # Mirror Annotated[list, operator.add] — stream patches omit prior steps.
                    merged.update({k: v for k, v in patch.items() if k != "steps"})
                    if steps:
                        merged["steps"] = list(merged.get("steps") or []) + steps
                    note = steps[-1].get("note") if steps else None
                    yield {
                        "type": "node_completed",
                        "node": node_name,
                        "status": "completed",
                        "note": note,
                        "step": steps[-1] if steps else None,
                        "cycle": merged.get("cycle"),
                        "best_val_f1": merged.get("best_val_f1"),
                        "verdict": merged.get("verdict"),
                    }
                else:
                    yield {"type": "node_completed", "node": node_name, "status": "completed"}
                last_node = node_name
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": str(exc)}
        return

    yield {"type": "run_completed", "result": _finalize_payload(merged)}


if __name__ == "__main__":
    ds = REPO / "datasets" / "Titanic-Dataset.csv"
    result = run_solution_loop(dataset_path=str(ds), target_column="Survived", max_cycles=3)
    print(json.dumps(result, indent=2, default=str))
