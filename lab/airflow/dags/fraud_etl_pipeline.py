r"""Fraud ETL pipeline (IDS706 Week 11 mini-assignment, Part 1).

Flow (Graph tab in the Airflow UI):

    [customers]    extract -> clean ----------------------\
    [transactions] extract -> clean ------------------------> merge -> load_to_postgres -> quality_check
    [cards]        extract -> clean -> aggregate_per_customer /                                  |
                                                                    +-----------------+----------+--------+
                                                                    v                 v                   v
                                                             eda_from_db     train_models_from_db   export_for_agents
                                                                    \_________________|__________________/
                                                                                      v
                                                                                write_report

* The three TaskGroups do not depend on each other, so Airflow runs them in parallel.
* The three analysis tasks after quality_check also run in parallel.
* Every analysis task reads the cleaned table back from PostgreSQL (not from the CSV files).
* Large data moves through files / the database; XCom only carries small dicts (paths, counts).
"""

from __future__ import annotations

import io
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from airflow.decorators import dag, task
from airflow.utils.task_group import TaskGroup

# ---------------------------------------------------------------------------
# Paths inside the Airflow containers (/repo is the project root on the host)
# ---------------------------------------------------------------------------
REPO = Path(os.getenv("FRAUD_REPO_ROOT", "/repo"))
DATA_DIR = REPO / "datasets" / "airflow"
CUSTOMERS_CSV = DATA_DIR / "customers.csv"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
CARDS_CSV = DATA_DIR / "cards.csv"

RUNS_DIR = REPO / "artifacts" / "airflow" / "fraud_runs"  # per-run working files
REPORT_DIR = REPO / "reports" / "part1"                   # stable copy for the README
AGENT_EXPORT = DATA_DIR / "fraud_clean_latest.csv"        # input for the Part 2 agent loop

# ---------------------------------------------------------------------------
# Database: the Postgres container that ships with this Airflow stack
# ---------------------------------------------------------------------------
DB_DSN = os.getenv(
    "FRAUD_DB_DSN",
    "host=airflow-db port=5432 dbname=airflow user=airflow password=airflow",
)
CLEAN_TABLE = "fraud_transactions_clean"   # final merged + cleaned table
RUNS_TABLE = "fraud_pipeline_runs"         # one audit row per pipeline run

TARGET = "is_fraud"
JOIN_KEY = "customer_id"
# fraud_probability is an upstream score that perfectly encodes the label (data leakage)
LEAKY_COLUMNS = ["fraud_probability"]
ID_COLUMNS = ["transaction_id", "customer_id", "merchant_id"]
SEED = 42

DEFAULT_ARGS = {
    "owner": "chen",
    "retries": 2,                          # retry a failed task twice
    "retry_delay": timedelta(seconds=10),
}


# ===========================================================================
# Small helpers (plain Python, shared by several tasks)
# ===========================================================================
def run_dir(context: dict) -> Path:
    """Folder for this run's intermediate files, e.g. fraud_runs/20261109T000000."""
    path = RUNS_DIR / context["ts_nodash"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def yes_no_to_int(series: pd.Series) -> pd.Series:
    """Map 'Yes'/'No' strings to 1/0 (anything else becomes NaN)."""
    return series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})


def extract_csv(path: Path, required: list[str], unique_key: str) -> dict:
    """Shared extract step: schema, completeness and uniqueness gates."""
    if not path.is_file():
        raise FileNotFoundError(f"source file missing: {path}")
    df = pd.read_csv(path)

    # Completeness gate: required columns must exist and must not be empty
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path.name}: missing columns {missing_cols}")
    if df.empty:
        raise ValueError(f"{path.name}: zero rows received")
    null_keys = int(df[required].isna().any(axis=1).sum())

    # Uniqueness gate: the primary key must identify each row once
    dup_keys = int(df[unique_key].duplicated().sum())

    return {
        "path": str(path),
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "null_required_values": null_keys,
        "duplicate_keys": dup_keys,
    }


def pg_connect():
    """Open a psycopg2 connection to the pipeline database."""
    import psycopg2  # imported lazily so the DAG file parses fast

    return psycopg2.connect(DB_DSN)


def sql_type(dtype) -> str:
    """Translate a pandas dtype into a PostgreSQL column type."""
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    return "TEXT"


def write_table(df: pd.DataFrame, table: str) -> None:
    """Replace `table` with `df` in one transaction (DROP + CREATE + COPY).

    Replacing (not appending) makes reruns idempotent: running the DAG twice
    for the same day can never double the rows.
    """
    columns_sql = ", ".join(f'"{c}" {sql_type(t)}' for c, t in df.dtypes.items())
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, date_format="%Y-%m-%d %H:%M:%S")
    buffer.seek(0)
    col_list = ", ".join(f'"{c}"' for c in df.columns)
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{table}"')
        cur.execute(f'CREATE TABLE "{table}" ({columns_sql})')
        cur.copy_expert(f'COPY "{table}" ({col_list}) FROM STDIN WITH (FORMAT csv)', buffer)
    conn.close()


def read_sql(query: str) -> pd.DataFrame:
    """Run a SELECT and return the rows as a DataFrame."""
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=cols)


def model_frame(df: pd.DataFrame, include_leaky: bool = False):
    """Build X / y for modeling: drop ids, raw timestamp and (by default) leaky columns."""
    drop = [TARGET, "timestamp", *ID_COLUMNS]
    if not include_leaky:
        drop += LEAKY_COLUMNS
    X = df.drop(columns=[c for c in drop if c in df.columns])
    X = pd.get_dummies(X, drop_first=True, dtype=float).fillna(0)
    y = df[TARGET].astype(int)
    return X, y


# ===========================================================================
# DAG
# ===========================================================================
@dag(
    dag_id="fraud_etl_pipeline",
    description="3 sources -> parallel clean (TaskGroups) -> merge -> Postgres -> parallel analysis",
    start_date=datetime(2026, 11, 1),
    schedule="@daily",          # new transactions arrive daily -> refresh once a day (00:00 UTC)
    catchup=False,              # do not backfill every day since start_date when unpaused
    max_active_runs=1,          # never let two runs overwrite the same table at the same time
    default_args=DEFAULT_ARGS,
    tags=["ids706", "fraud", "etl", "postgres"],
)
def fraud_etl_pipeline():
    # -----------------------------------------------------------------------
    # TaskGroup 1: customers (dimension table, 1 row per customer)
    # -----------------------------------------------------------------------
    with TaskGroup(group_id="customers", tooltip="Extract + clean customers.csv"):

        @task(task_id="extract")
        def extract_customers() -> dict:
            return extract_csv(CUSTOMERS_CSV, [JOIN_KEY, "age", "credit_score"], JOIN_KEY)

        @task(task_id="clean")
        def clean_customers(meta: dict, **context) -> dict:
            df = pd.read_csv(meta["path"], keep_default_na=False, na_values=[""])
            n_in = len(df)

            # Uniqueness: keep one row per customer
            df = df.drop_duplicates(subset=JOIN_KEY, keep="first")

            # Validity: numeric fields must be numbers inside a sensible range
            numeric_rules = {
                "age": (18, 100),
                "credit_score": (300, 900),
                "annual_income": (0, None),
                "avg_monthly_balance": (0, None),
                "total_accounts": (0, None),
            }
            invalid_counts = {}
            for col, (low, high) in numeric_rules.items():
                values = pd.to_numeric(df[col], errors="coerce")   # "987 note:11" -> NaN
                bad = values.isna() | (values < low)
                if high is not None:
                    bad |= values > high
                invalid_counts[col] = int(bad.sum())
                df[col] = values.where(~bad).fillna(values[~bad].median())

            # Dates -> datetime; city/state are dropped (high-cardinality, often inconsistent)
            df["customer_since"] = pd.to_datetime(df["customer_since"], errors="coerce")
            df["loan_status"] = df["loan_status"].replace({"None": "no_loan"}).fillna("no_loan")
            df = df.drop(columns=["city", "state"])
            df = df.rename(columns={"country": "customer_country"})

            out = run_dir(context) / "customers_clean.csv"
            df.to_csv(out, index=False)
            return {"path": str(out), "rows_in": n_in, "rows_out": int(len(df)),
                    "invalid_values_imputed": invalid_counts}

        customers_clean = clean_customers(extract_customers())

    # -----------------------------------------------------------------------
    # TaskGroup 2: transactions (fact table, 1 row per transaction, has the label)
    # -----------------------------------------------------------------------
    with TaskGroup(group_id="transactions", tooltip="Extract + clean transactions.csv"):

        @task(task_id="extract")
        def extract_transactions() -> dict:
            return extract_csv(
                TRANSACTIONS_CSV, ["transaction_id", JOIN_KEY, "amount", TARGET], "transaction_id"
            )

        @task(task_id="clean")
        def clean_transactions(meta: dict, **context) -> dict:
            df = pd.read_csv(meta["path"])
            n_in = len(df)
            out_dir = run_dir(context)

            # Uniqueness: one row per transaction id
            df = df.drop_duplicates(subset="transaction_id", keep="first")

            # Validity: parse types; rows that break hard rules go to quarantine (not silently dropped)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
            bad = df["timestamp"].isna() | df["amount"].isna() | (df["amount"] < 0) | ~df[TARGET].isin([0, 1])
            quarantine = df[bad]
            quarantine.to_csv(out_dir / "transactions_quarantine.csv", index=False)
            df = df[~bad].copy()
            df[TARGET] = df[TARGET].astype(int)

            # Transform: Yes/No flags -> 0/1, drop constant column, add simple features
            for col in ["weekend", "international"]:
                df[col] = yes_no_to_int(df[col]).fillna(0).astype(int)
            df = df.drop(columns=["currency"])                 # always INR -> no information
            df["log_amount"] = np.log1p(df["amount"])
            df["amount_vs_30d_avg"] = df["amount"] / (df["average_spending_30d"] + 1.0)
            df["is_night"] = (df["transaction_hour"] < 6).astype(int)

            out = out_dir / "transactions_clean.csv"
            df.to_csv(out, index=False)
            return {"path": str(out), "rows_in": n_in, "rows_out": int(len(df)),
                    "quarantined": int(len(quarantine)),
                    "fraud_rate": round(float(df[TARGET].mean()), 4)}

        transactions_clean = clean_transactions(extract_transactions())

    # -----------------------------------------------------------------------
    # TaskGroup 3: cards (many cards per customer -> aggregate before joining)
    # -----------------------------------------------------------------------
    with TaskGroup(group_id="cards", tooltip="Extract + clean + aggregate cards.csv"):

        @task(task_id="extract")
        def extract_cards() -> dict:
            return extract_csv(CARDS_CSV, ["card_id", JOIN_KEY, "card_limit"], "card_id")

        @task(task_id="clean")
        def clean_cards(meta: dict, **context) -> dict:
            df = pd.read_csv(meta["path"])
            n_in = len(df)
            df = df.drop_duplicates(subset="card_id", keep="first")
            df["card_limit"] = pd.to_numeric(df["card_limit"], errors="coerce").clip(lower=0)
            # Validity: utilization is a share, so it must stay inside [0, 1]
            df["utilization_rate"] = pd.to_numeric(df["utilization_rate"], errors="coerce").clip(0, 1)
            df["reward_program"] = yes_no_to_int(df["reward_program"]).fillna(0).astype(int)
            out = run_dir(context) / "cards_clean.csv"
            df.to_csv(out, index=False)
            return {"path": str(out), "rows_in": n_in, "rows_out": int(len(df))}

        @task(task_id="aggregate_per_customer")
        def aggregate_cards(meta: dict, **context) -> dict:
            """Collapse to 1 row per customer so the later join cannot multiply rows."""
            df = pd.read_csv(meta["path"])
            agg = df.groupby(JOIN_KEY).agg(
                n_cards=("card_id", "count"),
                n_credit_cards=("card_type", lambda s: int((s == "Credit").sum())),
                total_card_limit=("card_limit", "sum"),
                avg_utilization=("utilization_rate", "mean"),
                max_utilization=("utilization_rate", "max"),
                has_rewards=("reward_program", "max"),
                avg_card_age=("card_age", "mean"),
            ).reset_index()
            out = run_dir(context) / "cards_per_customer.csv"
            agg.to_csv(out, index=False)
            return {"path": str(out), "n_customers_with_cards": int(len(agg))}

        cards_agg = aggregate_cards(clean_cards(extract_cards()))

    # -----------------------------------------------------------------------
    # Merge the three cleaned sources (waits for all three TaskGroups)
    # -----------------------------------------------------------------------
    @task
    def merge(tx_meta: dict, cust_meta: dict, cards_meta: dict, **context) -> dict:
        tx = pd.read_csv(tx_meta["path"], parse_dates=["timestamp"])
        cust = pd.read_csv(cust_meta["path"], parse_dates=["customer_since"])
        cards = pd.read_csv(cards_meta["path"])

        # validate="many_to_one" makes pandas fail loudly if a key is duplicated on the right
        merged = tx.merge(cust, on=JOIN_KEY, how="left", validate="many_to_one")
        merged = merged.merge(cards, on=JOIN_KEY, how="left", validate="many_to_one")

        # Customers without any card get zeros instead of NaN
        card_cols = [c for c in cards.columns if c != JOIN_KEY]
        merged["has_card"] = merged["n_cards"].notna().astype(int)
        merged[card_cols] = merged[card_cols].fillna(0)

        # Cross-table feature + consistency check: tenure at the time of the transaction
        merged["tenure_days"] = (merged["timestamp"] - merged["customer_since"]).dt.days
        before_signup = int((merged["tenure_days"] < 0).sum())
        merged["tx_before_signup"] = (merged["tenure_days"] < 0).astype(int)
        merged["tenure_days"] = merged["tenure_days"].clip(lower=0)
        merged = merged.drop(columns=["customer_since"])

        # Reconcile counts at the join boundary: a left join must keep every transaction once
        if len(merged) != tx_meta["rows_out"]:
            raise ValueError(f"join changed row count: {tx_meta['rows_out']} -> {len(merged)}")
        unmatched = int(merged["age"].isna().sum())

        out = run_dir(context) / "merged.csv"
        merged.to_csv(out, index=False)
        return {"path": str(out), "n_rows": int(len(merged)), "n_columns": int(merged.shape[1]),
                "unmatched_customers": unmatched, "tx_before_signup": before_signup}

    # -----------------------------------------------------------------------
    # Load into PostgreSQL
    # -----------------------------------------------------------------------
    @task
    def load_to_postgres(merge_meta: dict, **context) -> dict:
        df = pd.read_csv(merge_meta["path"], parse_dates=["timestamp"])
        write_table(df, CLEAN_TABLE)

        # Audit trail: append one row per run (keeps history instead of overwriting it)
        with pg_connect() as conn, conn.cursor() as cur:
            cur.execute(
                f'CREATE TABLE IF NOT EXISTS "{RUNS_TABLE}" ('
                "run_ts TEXT, loaded_at TIMESTAMP, n_rows BIGINT, n_columns INT)"
            )
            cur.execute(
                f'INSERT INTO "{RUNS_TABLE}" VALUES (%s, NOW(), %s, %s)',
                (context["ts_nodash"], int(len(df)), int(df.shape[1])),
            )
        conn.close()
        return {"table": CLEAN_TABLE, "rows_sent": int(len(df))}

    # -----------------------------------------------------------------------
    # Quality gate on what actually landed in the database
    # -----------------------------------------------------------------------
    @task
    def quality_check(load_meta: dict) -> dict:
        stats = read_sql(
            f'SELECT COUNT(*) AS n_rows, COUNT(DISTINCT transaction_id) AS n_ids, '
            f'SUM(CASE WHEN "{TARGET}" NOT IN (0, 1) THEN 1 ELSE 0 END) AS bad_labels, '
            f'SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) AS null_amounts '
            f'FROM "{CLEAN_TABLE}"'
        ).iloc[0]
        n_rows, n_ids = int(stats["n_rows"]), int(stats["n_ids"])
        checks = {
            "completeness_rows_loaded_equal_rows_sent": n_rows == load_meta["rows_sent"],
            "uniqueness_transaction_id": n_rows == n_ids,
            "validity_label_is_binary": int(stats["bad_labels"] or 0) == 0,
            "completeness_no_null_amount": int(stats["null_amounts"] or 0) == 0,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            # Fail loudly: a green run must mean the data is complete (job success != data completeness)
            raise ValueError(f"quality checks failed: {failed}")
        return {"rows_in_db": n_rows, "checks": checks}

    # -----------------------------------------------------------------------
    # Analysis 1: EDA charts (reads from Postgres)
    # -----------------------------------------------------------------------
    @task
    def eda_from_db(qc: dict, **context) -> dict:
        import matplotlib

        matplotlib.use("Agg")  # no screen inside the container
        import matplotlib.pyplot as plt

        df = read_sql(
            f'SELECT "{TARGET}", amount, transaction_hour, payment_method, '
            f'merchant_category, fraud_probability FROM "{CLEAN_TABLE}"'
        )
        out_dir = run_dir(context) / "plots"
        out_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        df[TARGET].value_counts().sort_index().plot.bar(ax=axes[0, 0], color=["#4C78A8", "#E45756"])
        axes[0, 0].set_title(f"Class balance (fraud rate = {df[TARGET].mean():.1%})")
        axes[0, 0].set_xticklabels(["legit (0)", "fraud (1)"], rotation=0)

        df.groupby("transaction_hour")[TARGET].mean().plot(ax=axes[0, 1], marker="o")
        axes[0, 1].set_title("Fraud rate by hour of day")
        axes[0, 1].set_ylabel("fraud rate")

        df.groupby("payment_method")[TARGET].mean().sort_values().plot.barh(ax=axes[1, 0])
        axes[1, 0].set_title("Fraud rate by payment method")

        df.boxplot(column="amount", by=TARGET, ax=axes[1, 1])
        axes[1, 1].set_title("Amount by class")
        axes[1, 1].set_xlabel(TARGET)
        fig.suptitle("EDA on fraud_transactions_clean (read from PostgreSQL)")
        fig.tight_layout()
        fig.savefig(out_dir / "eda_overview.png", dpi=120)
        plt.close(fig)

        # Leakage check: fraud_probability separates the classes perfectly
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, color in [(0, "#4C78A8"), (1, "#E45756")]:
            ax.hist(df.loc[df[TARGET] == label, "fraud_probability"], bins=30, alpha=0.7,
                    color=color, label=f"{TARGET}={label}")
        ax.set_title("fraud_probability by class: no overlap -> leaky column")
        ax.set_xlabel("fraud_probability")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "leakage_fraud_probability.png", dpi=120)
        plt.close(fig)

        rates = df.groupby("merchant_category")[TARGET].mean().round(4).to_dict()
        return {"plots_dir": str(out_dir), "fraud_rate": round(float(df[TARGET].mean()), 4),
                "fraud_rate_by_category": rates}

    # -----------------------------------------------------------------------
    # Analysis 2: train simple models (reads from Postgres)
    # -----------------------------------------------------------------------
    @task
    def train_models_from_db(qc: dict, **context) -> dict:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        df = read_sql(f'SELECT * FROM "{CLEAN_TABLE}"')

        def evaluate(name: str, model, include_leaky: bool) -> dict:
            X, y = model_frame(df, include_leaky=include_leaky)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=SEED
            )
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            return {
                "model": name,
                "uses_leaky_column": include_leaky,
                "f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
                "pr_auc": round(float(average_precision_score(y_te, proba)), 4),
                "n_features": int(X.shape[1]),
            }

        results = [
            evaluate("logistic_regression",
                     make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
                     include_leaky=False),
            evaluate("random_forest",
                     RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                            n_jobs=-1, random_state=SEED),
                     include_leaky=False),
            evaluate("logistic_regression_WITH_LEAK",
                     make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
                     include_leaky=True),
        ]
        out = run_dir(context) / "model_metrics.json"
        out.write_text(json.dumps(results, indent=2))
        return {"metrics_path": str(out), "results": results,
                "fraud_rate_baseline_pr_auc": round(float(df[TARGET].mean()), 4)}

    # -----------------------------------------------------------------------
    # Analysis 3: export the cleaned table (from Postgres) for the Part 2 agent loop
    # -----------------------------------------------------------------------
    @task
    def export_for_agents(qc: dict) -> dict:
        df = read_sql(f'SELECT * FROM "{CLEAN_TABLE}" ORDER BY transaction_id')
        AGENT_EXPORT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(AGENT_EXPORT, index=False)
        return {"path": str(AGENT_EXPORT), "n_rows": int(len(df)), "target_column": TARGET}

    # -----------------------------------------------------------------------
    # Final report: collect every task's small outputs into one JSON + markdown table
    # -----------------------------------------------------------------------
    @task
    def write_report(quality: dict, eda: dict, models: dict, export: dict, merge_meta: dict, **context) -> dict:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        for png in Path(eda["plots_dir"]).glob("*.png"):
            shutil.copy(png, REPORT_DIR / png.name)

        summary = {
            "run_ts": context["ts_nodash"],
            "rows_in_db": quality["rows_in_db"],
            "quality_checks": quality["checks"],
            "merge": merge_meta,
            "fraud_rate": eda["fraud_rate"],
            "models": models["results"],
            "agent_dataset": export["path"],
        }
        (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

        lines = ["| model | uses leaky column | F1 | ROC-AUC | PR-AUC |", "|---|---|---|---|---|"]
        for r in models["results"]:
            lines.append(f"| {r['model']} | {r['uses_leaky_column']} | {r['f1']} | {r['roc_auc']} | {r['pr_auc']} |")
        (REPORT_DIR / "model_results.md").write_text("\n".join(lines) + "\n")
        return {"report_dir": str(REPORT_DIR)}

    # -----------------------------------------------------------------------
    # WIRING: passing one task's return value into another draws the arrow
    # -----------------------------------------------------------------------
    merged = merge(transactions_clean, customers_clean, cards_agg)   # waits for all 3 groups
    loaded = load_to_postgres(merged)
    qc = quality_check(loaded)
    eda = eda_from_db(qc)                # these three only need qc,
    models = train_models_from_db(qc)    # so they run in parallel
    export = export_for_agents(qc)
    write_report(qc, eda, models, export, merged)


# Register the DAG with Airflow
fraud_etl_pipeline()
