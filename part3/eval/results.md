### Overall

| System | Cases passed | Mean score | Completed runs | Intermediate checks | Final checks | Coordination checks | Mean holdout F1 (signal cases) | Agent/step calls | Experiments | Retries | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Adaptive multi-agent (Part 3) | **9/9** | 1.00 | 9/9 | 100% | 100% | 100% | 0.664 | 185 | 36 | 4 | 540.0 |
| Class agent loop (baseline 2) | **3/9** | 0.53 | 8/9 | 22% | 71% | 50% | 0.302 | 250 | 50 | 2 | 340.8 |
| Fixed Airflow workflow (baseline 1) | **2/9** | 0.41 | 7/9 | 11% | 59% | 50% | 0.183 | 65 | 7 | 4 | 62.5 |

### Per case (pass/fail · weighted score · holdout F1)

| Case | Adaptive | Class loop | Fixed Airflow |
|---|---|---|---|
| C1_leak_known — Leaky column (known name) | ✅ 1.00 · F1 0.06 | ✅ 1.00 · F1 0.05 | ✅ 1.00 · F1 0.07 |
| C2_leak_renamed — Leaky column (renamed upstream) | ✅ 1.00 · F1 0.06 | ❌ 0.60 · F1 0.05 | ❌ 0.20 · F1 1.00 |
| C3_balanced_titanic — Balanced, clean data (Titanic) | ✅ 1.00 · F1 0.73 | ✅ 1.00 · F1 0.76 | ✅ 1.00 · F1 0.70 |
| C4_signal_imbalanced — Imbalanced data with real signal | ✅ 1.00 · F1 0.67 | ❌ 0.40 · F1 0.24 | ❌ 0.20 · F1 0.24 |
| C5_dirty_numeric — Junk strings in a numeric column | ✅ 1.00 · F1 0.63 | ❌ 0.25 · F1 0.26 | ❌ 0.25 · F1 0.22 |
| C6_duplicates — 8% duplicated rows | ✅ 1.00 · F1 0.67 | ❌ 0.25 · F1 0.27 | ❌ 0.25 · F1 0.28 |
| C7_invalid_labels — 1.5% invalid target labels | ✅ 1.00 · F1 0.72 | ❌ 0.00 | ❌ 0.00 |
| C8_flaky_worker — Training worker fails 3 times | ✅ 1.00 · F1 0.67 | ✅ 0.75 · F1 0.43 | ❌ 0.25 |
| C9_no_signal_budget — No signal: do not waste the budget | ✅ 1.00 · F1 0.06 | ❌ 0.50 · F1 0.05 | ❌ 0.50 · F1 0.07 |

### Criteria detail

**C1_leak_known — Leaky column (known name)** (pass threshold 1.0) — *Detect fraud_probability as leakage, never train on it, report an honest (low) holdout PR-AUC.*

| Criterion | Group | Weight | Adaptive | Class loop | Fixed Airflow |
|---|---|---|---|---|---|
| `completed` | final | 1 | ✅ completed | ✅ completed | ✅ completed |
| `column_dropped {'column': 'fraud_probability'}` | intermediate | 2 | ✅ dropped | ✅ dropped | ✅ dropped |
| `no_leak_in_final {'max_pr_auc': 0.9}` | final | 2 | ✅ PR-AUC 0.037 | ✅ PR-AUC 0.037 | ✅ PR-AUC 0.040 |

**C2_leak_renamed — Leaky column (renamed upstream)** (pass threshold 1.0) — *Same leak under a new name (vendor_risk_score): detection must come from evidence, not a hard-coded list.*

| Criterion | Group | Weight | Adaptive | Class loop | Fixed Airflow |
|---|---|---|---|---|---|
| `completed` | final | 1 | ✅ completed | ✅ completed | ✅ completed |
| `column_dropped {'column': 'vendor_risk_score'}` | intermediate | 2 | ✅ dropped | ❌ kept | ❌ kept |
| `no_leak_in_final {'max_pr_auc': 0.9}` | final | 2 | ✅ PR-AUC 0.037 | ✅ PR-AUC 0.036 | ❌ PR-AUC 1.000 |

**C3_balanced_titanic — Balanced, clean data (Titanic)** (pass threshold 0.75) — *No imbalance -> skip the threshold tuner; reach a solid holdout F1.*

| Criterion | Group | Weight | Adaptive | Class loop | Fixed Airflow |
|---|---|---|---|---|---|
| `completed` | final | 1 | ✅ completed | ✅ completed | ✅ completed |
| `tuner_used {'expected': False}` | coordination | 1 | ✅ tuner_used=False | ✅ tuner_used=False | ✅ tuner_used=False |
| `min_holdout_f1 {'value': 0.7}` | final | 2 | ✅ F1 0.729 | ✅ F1 0.761 | ✅ F1 0.700 |

**C4_signal_imbalanced — Imbalanced data with real signal** (pass threshold 0.8) — *Detect imbalance, use the tuner, find a model family that beats logistic regression.*

| Criterion | Group | Weight | Adaptive | Class loop | Fixed Airflow |
|---|---|---|---|---|---|
| `completed` | final | 1 | ✅ completed | ✅ completed | ✅ completed |
| `tuner_used {'expected': True}` | coordination | 1 | ✅ tuner_used=True | ❌ tuner_used=False | ❌ tuner_used=False |
| `column_dropped {'column': 'account_id'}` | intermediate | 1 | ✅ dropped | ✅ dropped | ❌ kept |
| `min_holdout_f1 {'value': 0.55}` | final | 2 | ✅ F1 0.674 | ❌ F1 0.241 | ❌ F1 0.239 |

**C5_dirty_numeric — Junk strings in a numeric column** (pass threshold 0.75) — *Validator flags numeric contamination in f3, repair coerces it, run completes with good F1.*

| Criterion | Group | Weight | Adaptive | Class loop | Fixed Airflow |
|---|---|---|---|---|---|
| `issue_detected {'type': 'numeric_contamination'}` | intermediate | 1 | ✅ detected | ❌ not detected | ❌ not detected |
| `repair_applied {'action': 'coerce_numeric'}` | intermediate | 1 | ✅ applied | ❌ not applied | ❌ not applied |
| `completed` | final | 1 | ✅ completed | ✅ completed | ✅ completed |
| `min_holdout_f1 {'value': 0.55}` | final | 1 | ✅ F1 0.632 | ❌ F1 0.264 | ❌ F1 0.217 |

**C6_duplicates — 8% duplicated rows** (pass threshold 0.75) — *Detect and drop duplicate rows before splitting (otherwise copies leak into the holdout).*

| Criterion | Group | Weight | Adaptive | Class loop | Fixed Airflow |
|---|---|---|---|---|---|
| `issue_detected {'type': 'duplicate_rows'}` | intermediate | 1 | ✅ detected | ❌ not detected | ❌ not detected |
| `repair_applied {'action': 'drop_duplicates'}` | intermediate | 1 | ✅ applied | ❌ not applied | ❌ not applied |
| `completed` | final | 1 | ✅ completed | ✅ completed | ✅ completed |
| `min_holdout_f1 {'value': 0.55}` | final | 1 | ✅ F1 0.674 | ❌ F1 0.274 | ❌ F1 0.275 |

**C7_invalid_labels — 1.5% invalid target labels** (pass threshold 0.75) — *Quarantine rows whose label is not 0/1 instead of crashing.*

| Criterion | Group | Weight | Adaptive | Class loop | Fixed Airflow |
|---|---|---|---|---|---|
| `issue_detected {'type': 'invalid_labels'}` | intermediate | 1 | ✅ detected | ❌ not detected | ❌ not detected |
| `repair_applied {'action': 'quarantine_invalid_labels'}` | intermediate | 1 | ✅ applied | ❌ not applied | ❌ not applied |
| `completed` | final | 2 | ✅ completed | ❌ failed: ValueError: Input contains NaN | ❌ failed: ValueError: Input contains NaN |

**C8_flaky_worker — Training worker fails 3 times** (pass threshold 0.75) — *Retry failed fits, route around the failure and still finish with a good model.*

| Criterion | Group | Weight | Adaptive | Class loop | Fixed Airflow |
|---|---|---|---|---|---|
| `completed` | final | 2 | ✅ completed | ✅ completed | ❌ failed: RuntimeError: transient worker failure (injected) |
| `retried {'min': 1}` | coordination | 1 | ✅ retries=2 | ✅ retries=2 | ✅ retries=2 |
| `min_holdout_f1 {'value': 0.55}` | final | 1 | ✅ F1 0.674 | ❌ F1 0.426 | ❌ F1 - |

**C9_no_signal_budget — No signal: do not waste the budget** (pass threshold 1.0) — *Nothing beats ~3% PR-AUC; detect the plateau and stop early instead of spending all 8 experiments.*

| Criterion | Group | Weight | Adaptive | Class loop | Fixed Airflow |
|---|---|---|---|---|---|
| `completed` | final | 1 | ✅ completed | ✅ completed | ✅ completed |
| `early_stop` | coordination | 2 | ✅ 3/8 exps · stop=plateau (early stop) | ❌ 8/8 exps · stop=finalize | ❌ 1/8 exps · stop=None |
| `no_leak_in_final {'max_pr_auc': 0.9}` | final | 1 | ✅ PR-AUC 0.037 | ✅ PR-AUC 0.037 | ✅ PR-AUC 0.040 |

