| cycle | model | hyperparameters | val F1 @0.5 | val F1 tuned | threshold |
|---|---|---|---|---|---|
| 1 | logistic_regression | `{'C': 0.25, 'max_iter': 300}` | 0.046 | 0.060 | 0.6 |
| 2 | logistic_regression | `{'C': 0.5, 'max_iter': 300}` | 0.051 | 0.062 | 0.6 |
| 3 | random_forest | `{'n_estimators': 160, 'max_depth': 8}` | 0.000 | 0.059 | 0.1 |
| 4 | logistic_regression | `{'C': 2.0, 'max_iter': 300}` | 0.050 | 0.063 | 0.6 |
| 5 | logistic_regression | `{'C': 4.0, 'max_iter': 300}` | 0.053 | 0.059 | 0.3 |

**Sealed holdout (best spec):** F1 = 0.057, ROC-AUC = 0.508, PR-AUC = 0.037 (positive rate = 0.030), accuracy = 0.951, threshold = 0.6
