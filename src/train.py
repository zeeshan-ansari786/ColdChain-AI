# train.py
# Trains every model from our syllabus on both tasks:
#   Task 1 (regression): predict temp_next (temp of next hour)
#   Task 2 (classification): predict excursion (1 = temp will go out of 2-8C)
#
# Model selection rule (decided before training):
#   - regression winner = lowest RMSE on the test set
#   - classification winner = highest F1 on the test set
#     (accuracy misleads because only ~14% of hours are excursions)
#
# Because this is time data, we do NOT shuffle: first 80% of hours are
# train, last 20% are test. Cross-validation uses TimeSeriesSplit so the
# model never trains on the future and tests on the past.

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score)

df = pd.read_csv("data/coldchain_feat.csv")
X = df.drop(columns=["temp_next", "excursion"])
print("features:", list(X.columns))

tscv = TimeSeriesSplit(n_splits=5)
rows = []

# chronological split: no shuffling for time data
def chrono_split(X, y, frac=0.8):
    n = int(len(X) * frac)
    return (X.iloc[:n], X.iloc[n:], y.iloc[:n], y.iloc[n:])

# random oversampling of the minority class (for models like KNN and
# NaiveBayes that have no class_weight option, so the comparison stays fair)
def oversample(Xtr, ytr):
    rng = np.random.RandomState(42)
    pos = np.where(ytr == 1)[0]
    need = (ytr == 0).sum() - len(pos)
    extra = rng.choice(pos, size=need, replace=True)
    idx = np.concatenate([np.arange(len(ytr)), extra])
    rng.shuffle(idx)
    return Xtr.iloc[idx].reset_index(drop=True), ytr.iloc[idx].reset_index(drop=True)

# ---------------- Task 1: REGRESSION (predict temp_next) ----------------
print("\n===== TASK 1: regression -> temp_next =====")
y_reg = df["temp_next"]
Xtr, Xte, ytr, yte = chrono_split(X, y_reg)

# naive baseline: next hour = this hour (true persistence).
# Xte["temp"] is the current hour's temp, temp_next is next hour's.
base_pred = Xte["temp"]
base_rmse = mean_squared_error(yte, base_pred) ** 0.5
print(f"Baseline (persistence)  RMSE={base_rmse:.4f}")
rows.append({"task": "regression", "model": "Baseline-persistence",
             "best_params": "-", "rmse": round(base_rmse, 4),
             "mae": round(mean_absolute_error(yte, base_pred), 4),
             "r2": round(r2_score(yte, base_pred), 4),
             "accuracy": "", "precision": "", "recall": "", "f1": "",
             "time_s": 0})

reg_models = {
    # LinearRegression has no hyperparameters, so we fit it directly
    # instead of a fake GridSearchCV with an empty grid.
    # NOTE (Oct 4, 2026): every model below is taken strictly from the
    # official Data Science-I syllabus (25MCA3001, verified against the
    # DDU GNUMS portal syllabus PDF). Nothing outside the syllabus is used.
    "LinearRegression": ("direct", LinearRegression()),
    "KNNRegressor": ("grid", Pipeline([("s", StandardScaler()), ("m", KNeighborsRegressor())]),
                     {"m__n_neighbors": [3, 5, 9, 15, 25, 35]}),
    "DecisionTreeReg": ("grid", DecisionTreeRegressor(random_state=42),
                        {"max_depth": [3, 5, 8, 12, None],
                         "min_samples_split": [2, 5, 10]}),
    "RandomForestReg": ("random", RandomForestRegressor(random_state=42),
                        {"n_estimators": [50, 100, 200],
                         "max_depth": [5, 10, 15, None],
                         "min_samples_split": [2, 5]}),
    "SVR": ("grid", Pipeline([("s", StandardScaler()), ("m", SVR())]),
            {"m__C": [0.1, 1, 10], "m__gamma": ["scale", 0.01, 0.1]}),
}

for name, spec in reg_models.items():
    t0 = time.time()
    if spec[0] == "direct":
        model = spec[1]
        model.fit(Xtr, ytr)
        params = {}
    elif spec[0] == "grid":
        search = GridSearchCV(spec[1], spec[2], cv=tscv, n_jobs=-1)
        search.fit(Xtr, ytr)
        model, params = search, search.best_params_
    else:
        search = RandomizedSearchCV(spec[1], spec[2], cv=tscv, n_iter=10,
                                    random_state=42, n_jobs=-1)
        search.fit(Xtr, ytr)
        model, params = search, search.best_params_
    pred = model.predict(Xte)
    rmse = mean_squared_error(yte, pred) ** 0.5
    rows.append({"task": "regression", "model": name,
                 "best_params": str(params),
                 "rmse": round(rmse, 4),
                 "mae": round(mean_absolute_error(yte, pred), 4),
                 "r2": round(r2_score(yte, pred), 4),
                 "accuracy": "", "precision": "", "recall": "", "f1": "",
                 "time_s": round(time.time() - t0, 1)})
    print(f"{name:18s} RMSE={rmse:.4f} R2={r2_score(yte, pred):.4f}  {params}")

# ---------------- Task 2: CLASSIFICATION (predict excursion) ----------------
print("\n===== TASK 2: classification -> excursion =====")
y_clf = df["excursion"]
Xtr, Xte, ytr, yte = chrono_split(X, y_clf)

# naive baseline: always predict "no excursion" (majority class)
base_pred = np.zeros(len(yte), dtype=int)
print(f"Baseline (majority)   acc={accuracy_score(yte, base_pred):.4f} "
      f"f1={f1_score(yte, base_pred):.4f}")
rows.append({"task": "classification", "model": "Baseline-majority",
             "best_params": "-", "rmse": "", "mae": "", "r2": "",
             "accuracy": round(accuracy_score(yte, base_pred), 4),
             "precision": 0.0, "recall": 0.0, "f1": 0.0, "time_s": 0})

# Tie-break rule: if two models tie on the selection metric, the simpler
# model wins (fewer params, faster, easier to explain in viva).
#
# Fair-imbalance note: KNN and NaiveBayes have no class_weight option.
# For them we oversample the minority class — but ONLY inside each CV
# fold (never before the split), so validation scores stay honest.

# manual grid search with in-fold oversampling (used for KNN only)
def tune_knn(Xtr, ytr):
    from itertools import product
    best_score, best_k = -1, None
    for k in [3, 5, 9, 15, 25, 35]:
        scores = []
        for tri, vai in tscv.split(Xtr):
            Xo, yo = oversample(Xtr.iloc[tri], ytr.iloc[tri])
            m = Pipeline([("s", StandardScaler()),
                          ("m", KNeighborsClassifier(n_neighbors=k))])
            m.fit(Xo, yo)
            scores.append(f1_score(ytr.iloc[vai], m.predict(Xtr.iloc[vai]),
                                   zero_division=0))
        s = float(np.mean(scores))
        if s > best_score:
            best_score, best_k = s, k
    return best_k

clf_models = {
    # (search_type, model, grid) — "manual" = own CV loop, "direct" = no tuning
    "LogisticRegression": ("grid", Pipeline([("s", StandardScaler()),
                                             ("m", LogisticRegression(max_iter=2000,
                                                                      class_weight="balanced"))]),
                           {"m__C": [0.1, 1.0, 10.0]}),
    "KNNClassifier": ("manual", None, None),
    "NaiveBayes": ("direct", GaussianNB(), {}),
    "DecisionTreeClf": ("grid", DecisionTreeClassifier(random_state=42,
                                                       class_weight="balanced"),
                        {"max_depth": [3, 5, 8, 12, None],
                         "min_samples_split": [2, 5, 10]}),
    "RandomForestClf": ("random", RandomForestClassifier(random_state=42,
                                                         class_weight="balanced"),
                        {"n_estimators": [50, 100, 200],
                         "max_depth": [5, 10, 15, None],
                         "min_samples_split": [2, 5]}),
    "SVC": ("grid", Pipeline([("s", StandardScaler()),
                              ("m", SVC(class_weight="balanced"))]),
            {"m__C": [0.1, 1, 10], "m__gamma": ["scale", 0.01, 0.1]}),
}

for name, (how, model, grid) in clf_models.items():
    t0 = time.time()
    if how == "manual":  # KNN: tune k with honest in-fold oversampling
        best_k = tune_knn(Xtr, ytr)
        params = {"m__n_neighbors": best_k}
        Xo, yo = oversample(Xtr, ytr)
        model = Pipeline([("s", StandardScaler()),
                          ("m", KNeighborsClassifier(n_neighbors=best_k))])
        model.fit(Xo, yo)
    elif how == "direct":  # NaiveBayes: nothing to tune, fit on oversampled train
        Xo, yo = oversample(Xtr, ytr)
        model.fit(Xo, yo)
        params = {}
    elif how == "grid":
        search = GridSearchCV(model, grid, cv=tscv, n_jobs=-1, scoring="f1")
        search.fit(Xtr, ytr)
        model, params = search, search.best_params_
    else:
        search = RandomizedSearchCV(model, grid, cv=tscv, n_iter=10,
                                    random_state=42, n_jobs=-1, scoring="f1")
        search.fit(Xtr, ytr)
        model, params = search, search.best_params_
    pred = model.predict(Xte)
    rows.append({"task": "classification", "model": name,
                 "best_params": str(params),
                 "rmse": "", "mae": "", "r2": "",
                 "accuracy": round(accuracy_score(yte, pred), 4),
                 "precision": round(precision_score(yte, pred, zero_division=0), 4),
                 "recall": round(recall_score(yte, pred, zero_division=0), 4),
                 "f1": round(f1_score(yte, pred, zero_division=0), 4),
                 "time_s": round(time.time() - t0, 1)})
    print(f"{name:18s} acc={accuracy_score(yte, pred):.4f} "
          f"f1={f1_score(yte, pred, zero_division=0):.4f}  {params}")

pd.DataFrame(rows).to_csv("results/metrics.csv", index=False)
print("\nsaved results/metrics.csv")
