# plots.py
# Makes the result plots the professor asked for:
# confusion matrix (best classifier), residual plot (best regressor),
# feature importance (random forest classifier).
# Best models and params are read from results/metrics.csv.

import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data/coldchain_feat.csv")
X = df.drop(columns=["temp_next", "excursion"])
m = pd.read_csv("results/metrics.csv")

n = int(len(X) * 0.8)
Xtr, Xte = X.iloc[:n], X.iloc[n:]

# ---- best classifier -> confusion matrix + feature importance ----
clf = m[(m["task"] == "classification") & (~m["model"].str.startswith("Baseline"))]
best = clf.loc[clf["f1"].astype(float).idxmax()]
print("best classifier:", best["model"], "f1 =", best["f1"])
params = ast.literal_eval(best["best_params"])
ytr, yte = df["excursion"].iloc[:n], df["excursion"].iloc[n:]

if "RandomForest" in best["model"]:
    model = RandomForestClassifier(random_state=42, class_weight="balanced",
                                   **params)
elif "DecisionTree" in best["model"]:
    model = DecisionTreeClassifier(random_state=42, class_weight="balanced",
                                   **params)
elif "SVC" in best["model"]:
    model = Pipeline([("s", StandardScaler()),
                      ("m", SVC(class_weight="balanced",
                                C=params["m__C"], gamma=params["m__gamma"]))])
else:
    raise SystemExit("add builder for " + best["model"])
model.fit(Xtr, ytr)
pred = model.predict(Xte)

cm = confusion_matrix(yte, pred)
plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion matrix - {best['model']}")
plt.xlabel("predicted"); plt.ylabel("actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout(); plt.savefig("results/confusion_matrix.png"); plt.close()

if hasattr(model, "feature_importances_"):
    imp = pd.Series(model.feature_importances_, index=X.columns)
    imp.sort_values().plot(kind="barh", figsize=(7, 5))
    plt.title(f"Feature importance - {best['model']}")
    plt.tight_layout(); plt.savefig("results/feat_importance.png"); plt.close()
    print(imp.sort_values(ascending=False).round(3).head(6))

# ---- best regressor -> residual plot ----
reg = m[(m["task"] == "regression") & (~m["model"].str.startswith("Baseline"))]
best = reg.loc[reg["rmse"].astype(float).idxmin()]
print("best regressor:", best["model"], "rmse =", best["rmse"])
params = ast.literal_eval(best["best_params"])
ytr, yte = df["temp_next"].iloc[:n], df["temp_next"].iloc[n:]

if best["model"] == "LinearRegression":
    rmodel = LinearRegression()
else:
    raise SystemExit("plots.py: add builder for regressor " + best["model"])
rmodel.fit(Xtr, ytr)
pred = rmodel.predict(Xte)

plt.figure()
plt.scatter(pred, yte - pred, s=8, alpha=0.5)
plt.axhline(0, color="r", ls="--")
plt.title(f"Residual plot - {best['model']}")
plt.xlabel("predicted temp"); plt.ylabel("residual (actual - predicted)")
plt.tight_layout(); plt.savefig("results/residuals.png"); plt.close()
print("plots saved")
