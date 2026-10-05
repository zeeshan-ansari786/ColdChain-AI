# eda.py
# Quick look at the data before modelling: shapes, missing values,
# correlations and a few plots saved in results/.

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/coldchain.csv")
print("shape:", df.shape)
print("\nmissing values:\n", df.isnull().sum())
print("\ndescribe:\n", df.describe().round(2))

# 1. temp over first 7 days
plt.figure(figsize=(10, 3))
plt.plot(df["temp"][:168])
plt.axhline(8, color="r", ls="--", label="safe max 8C")
plt.axhline(2, color="r", ls="--", label="safe min 2C")
plt.title("Room temp - first week")
plt.xlabel("hour"); plt.ylabel("temp (C)"); plt.legend()
plt.tight_layout(); plt.savefig("results/temp_week.png"); plt.close()

# 2. excursion count
ax = df["excursion"].value_counts().plot(kind="bar")
plt.title("Normal vs excursion hours")
plt.xlabel("excursion (0=no, 1=yes)"); plt.ylabel("hours")
plt.tight_layout(); plt.savefig("results/excursion_bar.png"); plt.close()

# 3. correlation heatmap
plt.figure(figsize=(8, 6))
corr = df.corr(numeric_only=True)
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr)), corr.columns)
plt.title("Feature correlation")
plt.colorbar(); plt.tight_layout()
plt.savefig("results/corr.png"); plt.close()

print("\ncorr with temp_next:\n", corr["temp_next"].sort_values(ascending=False).round(3))
print("\ncorr with excursion:\n", corr["excursion"].sort_values(ascending=False).round(3))
print("\nplots saved in results/")
