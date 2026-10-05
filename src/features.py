# features.py
# Feature engineering: past readings help predict the next hour.
# We add lag features (what happened 1-2 hours ago) and a short
# rolling average of temp. This is the "feature engineering" step
# from our proposal.
#
# Fixes after professor review:
# - hour is cyclic (23 and 0 are neighbours), so we use sin/cos instead
#   of the raw hour number.
# - dropped 'day' (plain 0-6 number means nothing to a model).
# - dropped 'compressor_on' (it is exactly 1 - power_bad, so keeping
#   both only confuses the models).

import numpy as np
import pandas as pd

df = pd.read_csv("data/coldchain.csv")

# cyclic encoding of hour of day
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# what the room looked like 1 and 2 hours ago
for lag in [1, 2]:
    df[f"temp_lag{lag}"] = df["temp"].shift(lag)
    df[f"door_lag{lag}"] = df["door_open_min"].shift(lag)
    df[f"power_lag{lag}"] = df["power_bad"].shift(lag)

# average temp of last 3 hours (smooths sensor noise)
df["temp_roll3"] = df["temp"].rolling(3).mean()

df = df.drop(columns=["hour", "day", "compressor_on"])
df = df.dropna().reset_index(drop=True)
df.to_csv("data/coldchain_feat.csv", index=False)
print("shape after feature engineering:", df.shape)
print("columns:", list(df.columns))
