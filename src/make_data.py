# make_data.py
# Makes a fake cold-storage sensor dataset (hourly readings, 120 days).
# Columns are the same ones in our proposal: temp, humidity, door open
# time, power status, load etc. We use it to train and test our models.

import numpy as np
import pandas as pd

np.random.seed(42)  # same data every time we run

HOURS = 120 * 24  # 120 days of hourly readings
t = np.arange(HOURS)

# --- base signals ---
hour_of_day = t % 24
# room temp moves a little with day/night cycle (base 4 deg)
base_temp = 4.0 + 0.8 * np.sin(2 * np.pi * hour_of_day / 24)
# outside (ambient) temp: hot afternoons, cool nights
ambient = 28 + 7 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)

# door open minutes in that hour: mostly small, sometimes long
door_open = np.random.poisson(4, HOURS).astype(float)
long_open = np.random.rand(HOURS) < 0.12
door_open[long_open] = np.random.uniform(10, 40, long_open.sum())

# power problem in that hour (1 = fluctuation / cut).
# Power cuts last a few hours: if bad now, 55% chance it stays bad next hour.
power_bad = np.zeros(HOURS, dtype=int)
power_bad[0] = 1 if np.random.rand() < 0.08 else 0
for i in range(1, HOURS):
    if power_bad[i - 1] == 1:
        power_bad[i] = 1 if np.random.rand() < 0.55 else 0
    else:
        power_bad[i] = 1 if np.random.rand() < 0.045 else 0

# storage load % (how full the room is)
load = np.clip(np.random.normal(65, 12, HOURS), 30, 98)

# compressor on/off (off during power problem)
compressor = 1 - power_bad

# rare thermostat fault: cooling stuck ON -> room overcools for a few hours.
# This gives us some below-2C excursions too (not only heat).
fault = np.zeros(HOURS, dtype=int)
fault[0] = 1 if np.random.rand() < 0.02 else 0
for i in range(1, HOURS):
    if fault[i - 1] == 1:
        fault[i] = 1 if np.random.rand() < 0.5 else 0
    else:
        fault[i] = 1 if np.random.rand() < 0.012 else 0

# --- build inside temp from the signals ---
# A cold room has thermal inertia: temp changes slowly, so today's temp
# depends on yesterday's temp too (AR process). door open lets warm air
# in, power cut stops cooling.
temp = np.zeros(HOURS)
temp[0] = 4.0
for i in range(1, HOURS):
    target = (base_temp[i]
              + 0.28 * door_open[i]       # warm air in
              + 5.0 * power_bad[i]        # no cooling
              - 6.0 * fault[i]            # thermostat fault: overcooling
              + 0.03 * (ambient[i] - 28)  # hot day leaks in a bit
              - 0.01 * (load[i] - 65))    # full room stays cooler
    temp[i] = 0.60 * temp[i - 1] + 0.40 * target + np.random.normal(0, 0.3)

# humidity rises when door is open
humidity = np.clip(70 + 0.5 * door_open + np.random.normal(0, 3, HOURS),
                   55, 95)

df = pd.DataFrame({
    "hour": hour_of_day,
    "day": (t // 24) % 7,          # 0 = monday ... 6 = sunday
    "temp": np.round(temp, 2),
    "humidity": np.round(humidity, 1),
    "door_open_min": np.round(door_open, 1),
    "power_bad": power_bad,
    "load_pct": np.round(load, 1),
    "compressor_on": compressor,
    "ambient_temp": np.round(ambient, 1),
})

# regression target: temp in the NEXT hour
df["temp_next"] = df["temp"].shift(-1)
# classification target: 1 if temp goes out of safe range 2-8 deg
df["excursion"] = ((df["temp_next"] < 2.0) | (df["temp_next"] > 8.0)).astype(int)
df = df.dropna().reset_index(drop=True)

df.to_csv("data/coldchain.csv", index=False)
print("rows:", len(df))
print("excursion rate: %.2f %%" % (100 * df["excursion"].mean()))
print(df.head())
