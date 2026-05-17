# Converted from 01_construct_input_window_and_5d_labels.ipynb
# This script is provided as a readable backup in case GitHub cannot preview the notebook.


# ==============================================================================
# #### Step 1: Load the data
# ==============================================================================


# %% [code] Cell 2
import pandas as pd
import os

folder_path = "."
file_path = os.path.join(folder_path, "CSI300_merged_2005_2026.csv")

df = pd.read_csv(file_path)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Keep only the columns used later in the workflow.
df = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()

print(df.head())
print(df.tail())
print(df.shape)


# ==============================================================================
# #### Step 2: Create the future 5-day return and the binary label
#
# Definitions used in this notebook:
#
# - `future_5d_ret`: return over the next 5 trading days
# - `label_5d`: 1 if the future 5-day return is positive, otherwise 0
# ==============================================================================


# ==============================================================================
# Step 2 reminder:
#
# - `future_5d_ret`: return over the next 5 trading days
# - `label_5d`: 1 if the future 5-day return is positive, otherwise 0
# ==============================================================================


# %% [code] Cell 5
# Future 5-day return.
df["future_5d_ret"] = df["close"].shift(-5) / df["close"] - 1

# Binary label: 1 if the future 5-day return is positive, otherwise 0.
df["label_5d"] = (df["future_5d_ret"] > 0).astype(int)

print(df[["date", "close", "future_5d_ret", "label_5d"]].head(10))
print(df[["date", "close", "future_5d_ret", "label_5d"]].tail(10))


# ==============================================================================
# #### Step 3: Reserve the valid sample range for a 20-day lookback window
#
# At this stage we are not drawing images yet.  
# We only keep rows that can serve as valid samples.
#
# Because the model uses the past 20 trading days to predict the next 5 trading days:
# - the first 19 rows cannot form a full lookback window,
# - the last 5 rows cannot form a full forward label.
# ==============================================================================


# %% [code] Cell 7
WINDOW = 20
HORIZON = 5

# A row is usable only if it has both a full past window and a full future label.
usable_df = df.iloc[WINDOW - 1 : len(df) - HORIZON].copy().reset_index(drop=True)

print("Number of usable samples:", len(usable_df))
print(usable_df.head())
print(usable_df.tail())


# ==============================================================================
# #### Step 4: Build the rolling-window dataset
#
# This step extracts the past 20-day OHLC window for every sample.
#
# Once the data are stored as arrays, the same dataset can later be used for:
# - direct input into traditional machine learning models,
# - image rendering,
# - CNN / MLP models.
# ==============================================================================


# %% [code] Cell 9
import numpy as np

WINDOW = 20
feature_cols = ["open", "high", "low", "close"]

X = []
y = []
sample_dates = []

for i in range(WINDOW - 1, len(df) - HORIZON):
    window_data = df.loc[i - WINDOW + 1 : i, feature_cols].values   # Past 20-day OHLC window
    label = df.loc[i, "label_5d"]                                   # Future 5-day label aligned with date i
    sample_date = df.loc[i, "date"]

    X.append(window_data)
    y.append(label)
    sample_dates.append(sample_date)

X = np.array(X)   # shape: (n_samples, 20, 4)
y = np.array(y)   # shape: (n_samples,)
sample_dates = np.array(sample_dates)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("First 3 sample dates:", sample_dates[:3])
print("First 3 labels:", y[:3])


# %% [code] Cell 10
save_path = os.path.join(folder_path, "CSI300_window20_label5d.npz")

np.savez(
    save_path,
    X=X,
    y=y,
    dates=sample_dates.astype(str)
)

print("Saved to:", save_path)
