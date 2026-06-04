import pandas as pd
import numpy as np
import glob
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

files = glob.glob("dataset/*.csv")

print("Found files:", files)

dfs = []

for file in files:
    try:
        df = pd.read_csv(file, low_memory=False)
        dfs.append(df)
    except:
        print("Skipping:", file)

data = pd.concat(dfs, ignore_index=True)

# Keep numeric columns only
data = data.select_dtypes(include=np.number)

# Clean data
data = data.fillna(data.mean())

# Use only first 10 columns
X = data.iloc[:, :10]

# Create stable target (normalized)
y = (X.mean(axis=1)) / 100

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Training completed")