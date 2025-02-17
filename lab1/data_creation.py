import numpy as np
import pandas as pd

import os

from sklearn.model_selection import train_test_split


np.random.seed(42)
n_samples = 1000
time = np.arange(n_samples)

temperature = 20 + 0.1 * time + np.random.normal(0, 2, size=n_samples)

temperature[::100] += np.random.normal(10, 5, size=10)

X = time.reshape(-1, 1)
y = temperature

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

train_data = pd.DataFrame({"time": X_train.flatten(), "temperature": y_train})
test_data = pd.DataFrame({"time": X_test.flatten(), "temperature": y_test})

train_data.to_csv("train/train_data.csv", index=False)
test_data.to_csv("test/test_data.csv", index=False)
