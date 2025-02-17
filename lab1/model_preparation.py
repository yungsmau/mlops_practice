import pickle
import pandas as pd

from sklearn.linear_model import LinearRegression

train_data = pd.read_csv("train/train_data_scaled.csv")

X_train = train_data[["time"]]
y_train = train_data[["temperature"]]

model = LinearRegression()
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
