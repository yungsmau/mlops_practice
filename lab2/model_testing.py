import pickle
import pandas as pd

from sklearn.metrics import mean_squared_error


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

test_data = pd.read_csv("test/test_data_scaled.csv")

X_test = test_data[["time"]]
y_test = test_data[["temperature"]]

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Model test MSE is: {mse}")
