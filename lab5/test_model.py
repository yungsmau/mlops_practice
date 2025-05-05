import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def evaluate(dataset_path):
    df = pd.read_csv(dataset_path)
    X = df[["x"]]
    y = df[["y"]]
    y_pred = model.predict(X)
    return mean_squared_error(y, y_pred), r2_score(y, y_pred)


def test_clean():
    mse, r2 = evaluate("data/clean.csv")
    assert mse < 1e-6, f"MSE слишком большая для clean.csv: {mse}"
    assert r2 > 0.9999, f"R2 слишком низний для clean.csv: {r2}"


def test_little_noise():
    mse, r2 = evaluate("data/little_noise.csv")
    assert (
        mse < 2.0
    ), f"MSE слишком большая для little_noise.csv с небольшим шумом: {mse}"
    assert r2 > 0.98, f"R2 слишком низкий для little_noise.csv с небольшим шумом: {r2}"


def test_noisy():
    mse, r2 = evaluate("data/noisy.csv")
    assert mse > 5.0, f"MSE неожиданно мала для noisy.csv: {mse}"
    assert r2 < 0.95, f"R2 слишком высок для noisy.csv: {r2}"
