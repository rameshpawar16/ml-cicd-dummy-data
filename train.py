# train_model.py
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

X = np.array([[0], [1], [2], [3]])
y = np.array([0, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")