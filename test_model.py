import joblib
import numpy as np
import sys

# Load model
model = joblib.load("model.pkl")

# Dummy input
dummy_input = np.array([[1.5]])

# Predict
prediction = model.predict(dummy_input)

# Check output type
if isinstance(prediction[0], (int, float, np.integer, np.floating)):
    print("✅ Prediction test passed")
    sys.exit(0)
else:
    print("❌ Prediction output type invalid")
    sys.exit(1)