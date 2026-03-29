import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# Load Model
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "xgb_ipm_multiclass.pkl"

print("Loading model from:", MODEL_PATH)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!")

LABELS = ["Susceptible", "Intermediate", "Resistant"]

# =========================
# Prediction Function
# =========================
def predict_resistance(input_features):
    """
    Predict antibiotic resistance with confidence
    """

    # Convert to numpy
    input_features = np.array(input_features)

    # Feature Engineering (MUST match training)
    res_count = (input_features == 1).sum()
    res_ratio = res_count / len(input_features)

    # Add engineered features
    full_features = np.append(input_features, [res_count, res_ratio])

    # Convert to DataFrame (safer for XGBoost)
    sample_df = pd.DataFrame([full_features])

    # Prediction
    probs = model.predict_proba(sample_df)[0]

    idx = np.argmax(probs)

    return {
        "prediction": LABELS[idx],
        "confidence": float(probs[idx])
    }

# =========================
# Main (for testing)
# =========================
if __name__ == "__main__":

    print("\nModel expects features:", model.n_features_in_)

    #  Adjust this length if needed
    base_features = [0,1,0,1,0,1,0,0,1]

    # Auto-fix size if mismatch
    expected_base_len = model.n_features_in_ - 2

    if len(base_features) != expected_base_len:
        print(f"Fixing input size to {expected_base_len}")
        base_features = [0] * expected_base_len

    result = predict_resistance(base_features)

    print("\n Prediction Result:")
    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']*100:.2f}%")