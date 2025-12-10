# src/prediction.py
import joblib
import numpy as np
from tensorflow import keras

MODEL_PATH = "models/ann.keras"
OHE_PATH = "encoders/ohe.joblib"
SCALER_PATH = "encoders/scaler.joblib"

def load_inference():
    model = keras.models.load_model(MODEL_PATH)
    ohe = joblib.load(OHE_PATH)
    scaler = joblib.load(SCALER_PATH)
    return {"model": model, "ohe": ohe, "scaler": scaler}

def predict_single(infer, dow: str, hour: int, nbhd: str, ctype: str):
    ohe = infer["ohe"]; scaler = infer["scaler"]; model = infer["model"]
    # shape (1,1) for scaler, and one-hot for 3 categoricals
    X_hour = scaler.transform([[hour]])
    X_cat = ohe.transform([[dow, nbhd, ctype]])
    X = np.hstack([X_hour, X_cat])
    probs = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = ["Low", "Medium", "High"][idx]
    return label, probs.tolist()