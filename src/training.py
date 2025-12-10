import os, json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,   
    balanced_accuracy_score,           
    cohen_kappa_score,                 
)

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None
    keras = None

MODEL_PATH   = "models/ann.keras"
HISTORY_PATH = "models/ann_history.json"

def _build_model(input_dim: int, num_classes: int = 3):
    if keras is None:
        raise RuntimeError("TensorFlow/Keras not available. Install requirements and retry.")
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    # run_eagerly=False is default; keep it fast/stable
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def _encode_labels(y_str):
    map_ = {"Low": 0, "Medium": 1, "High": 2}
    y = np.array([map_.get(v, 0) for v in y_str], dtype=np.int32)
    return y, map_

def _load_history_if_any():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _validate_arrays(X_train, y_train, X_val, y_val):
    # shapes
    if X_train is None or X_val is None or y_train is None or y_val is None:
        raise ValueError("Training arrays are None.")
    if X_train.size == 0 or X_val.size == 0:
        raise ValueError("No rows after preprocessing. Please check the uploaded dataset.")
    if X_train.shape[1] == 0:
        raise ValueError("No input features available after encoding.")
    # NaN/inf
    if np.isnan(X_train).any() or np.isnan(X_val).any():
        raise ValueError("Found NaN values in features. Please check preprocessing.")
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Found non-finite values in features.")
    # at least 2 classes in training
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training set has fewer than 2 classes. Provide more varied data.")

def load_or_train_model(X_train, y_train_str, X_val, y_val_str, encoders, scaler, label_info, retrain=False):
    """
    Return (history_obj_or_None, metrics_dict, artifacts_dict).
    metrics_dict['history'] is a dict if available (loaded/saved).
    """
    if keras is None:
        raise RuntimeError("TensorFlow/Keras not available. Install requirements and retry.")

    # Cast to float32 for safety/stability across datasets
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val   = np.asarray(X_val,   dtype=np.float32)

    y_train, _ = _encode_labels(y_train_str)
    y_val, _   = _encode_labels(y_val_str)

    # Validate before training
    _validate_arrays(X_train, y_train, X_val, y_val)

    history_obj  = None
    history_dict = None

    if (not retrain) and os.path.exists(MODEL_PATH):
        # Load model + any saved history
        model = keras.models.load_model(MODEL_PATH)
        history_dict = _load_history_if_any()
    else:
        model = _build_model(X_train.shape[1], 3)

        # Try EarlyStopping; if TF/Keras misbehaves, fall back to no-callback training.
        try:
            callbacks = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
            history_obj = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=10, batch_size=256, verbose=0, callbacks=callbacks
            )
        except Exception:
            # Fallback path (fixes the NoneType name-scope crash for some datasets)
            history_obj = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=10, batch_size=256, verbose=0
            )

        model.save(MODEL_PATH)

        # Persist history for future sessions
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        history_dict = history_obj.history if hasattr(history_obj, "history") else None
        try:
            if history_dict is not None:
                with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                    json.dump(history_dict, f)
        except Exception:
            pass

    # -------------- Evaluate --------------
    y_prob = model.predict(X_val, verbose=0)
    y_pred = y_prob.argmax(axis=1)

    # Core metrics
    acc          = float(accuracy_score(y_val, y_pred))
    f1_macro     = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
    f1_weighted  = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
    f1_micro     = float(f1_score(y_val, y_pred, average="micro", zero_division=0))  # == accuracy
    bal_acc      = float(balanced_accuracy_score(y_val, y_pred))
    kappa        = float(cohen_kappa_score(y_val, y_pred))
    cm           = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])

    # Per-class diagnostics
    prec, rec, f1_per_class, support = precision_recall_fscore_support(
        y_val, y_pred, labels=[0, 1, 2], zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,        # handy to compare with macro
        "micro_f1": f1_micro,              # will equal accuracy
        "balanced_accuracy": bal_acc,      # insensitive to class imbalance
        "kappa": kappa,                    # agreement beyond chance
        "confusion": cm,
        "labels": ["Low", "Medium", "High"],
        "history": history_dict,
        # diagnostics (useful in Logs tab if you want)
        "per_class": {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "f1": f1_per_class.tolist(),
            "support": support.tolist(),
        },
    }
    artifacts = {"model": MODEL_PATH, "encoders": encoders, "scaler": scaler, "label_info": label_info}
    return history_obj, metrics, artifacts