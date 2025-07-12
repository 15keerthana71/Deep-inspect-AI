import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import (LSTM, Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

DATA_DIR = Path("data")  # change if you keep files elsewhere
PM_CSV = DATA_DIR / "predictive_maintenance.csv"
DEFECT_ZIP = DATA_DIR / "neu_defects.zip"
DEFECT_EXTRACT_DIR = DATA_DIR / "neu_defects"
IMG_SIZE = (128, 128)


def train_predictive_maintenance(csv_path: Path):
    """Train LSTM on predictive‑maintenance tabular data."""
    print("\n🔧 Training Predictive‑Maintenance Model …")
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["Failure"])
    y = df["Failure"].astype(int)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_reshaped, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(1, X_scaled.shape[1])),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_tr, y_tr, epochs=30, batch_size=32, validation_data=(X_te, y_te), verbose=0)

    y_hat = (model.predict(X_te) > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_te, y_hat),
        "precision": precision_score(y_te, y_hat, zero_division=0),
        "recall": recall_score(y_te, y_hat, zero_division=0),
        "f1": f1_score(y_te, y_hat, zero_division=0),
        "confusion_matrix": confusion_matrix(y_te, y_hat).tolist(),
    }
    print("✅ Predictive‑Maintenance Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return model, metrics


def extract_defect_zip(zip_path: Path, dest_dir: Path):
    if dest_dir.exists():
        print("✅ Dataset already extracted.")
        return dest_dir
    print("📂 Extracting defects dataset …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print("✅ Extraction complete.")
    return dest_dir


def load_defect_images(dataset_root: Path):
    """Load images under train/ sub‑folder. Assumes structure train/<class>/*.bmp"""
    X, y = [], []
    train_dir = dataset_root / "train"
    categories = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])

    for label in categories:
        class_dir = train_dir / label
        for img_file in class_dir.iterdir():
            img = load_img(img_file, target_size=IMG_SIZE)
            img = img_to_array(img) / 255.0
            X.append(img)
            y.append(label)

    X = np.array(X, dtype="float32")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc, num_classes=len(categories))

    return X, y_cat, categories


def train_surface_defect_cnn(zip_path: Path, extract_dir: Path):
    """Train simple CNN to classify NEU metal surface defects."""
    dataset_root = extract_defect_zip(zip_path, extract_dir)
    X, y, classes = load_defect_images(dataset_root)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(len(classes), activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_tr, y_tr, epochs=15, batch_size=32, validation_data=(X_te, y_te), verbose=0)

    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"✅ Surface Defect CNN accuracy: {acc*100:.2f}% on test set ({len(X_te)} samples)")
    return model, classes


if __name__ == "__main__":
    # 1️⃣ Train predictive‑maintenance model
    if PM_CSV.exists():
        _pm_model, _pm_metrics = train_predictive_maintenance(PM_CSV)
    else:
        print(f"❌ Predictive‑Maintenance CSV not found at {PM_CSV}. Download it first.")

    # 2️⃣ Train surface defect classifier
    if DEFECT_ZIP.exists():
        _defect_model, _classes = train_surface_defect_cnn(DEFECT_ZIP, DEFECT_EXTRACT_DIR)
    else:
        print(f"❌ Defects zip not found at {DEFECT_ZIP}. Download it first.")
 give simple read me for this 
