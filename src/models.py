import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from src.config import AB_COLS, CLSI_BREAKPOINTS, MODELS_DIR, FIGURES_DIR

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def encode_res(x):
    if x == "R": return 1
    if x == "S": return 0
    return np.nan   # drop I


def train_evaluate(name, model, X_train, X_test, y_train, y_test, labels):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  {name}")
    print(f"    Test accuracy : {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=labels))
    return model, y_pred


def run_all_models(df_primary, df_secondary):

    # ── Model 1: Predict IPM resistance ──────────────────────
    print("\n── Model 1: Predict Imipenem resistance ──")
    TARGET = "IPM"
    FEATURES = [c for c in AB_COLS if c != TARGET]

    df_m1 = df_primary[FEATURES + [TARGET]].copy()

    # Encode properly (drop I)
    for col in FEATURES + [TARGET]:
        df_m1[col] = df_m1[col].map(encode_res)

    df_m1 = df_m1.dropna()

    # 🔥 Feature engineering
    df_m1["res_count"] = (df_m1[FEATURES] == 1).sum(axis=1)
    df_m1["res_ratio"] = df_m1["res_count"] / len(FEATURES)

    FEATURES_EXT = FEATURES + ["res_count", "res_ratio"]

    X1 = SimpleImputer(strategy="median").fit_transform(df_m1[FEATURES_EXT])
    y1 = df_m1[TARGET].astype(int)

    # Handle imbalance
    scale_pos_weight = (y1 == 0).sum() / (y1 == 1).sum()

    X1_tr, X1_te, y1_tr, y1_te = train_test_split(
        X1, y1, test_size=0.2, random_state=42, stratify=y1
    )

    xgb1, _ = train_evaluate(
        "XGBoost",
        XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight
        ),
        X1_tr, X1_te, y1_tr, y1_te,
        ["Susceptible", "Resistant"]
    )

    with open(MODELS_DIR / "xgb_ipm_resistance.pkl", "wb") as f:
        pickle.dump(xgb1, f)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        xgb1, X1_te, y1_te,
        display_labels=["Susceptible", "Resistant"],
        cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix — IPM Resistance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_confusion_matrix_ipm.png", dpi=150)
    plt.close()


    # ── Model 2: Predict Augmentin resistance ─────────────────
    print("\n── Model 2: Predict Augmentin resistance ──")

    AB2_FEATURES = [ab for ab in CLSI_BREAKPOINTS if ab != "AUGMENTIN"]

    df_m2 = df_secondary[[ab + "_class" for ab in AB2_FEATURES] + ["AUGMENTIN_class", "Location_enc"]].copy()
    df_m2 = df_m2[df_m2["AUGMENTIN_class"].isin(["R", "S"])].dropna()

    df_m2["target"] = (df_m2["AUGMENTIN_class"] == "R").astype(int)

    # Encode R/S
    for col in [ab + "_class" for ab in AB2_FEATURES]:
        df_m2[col] = df_m2[col].map({"R": 1, "S": 0})

    X2 = df_m2[[ab + "_class" for ab in AB2_FEATURES] + ["Location_enc"]]
    y2 = df_m2["target"]

    scale_pos_weight2 = (y2 == 0).sum() / (y2 == 1).sum()

    X2_tr, X2_te, y2_tr, y2_te = train_test_split(
        X2, y2, test_size=0.2, random_state=42, stratify=y2
    )

    xgb2, _ = train_evaluate(
        "XGBoost",
        XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight2
        ),
        X2_tr, X2_te, y2_tr, y2_te,
        ["Susceptible", "Resistant"]
    )

    with open(MODELS_DIR / "xgb_augmentin_resistance.pkl", "wb") as f:
        pickle.dump(xgb2, f)


    # ── Model 3: MDR from clinical features ───────────────────
    print("\n── Model 3: Predict MDR from clinical features ──")
    CLINICAL = ["Hypertension_enc", "Diabetes_enc", "Hospital_before_enc"]

    df_m3 = df_primary[CLINICAL + ["is_MDR"]].dropna()
    X3 = df_m3[CLINICAL]
    y3 = df_m3["is_MDR"]

    scale_pos_weight3 = (y3 == 0).sum() / (y3 == 1).sum()

    X3_tr, X3_te, y3_tr, y3_te = train_test_split(
        X3, y3, test_size=0.2, random_state=42, stratify=y3
    )

    xgb3, _ = train_evaluate(
        "XGBoost",
        XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight3
        ),
        X3_tr, X3_te, y3_tr, y3_te,
        ["Non-MDR", "MDR"]
    )

    with open(MODELS_DIR / "xgb_mdr_clinical.pkl", "wb") as f:
        pickle.dump(xgb3, f)

    print("\n  All models saved to outputs/models/")