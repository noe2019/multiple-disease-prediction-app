"""Simulated clinical datasets (diabetes, heart disease, Parkinson's)."""
import numpy as np
import pandas as pd


def _label(rng, logit):
    return (rng.random(len(logit)) < 1 / (1 + np.exp(-logit))).astype(int)


def diabetes(n=2000, seed=1):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Pregnancies": r.poisson(3, n).clip(0, 15),
        "Glucose": r.normal(120, 30, n).clip(50, 220).round(),
        "BloodPressure": r.normal(72, 12, n).clip(40, 120).round(),
        "BMI": r.normal(31, 7, n).clip(16, 60).round(1),
        "Insulin": r.lognormal(4.5, 0.6, n).clip(15, 600).round(),
        "DiabetesPedigree": r.gamma(2, 0.22, n).clip(0.05, 2.4).round(3),
        "Age": r.gamma(4, 8, n).clip(21, 85).round(),
    })
    logit = (-9.5 + 0.04 * df.Glucose + 0.08 * df.BMI + 0.03 * df.Age + 0.9 * df.DiabetesPedigree
             + 0.08 * df.Pregnancies + 0.001 * df.Insulin)
    df["Outcome"] = _label(r, logit)
    return df


def heart(n=2000, seed=2):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Age": r.normal(55, 9, n).clip(29, 80).round(),
        "Sex_male": r.integers(0, 2, n),
        "ChestPainType": r.integers(0, 4, n),
        "RestingBP": r.normal(132, 17, n).clip(90, 200).round(),
        "Cholesterol": r.normal(245, 50, n).clip(120, 560).round(),
        "MaxHeartRate": r.normal(150, 22, n).clip(70, 205).round(),
        "ExerciseAngina": r.integers(0, 2, n),
        "STDepression": r.gamma(1.2, 0.9, n).clip(0, 6).round(1),
    })
    logit = (-4 + 0.05 * df.Age + 0.9 * df.Sex_male + 0.6 * df.ChestPainType + 0.012 * df.RestingBP
             + 0.004 * df.Cholesterol - 0.03 * df.MaxHeartRate + 1.2 * df.ExerciseAngina + 0.7 * df.STDepression)
    df["Outcome"] = _label(r, logit)
    return df


def parkinsons(n=1500, seed=3):
    r = np.random.default_rng(seed)
    y = r.random(n) < 0.6
    df = pd.DataFrame({
        "MDVP_Fo_Hz": np.where(y, r.normal(145, 30, n), r.normal(180, 45, n)).round(2),
        "Jitter_pct": np.where(y, r.gamma(3, 0.0025, n), r.gamma(3, 0.0013, n)).round(5),
        "Shimmer": np.where(y, r.gamma(4, 0.0085, n), r.gamma(4, 0.0045, n)).round(4),
        "HNR": np.where(y, r.normal(20.5, 4, n), r.normal(24.5, 4, n)).round(2),
        "RPDE": np.where(y, r.normal(0.52, 0.1, n), r.normal(0.44, 0.1, n)).round(3),
        "DFA": np.where(y, r.normal(0.72, 0.05, n), r.normal(0.69, 0.05, n)).round(3),
        "PPE": np.where(y, r.normal(0.23, 0.08, n), r.normal(0.12, 0.06, n)).clip(0.02).round(3),
    })
    df["Outcome"] = y.astype(int)
    return df


DATASETS = {"Type 2 diabetes": diabetes, "Heart disease": heart, "Parkinson's disease": parkinsons}
