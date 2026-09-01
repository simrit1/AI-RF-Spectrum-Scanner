"""Retrain the RF classifier from labeled feature CSV files.

Expected columns are FEATURE_NAMES plus a 'label' column. You can start from
logs/ and create a reviewed feature table, or use export_features.py on saved IQ.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from rf_core import FEATURE_NAMES

p = argparse.ArgumentParser()
p.add_argument("csv", nargs="+", help="One or more labeled feature CSV files")
p.add_argument("--out", default="models/local_house_rf_model.joblib")
p.add_argument("--label-column", default="label")
a = p.parse_args()

df = pd.concat([pd.read_csv(x) for x in a.csv], ignore_index=True)
missing = [x for x in FEATURE_NAMES + [a.label_column] if x not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")
X = df[FEATURE_NAMES].astype(float)
y = df[a.label_column].astype(str)
if y.nunique() < 2:
    raise SystemExit("Need at least two signal classes")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=570)
model = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=570)
model.fit(Xtr, ytr)
print(classification_report(yte, model.predict(Xte), zero_division=0))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"model":model, "features":FEATURE_NAMES, "source":"user-labeled-random-forest"}, a.out)
print("Saved", a.out)
