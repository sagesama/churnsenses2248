"""Train, compare and save the churn model. Run: python train.py"""
import json
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from common import *

df = load_data()
X, y = df[FEATURES], (df["Churn"] == "Yes").astype(int)
X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
    X, y, df["customerID"], test_size=0.2, stratify=y, random_state=42)

pre = ColumnTransformer([("num", StandardScaler(), NUM),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                            class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
results, fitted = {}, {}
for name, clf in candidates.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    results[name] = {"roc_auc": roc_auc_score(y_te, p), "f1": f1_score(y_te, p > .5)}
    fitted[name] = pipe
    print(name, results[name])

best = max(results, key=lambda k: results[k]["roc_auc"])
model = fitted[best]
proba = model.predict_proba(X_te)[:, 1]
pred = (proba >= 0.5).astype(int)
fpr, tpr, _ = roc_curve(y_te, proba)
imp = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=42)
importance = sorted(zip(FEATURES, imp.importances_mean), key=lambda t: -t[1])

metrics = {
    "best_model": best, "comparison": results,
    "accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred),
    "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred),
    "roc_auc": roc_auc_score(y_te, proba),
    "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
    "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    "importance": [{"feature": f, "importance": float(v)} for f, v in importance],
    "n_train": len(X_tr), "n_test": len(X_te),
}
joblib.dump(model, MODEL)
METRICS.write_text(json.dumps(metrics, indent=2))
pd.DataFrame({"customerID": id_te.values, "actual": y_te.values,
              "predicted": pred, "churn_probability": proba}).to_csv(PREDS, index=False)
print(f"Saved best model: {best}  AUC={metrics['roc_auc']:.3f}")
