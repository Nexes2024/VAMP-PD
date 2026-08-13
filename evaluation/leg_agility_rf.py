"""
Leg-agility score classifier (Score in {0,1,2,3}).

Methodology
-----------
- Samples: 72 (36 subjects x left/right leg). Left & right kept as SEPARATE
  samples (not aggregated) but grouped by subject to prevent leakage.
- Cross-validation: subject-independent 5-fold GroupKFold, group = subject ID.
  All samples from the same subject land in the same fold.
- Feature selection: inside each TRAINING fold only, rank features by Random
  Forest impurity-based importance (Mean Decrease Impurity), then take top-k.
- Top-k grid: [5,10,15,20,30,40,50] capped at the number of features.
- Models: RandomForest (baseline) vs DummyClassifier. If the accuracy gap is
  < 10 points, additional classifiers are evaluated.
- Metrics: Accuracy, Macro-F1, MAE, and a pooled out-of-fold confusion matrix.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_error, confusion_matrix)

RNG = 42
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="Input feature spreadsheet")
parser.add_argument("--outdir", default="outputs/leg_agility")
args = parser.parse_args()

PATH = args.data
os.makedirs(args.outdir, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Load & clean
# ----------------------------------------------------------------------
df = pd.read_excel(PATH, sheet_name="legagility_left")

# Columns that are IDs / paths / acquisition artifacts -> not features.
DROP = ["video", "rel_path", "leg", "status",
        "duration_s", "fps", "total_frames", "detected_frames",
        "detection_rate", "rep_count", "reps_per_second"]

groups = df["subject"].values          # grouping key (kept out of X)
y = df["score"].astype(int).values
X = df.drop(columns=DROP + ["subject", "score"])
feature_names = list(X.columns)
X = X.values

print(f"Samples: {len(y)} | Subjects: {pd.Series(groups).nunique()} "
      f"| Features: {len(feature_names)}")
print("Feature columns:", feature_names)
print("Score distribution:", dict(pd.Series(y).value_counts().sort_index()))
print()

# Top-k grid, capped at #features and de-duplicated.
n_feat = len(feature_names)
K_GRID = sorted({min(k, n_feat) for k in [5, 10, 15, 20, 30, 40, 50]})
print("k grid (capped):", K_GRID, "\n")

gkf = GroupKFold(n_splits=5)


# ----------------------------------------------------------------------
# 2. Helper: run an estimator factory through GroupKFold with optional
#    in-fold top-k MDI feature selection. Returns pooled OOF predictions.
# ----------------------------------------------------------------------
def run_cv(make_estimator, k=None):
    """make_estimator() -> fresh sklearn estimator.
       k = number of top MDI features to keep (selection fit on train only).
       Returns (oof_pred, oof_true) aligned to original row order."""
    oof_pred = np.empty(len(y), dtype=int)
    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte = X[tr], X[te]
        ytr = y[tr]

        if k is not None and k < n_feat:
            # Rank features on TRAINING data only (no leakage).
            ranker = RandomForestClassifier(n_estimators=300,
                                            random_state=RNG)
            ranker.fit(Xtr, ytr)
            top = np.argsort(ranker.feature_importances_)[::-1][:k]
            Xtr, Xte = Xtr[:, top], Xte[:, top]

        est = make_estimator()
        est.fit(Xtr, ytr)
        oof_pred[te] = est.predict(Xte)
    return oof_pred, y


def score_block(pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "mae": mean_absolute_error(y, pred),
    }


# ----------------------------------------------------------------------
# 3. Random Forest across the top-k grid
# ----------------------------------------------------------------------
rf_factory = lambda: RandomForestClassifier(n_estimators=300, random_state=RNG)

print("=== RandomForest: top-k feature selection sweep ===")
rf_results = {}
for k in K_GRID:
    pred, _ = run_cv(rf_factory, k=k)
    m = score_block(pred)
    rf_results[k] = (m, pred)
    print(f"  k={k:>2} | acc={m['accuracy']:.3f} | "
          f"macroF1={m['macro_f1']:.3f} | MAE={m['mae']:.3f}")

# Best setting chosen by macro-F1 (handles imbalance better than accuracy).
best_k = max(rf_results, key=lambda k: rf_results[k][0]["macro_f1"])
best_metrics, best_pred = rf_results[best_k]
print(f"\nBest RF setting -> k={best_k}: {best_metrics}\n")


# ----------------------------------------------------------------------
# 4. Dummy baseline
# ----------------------------------------------------------------------
dummy_factory = lambda: DummyClassifier(strategy="most_frequent",
                                        random_state=RNG)
dummy_pred, _ = run_cv(dummy_factory, k=None)
dummy_metrics = score_block(dummy_pred)
print("=== DummyClassifier (most_frequent) ===")
print(f"  acc={dummy_metrics['accuracy']:.3f} | "
      f"macroF1={dummy_metrics['macro_f1']:.3f} | MAE={dummy_metrics['mae']:.3f}\n")

gap = best_metrics["accuracy"] - dummy_metrics["accuracy"]
print(f"Accuracy gap RF - Dummy = {gap*100:.1f} points\n")



# ----------------------------------------------------------------------
# 6. Confusion matrix for the best RF setting (pooled OOF)
# ----------------------------------------------------------------------
labels = [0, 1, 2, 3]
cm = confusion_matrix(y, best_pred, labels=labels)
print("=== Confusion matrix (best RF, pooled out-of-fold) ===")
print("rows = true, cols = predicted, labels 0..3")
print(cm)

# Per-class recall / precision for the analysis section.
print("\nPer-class breakdown:")
for i, c in enumerate(labels):
    support = cm[i].sum()
    recall = cm[i, i] / support if support else float("nan")
    col = cm[:, i].sum()
    precision = cm[i, i] / col if col else float("nan")
    print(f"  class {c}: support={support:>2} | "
          f"recall={recall:.2f} | precision={precision:.2f}")

# Save artifacts for downstream use.
np.save(os.path.join(args.outdir, "cm.npy"), cm)
pd.DataFrame({"true": y, "pred": best_pred,
              "subject": groups}).to_csv(os.path.join(args.outdir, "oof_predictions.csv"),
                                         index=False)
print("\nBest k =", best_k)
