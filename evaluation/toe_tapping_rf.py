"""
Toe-tapping severity classification (Score in {0,1,2,3})
========================================================
Subject-independent Random Forest baseline with:
  - GroupKFold(5) grouped by subject  -> no subject leakage across folds
  - class_weight="balanced"           -> handles severe imbalance
  - MDI feature selection within each training fold      -> ranking done INSIDE each train fold
  - top-k sweep k in [5,10,15,20,30,40,50] capped by #features
  - comparison vs a most-frequent DummyClassifier
Metrics: Accuracy, Macro-F1, MAE, and the pooled out-of-fold confusion matrix.
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
parser.add_argument("--outdir", default="outputs/toe_tapping")
args = parser.parse_args()

PATH = args.data
os.makedirs(args.outdir, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Load & column selection
# ----------------------------------------------------------------------
df = pd.read_excel(PATH)

# Identifiers / metadata (the "first few" non-useful cols) + recording/processing
# artifacts. 'hand', 'cycle_count', 'cycles_per_second' were requested but are
# not present in this toe-tapping export, so they are silently ignored.
DROP_COLS = [
    "video", "rel_path", "status", "foot",          # metadata / constant / side id
    "duration_s", "fps", "total_frames",            # processing artifacts
    "detected_frames", "detection_rate",
    "hand", "cycle_count", "cycles_per_second",     # not present (ignored)
]

GROUP_COL  = "subject"
TARGET_COL = "score"

feature_cols = [c for c in df.columns
                if c not in DROP_COLS + [GROUP_COL, TARGET_COL]]

X      = df[feature_cols].copy()
y      = df[TARGET_COL].astype(int).values
groups = df[GROUP_COL].values

print(f"Samples           : {len(df)}")
print(f"Subjects (groups) : {df[GROUP_COL].nunique()}")
print(f"Features used     : {len(feature_cols)}")
print(f"  {feature_cols}")
print(f"Class counts      : {dict(pd.Series(y).value_counts().sort_index())}")
print(f"Majority class    : {pd.Series(y).mode()[0]} "
      f"({pd.Series(y).value_counts().max()}/{len(y)} = "
      f"{pd.Series(y).value_counts().max()/len(y):.1%})")

n_features = len(feature_cols)
K_GRID = sorted({min(k, n_features) for k in [5, 10, 15, 20, 30, 40, 50]})
print(f"Effective top-k grid (capped at {n_features}): {K_GRID}\n")

N_SPLITS = 5
gkf = GroupKFold(n_splits=N_SPLITS)


def make_rf():
    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RNG,
        n_jobs=-1,
    )


# ----------------------------------------------------------------------
# 2. CV with nested feature selection -> pooled out-of-fold predictions
# ----------------------------------------------------------------------
def run_cv_rf(k):
    """RF with top-k MDI feature selection done inside each training fold.
       Returns pooled OOF predictions + per-fold (acc, macroF1) and
       a counter of how often each feature was selected (stability)."""
    oof_pred = np.full(len(y), -1, dtype=int)
    fold_acc, fold_f1 = [], []
    sel_count = {f: 0 for f in feature_cols}

    for tr, te in gkf.split(X, y, groups):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        # rank features on TRAIN ONLY (MDI / mean decrease in impurity)
        ranker = make_rf().fit(X_tr, y_tr)
        order = np.argsort(ranker.feature_importances_)[::-1]
        top = [feature_cols[i] for i in order[:k]]
        for f in top:
            sel_count[f] += 1

        # final model on the selected features only
        clf = make_rf().fit(X_tr[top], y_tr)
        p = clf.predict(X_te[top])
        oof_pred[te] = p
        fold_acc.append(accuracy_score(y_te, p))
        fold_f1.append(f1_score(y_te, p, average="macro", zero_division=0))

    return oof_pred, np.array(fold_acc), np.array(fold_f1), sel_count


def run_cv_dummy():
    oof_pred = np.full(len(y), -1, dtype=int)
    for tr, te in gkf.split(X, y, groups):
        d = DummyClassifier(strategy="most_frequent").fit(X.iloc[tr], y[tr])
        oof_pred[te] = d.predict(X.iloc[te])
    return oof_pred


def summarize(name, oof):
    return {
        "model": name,
        "accuracy": accuracy_score(y, oof),
        "macro_f1": f1_score(y, oof, average="macro", zero_division=0),
        "mae": mean_absolute_error(y, oof),
    }


# ----------------------------------------------------------------------
# 3. Sweep k, pick best by macro-F1 (tie-break accuracy)
# ----------------------------------------------------------------------
results, oof_store, stab_store = [], {}, {}
for k in K_GRID:
    oof, facc, ff1, sel = run_cv_rf(k)
    r = summarize(f"RF (top-{k})", oof)
    r["fold_acc_mean"], r["fold_acc_std"] = facc.mean(), facc.std()
    r["fold_f1_mean"],  r["fold_f1_std"]  = ff1.mean(),  ff1.std()
    r["k"] = k
    results.append(r)
    oof_store[k] = oof
    stab_store[k] = sel

res_df = pd.DataFrame(results).sort_values(
    ["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)

dummy_oof = run_cv_dummy()
dummy_sum = summarize("Dummy (most_frequent)", dummy_oof)

best_k = int(res_df.iloc[0]["k"])
best_oof = oof_store[best_k]

# ----------------------------------------------------------------------
# 4. Report
# ----------------------------------------------------------------------
pd.set_option("display.width", 160, "display.max_columns", 20)
print("=" * 78)
print("CROSS-VALIDATED RESULTS  (pooled out-of-fold, 5-fold GroupKFold by subject)")
print("=" * 78)
show = res_df[["model", "accuracy", "macro_f1", "mae",
               "fold_acc_mean", "fold_acc_std", "fold_f1_mean", "fold_f1_std"]]
print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print()
print(f"DUMMY  acc={dummy_sum['accuracy']:.3f}  "
      f"macroF1={dummy_sum['macro_f1']:.3f}  mae={dummy_sum['mae']:.3f}")
print()
print(f">>> BEST SETTING: RF (top-{best_k})  "
      f"acc={res_df.iloc[0]['accuracy']:.3f}  "
      f"macroF1={res_df.iloc[0]['macro_f1']:.3f}  mae={res_df.iloc[0]['mae']:.3f}")

# overall feature importance (full-feature RF, averaged across folds) ---------
imp_acc = np.zeros(n_features)
for tr, _ in gkf.split(X, y, groups):
    imp_acc += make_rf().fit(X.iloc[tr], y[tr]).feature_importances_
imp_mean = imp_acc / N_SPLITS
imp_rank = pd.Series(imp_mean, index=feature_cols).sort_values(ascending=False)
print("\n--- Overall MDI feature importance (mean across folds) ---")
print(imp_rank.to_string(float_format=lambda v: f"{v:.4f}"))

print(f"\n--- Feature selection stability at best k={best_k} "
      f"(times chosen out of {N_SPLITS} folds) ---")
stab = pd.Series(stab_store[best_k]).sort_values(ascending=False)
print(stab[stab > 0].to_string())

# ----------------------------------------------------------------------
# 5. Confusion matrices
# ----------------------------------------------------------------------
labels = [0, 1, 2, 3]
cm_rf = confusion_matrix(y, best_oof, labels=labels)
cm_du = confusion_matrix(y, dummy_oof, labels=labels)
print(f"\n--- Confusion matrix: RF (top-{best_k})  [rows=true, cols=pred] ---")
print(pd.DataFrame(cm_rf, index=[f"t{l}" for l in labels],
                   columns=[f"p{l}" for l in labels]).to_string())
print("\nPer-class recall (RF):")
for i, l in enumerate(labels):
    tot = cm_rf[i].sum()
    print(f"  class {l}: {cm_rf[i,i]}/{tot} = "
          f"{(cm_rf[i,i]/tot if tot else 0):.2f}")

# save artifacts for plotting / sharing
np.save(os.path.join(args.outdir, "cm_rf.npy"), cm_rf)
np.save(os.path.join(args.outdir, "cm_dummy.npy"), cm_du)
res_df.to_csv(os.path.join(args.outdir, "cv_results.csv"), index=False)
imp_rank.to_csv(os.path.join(args.outdir, "feature_importance.csv"))
print("\nDONE.")
