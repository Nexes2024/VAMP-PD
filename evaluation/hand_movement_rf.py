"""
hand_movement_rf_pipeline.py
=============================
Subject-independent Random Forest classifier for hand-movement Score (0-3),
built from automated hand-movement video features.

Pipeline summary
-----------------
1. Load data, drop identifier/path columns and acquisition-artifact columns.
2. Keep BOTH left and right hand samples per subject (no aggregation), but
   guarantee leakage-free evaluation by grouping CV splits on `subject`, so a
   subject's left+right samples never get split across train/test.
3. 5-fold GroupKFold (group = subject). See report for why 10-fold is not
   recommended on this dataset (sparse rare-class folds).
4. Inside every training fold: impute missing values (median, fold-fit only),
   rank features by Random Forest Mean Decrease Impurity (MDI), then refit RF
   on the top-k features for k in {5,10,15,20,30,40,50}, each capped at the
   number of available features (29).
5. Compare best RandomForest setting to a DummyClassifier baseline. Because
   the gap exceeds 10 percentage points, the spec's escalation step (trying
   other classifier families) is not strictly required, but a small bonus
   comparison (Logistic Regression, SVM-RBF, KNN, Gradient Boosting) is run
   anyway for robustness, using the same CV scheme.
6. Report accuracy, macro-F1, MAE, and an out-of-fold confusion matrix.

Run:
    python hand_movement_rf_pipeline.py --data /path/to/handmovement-final.xlsx --outdir results/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, mean_absolute_error)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
TARGET_COL = "Score"
GROUP_COL = "subject"

# Acquisition/bookkeeping columns that carry no biomechanical signal and
# must never be used as model features (explicitly requested by the user).
ARTIFACT_COLS = [
    "video", "rel_path", "hand", "status", "duration_s", "fps",
    "total_frames", "detected_frames", "detection_rate",
    "cycle_count", "cycles_per_second",
]

K_GRID_REQUESTED = [5, 10, 15, 20, 30, 40, 50]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data(path):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]  # fixes trailing-space "Score "
    drop_cols = ARTIFACT_COLS + [GROUP_COL, TARGET_COL]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    groups = df[GROUP_COL].copy()
    return df, X, y, groups, feature_cols


def capped_k_list(n_features, k_grid=K_GRID_REQUESTED):
    return sorted(set(min(k, n_features) for k in k_grid))


# --------------------------------------------------------------------------- #
# Core RF + feature-ranking + top-k evaluation, all inside GroupKFold
# --------------------------------------------------------------------------- #
def rank_features_mdi(X_train_imp, y_train, feature_names):
    """Fit an RF on the full feature set and rank by impurity-based importance."""
    rf = RandomForestClassifier(
        n_estimators=1000, random_state=RANDOM_STATE,
        class_weight="balanced", n_jobs=-1,
    )
    rf.fit(X_train_imp, y_train)
    order = np.argsort(rf.feature_importances_)[::-1]
    return [(feature_names[i], rf.feature_importances_[i]) for i in order]


def fit_eval_rf(X_train, y_train, X_test):
    rf = RandomForestClassifier(
        n_estimators=1000, random_state=RANDOM_STATE,
        class_weight="balanced", n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf.predict(X_test)


def run_rf_groupkfold(X, y, groups, feature_cols, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    ks = capped_k_list(len(feature_cols))

    per_k = {k: {"acc": [], "f1m": [], "mae": []} for k in ks}
    test_preds_by_k = {k: np.full(len(y), -1, dtype=int) for k in ks}
    fold_rankings, dummy_preds = [], np.full(len(y), -1, dtype=int)
    dummy_acc, dummy_f1, dummy_mae = [], [], []

    for fold_idx, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=feature_cols, index=X_tr.index)
        X_te_imp = pd.DataFrame(imputer.transform(X_te), columns=feature_cols, index=X_te.index)

        ranked = rank_features_mdi(X_tr_imp, y_tr, feature_cols)
        fold_rankings.append(ranked)
        ranked_names = [f for f, _ in ranked]

        for k in ks:
            feats = ranked_names[:k]
            preds = fit_eval_rf(X_tr_imp[feats], y_tr, X_te_imp[feats])
            test_preds_by_k[k][te] = preds
            per_k[k]["acc"].append(accuracy_score(y_te, preds))
            per_k[k]["f1m"].append(f1_score(y_te, preds, average="macro"))
            per_k[k]["mae"].append(mean_absolute_error(y_te, preds))

        dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
        dpreds = dummy.predict(X_te)
        dummy_preds[te] = dpreds
        dummy_acc.append(accuracy_score(y_te, dpreds))
        dummy_f1.append(f1_score(y_te, dpreds, average="macro"))
        dummy_mae.append(mean_absolute_error(y_te, dpreds))

    summary = {k: {
        "acc_mean": float(np.mean(v["acc"])), "acc_std": float(np.std(v["acc"])),
        "f1m_mean": float(np.mean(v["f1m"])), "f1m_std": float(np.std(v["f1m"])),
        "mae_mean": float(np.mean(v["mae"])), "mae_std": float(np.std(v["mae"])),
    } for k, v in per_k.items()}

    dummy_summary = {
        "acc_mean": float(np.mean(dummy_acc)), "acc_std": float(np.std(dummy_acc)),
        "f1m_mean": float(np.mean(dummy_f1)), "f1m_std": float(np.std(dummy_f1)),
        "mae_mean": float(np.mean(dummy_mae)), "mae_std": float(np.std(dummy_mae)),
    }
    return dict(ks=ks, summary=summary, dummy_summary=dummy_summary,
                test_preds_by_k=test_preds_by_k, dummy_preds=dummy_preds,
                fold_rankings=fold_rankings)


def run_other_classifier(clf_builder, X, y, groups, n_splits=5, scale=False):
    gkf = GroupKFold(n_splits=n_splits)
    accs, f1ms, maes = [], [], []
    for tr, te in gkf.split(X, y, groups=groups):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        imp = SimpleImputer(strategy="median")
        X_tr_i, X_te_i = imp.fit_transform(X_tr), imp.transform(X_te)
        if scale:
            sc = StandardScaler()
            X_tr_i, X_te_i = sc.fit_transform(X_tr_i), sc.transform(X_te_i)
        clf = clf_builder().fit(X_tr_i, y_tr)
        preds = clf.predict(X_te_i)
        accs.append(accuracy_score(y_te, preds))
        f1ms.append(f1_score(y_te, preds, average="macro"))
        maes.append(mean_absolute_error(y_te, preds))
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1m_mean": float(np.mean(f1ms)), "f1m_std": float(np.std(f1ms)),
            "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes))}


def tenfold_feasibility_check(y, groups, n_splits=10):
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold_idx, (tr, te) in enumerate(gkf.split(y, y, groups=groups)):
        cc_test = y.iloc[te].value_counts().reindex([0, 1, 2, 3], fill_value=0)
        cc_train = y.iloc[tr].value_counts().reindex([0, 1, 2, 3], fill_value=0)
        rows.append({
            "fold": fold_idx, "n_test_subjects": groups.iloc[te].nunique(),
            "n_test_samples": len(te),
            "test_class0": cc_test[0], "test_class1": cc_test[1],
            "test_class2": cc_test[2], "test_class3": cc_test[3],
            "train_class3_n": cc_train[3],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(data_path, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, X, y, groups, feature_cols = load_data(data_path)
    print(f"N samples={len(df)} | N features={len(feature_cols)} | N subjects={groups.nunique()}")

    results = run_rf_groupkfold(X, y, groups, feature_cols, n_splits=5)
    rows = []
    for k in results["ks"]:
        s = results["summary"][k]
        rows.append({"k": k, **s})
    perk_df = pd.DataFrame(rows)
    perk_df.to_csv(outdir / "rf_per_k_cv_results.csv", index=False)
    print("\nRF per-k CV results (5-fold GroupKFold):\n", perk_df.to_string(index=False))

    ds = results["dummy_summary"]
    print(f"\nDummy (most_frequent): acc={ds['acc_mean']:.3f} macroF1={ds['f1m_mean']:.3f} MAE={ds['mae_mean']:.3f}")

    best_k = max(results["ks"], key=lambda k: results["summary"][k]["f1m_mean"])
    best = results["summary"][best_k]
    print(f"\nBest k (by mean macro-F1): {best_k}")
    print(f"RF(best k) vs Dummy -> acc diff={best['acc_mean']-ds['acc_mean']:.3f}, "
          f"macroF1 diff={best['f1m_mean']-ds['f1m_mean']:.3f}")

    preds_best = results["test_preds_by_k"][best_k]
    cm = confusion_matrix(y, preds_best, labels=[0, 1, 2, 3])
    pd.DataFrame(cm, index=[f"true_{i}" for i in range(4)],
                 columns=[f"pred_{i}" for i in range(4)]).to_csv(outdir / "confusion_matrix_best_k.csv")
    print("\nOut-of-fold confusion matrix (best k):\n", cm)
    print("\n", classification_report(y, preds_best, labels=[0, 1, 2, 3], digits=3, zero_division=0))

    # Feature stability across folds
    from collections import Counter
    top5_counter = Counter()
    rank_sum = {f: 0 for f in feature_cols}
    for ranked in results["fold_rankings"]:
        names = [f for f, _ in ranked]
        for f in names[:5]:
            top5_counter[f] += 1
        for pos, (f, _imp) in enumerate(ranked):
            rank_sum[f] += pos
    stability_df = pd.DataFrame({
        "feature": feature_cols,
        "avg_rank": [rank_sum[f] / 5 for f in feature_cols],
        "top5_selection_count_of_5_folds": [top5_counter.get(f, 0) for f in feature_cols],
    }).sort_values("avg_rank")
    stability_df.to_csv(outdir / "feature_stability.csv", index=False)
    print("\nMost stable / important features:\n", stability_df.head(10).to_string(index=False))

    # 10-fold feasibility check
    tf = tenfold_feasibility_check(y, groups, n_splits=10)
    tf.to_csv(outdir / "tenfold_feasibility_check.csv", index=False)
    print("\n10-fold GroupKFold feasibility check:\n", tf.to_string(index=False))

    # Bonus classifier comparison using the most stable top-5 feature set
    stable_top5 = stability_df["feature"].head(5).tolist()
    Xk = X[stable_top5]
    bonus = {
        "RandomForest(best_k)": best,
        "Dummy": ds,
        "LogisticRegression": run_other_classifier(
            lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            Xk, y, groups, scale=True),
        "SVM-RBF": run_other_classifier(
            lambda: SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
            Xk, y, groups, scale=True),
        "KNN(k=5)": run_other_classifier(
            lambda: KNeighborsClassifier(n_neighbors=5), Xk, y, groups, scale=True),
        "GradientBoosting": run_other_classifier(
            lambda: GradientBoostingClassifier(random_state=RANDOM_STATE), Xk, y, groups, scale=False),
    }
    bonus_df = pd.DataFrame(bonus).T
    bonus_df.to_csv(outdir / "classifier_comparison.csv")
    print("\nClassifier comparison (top-5 stable features):\n", bonus_df.to_string())

    with open(outdir / "summary.json", "w") as f:
        json.dump({
            "n_samples": len(df), "n_features": len(feature_cols), "n_subjects": int(groups.nunique()),
            "best_k": best_k, "best_k_metrics": best, "dummy_metrics": ds,
            "stable_top5_features": stable_top5,
        }, f, indent=2)

    print(f"\nAll result artifacts written to: {outdir.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    main(args.data, args.outdir)
