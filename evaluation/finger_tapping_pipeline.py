"""
Finger-tapping severity score (0-3) classification pipeline.

Methodology
-----------
- Each row = one hand-side measurement (left or right) for one subject.
  Rows are NOT aggregated (the task is to predict the per-side score),
  but GroupKFold (group = subject) guarantees both sides of a subject
  always fall in the same fold -> no subject-level leakage.
- Feature set excludes identifier / acquisition-artifact columns.
- For every outer fold:
    1. An inner GroupKFold loop (on the outer-training subjects only)
       is used to pick the best top-k feature count for that fold,
       using Random-Forest impurity importance computed on inner-train
       only (no peeking at the outer test fold).
    2. The winning k for that outer fold is then used to fit the final
       model (RF impurity importance computed on the *full* outer-train
       set) and evaluate once on the untouched outer-test fold.
  This nested design gives an (almost) unbiased generalization estimate
  even though we are also choosing a hyperparameter (k).
- Out-of-fold predictions from all 5 outer folds are pooled (every one
  of the 73 rows is predicted exactly once, by a model that never saw
  it during training or feature selection) to compute final metrics
  and the confusion matrix.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, confusion_matrix
)

warnings.filterwarnings("ignore")

RNG = 42
LABELS = [0, 1, 2, 3]

ARTIFACT_COLS = [
    "video", "subject", "rel_path", "hand", "duration_s", "fps",
    "total_frames", "detected_frames", "detection_rate",
    "tap_count", "taps_per_second",
]
TARGET_COL = "score"
GROUP_COL = "subject"
K_LIST_RAW = [5, 10, 15, 20, 30, 40, 50]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data(path):
    df = pd.read_excel(path)
    feature_cols = [c for c in df.columns if c not in ARTIFACT_COLS + [TARGET_COL]]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    groups = df[GROUP_COL].copy()
    return df, X, y, groups, feature_cols


def capped_k_list(n_features):
    ks = sorted(set(min(k, n_features) for k in K_LIST_RAW))
    return ks


# --------------------------------------------------------------------------- #
# Core building blocks
# --------------------------------------------------------------------------- #
def rf_importance_ranking(X_train, y_train):
    """Fit an RF on the full feature set and return features sorted by MDI importance (desc)."""
    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced_subsample", random_state=RNG, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    order = np.argsort(rf.feature_importances_)[::-1]
    return [X_train.columns[i] for i in order], rf.feature_importances_[order]


def build_classifier(name):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=RNG, n_jobs=-1,
        ), False
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=RNG,
        ), True
    if name == "svm_rbf":
        return SVC(kernel="rbf", class_weight="balanced", random_state=RNG), True
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=RNG), False
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5), True
    raise ValueError(name)


def fit_predict(clf_name, X_train, y_train, X_test, needs_scaling):
    imputer = SimpleImputer(strategy="median")
    Xtr = imputer.fit_transform(X_train)
    Xte = imputer.transform(X_test)
    if needs_scaling:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
    clf, _ = build_classifier(clf_name)
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    return pred, clf


def metrics_block(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "mae": mean_absolute_error(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Nested CV with top-k feature selection
# --------------------------------------------------------------------------- #
def nested_cv_topk(clf_name, X, y, groups, k_list, n_outer=5, n_inner=4,
                    select_metric="macro_f1", verbose=True):
    needs_scaling = build_classifier(clf_name)[1]
    outer_cv = GroupKFold(n_splits=n_outer, shuffle=True, random_state=RNG)

    oof_pred = pd.Series(index=X.index, dtype=int)
    fold_log = []

    for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y, groups)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        g_tr = groups.iloc[tr_idx]

        # ---- inner loop: pick best k using ONLY the outer-train data ----
        inner_cv = GroupKFold(n_splits=n_inner, shuffle=True, random_state=RNG)
        inner_scores = {k: [] for k in k_list}

        for in_tr_idx, in_te_idx in inner_cv.split(X_tr, y_tr, g_tr):
            Xi_tr, Xi_te = X_tr.iloc[in_tr_idx], X_tr.iloc[in_te_idx]
            yi_tr, yi_te = y_tr.iloc[in_tr_idx], y_tr.iloc[in_te_idx]

            imputer = SimpleImputer(strategy="median")
            Xi_tr_imp = pd.DataFrame(imputer.fit_transform(Xi_tr), columns=Xi_tr.columns, index=Xi_tr.index)
            ranking, _ = rf_importance_ranking(Xi_tr_imp, yi_tr)

            for k in k_list:
                feats = ranking[:k]
                pred, _ = fit_predict(clf_name, Xi_tr[feats], yi_tr, Xi_te[feats], needs_scaling)
                if select_metric == "macro_f1":
                    s = f1_score(yi_te, pred, labels=LABELS, average="macro", zero_division=0)
                elif select_metric == "accuracy":
                    s = accuracy_score(yi_te, pred)
                else:
                    s = -mean_absolute_error(yi_te, pred)
                inner_scores[k].append(s)

        avg_inner = {k: float(np.mean(v)) for k, v in inner_scores.items()}
        best_k = max(avg_inner, key=avg_inner.get)

        # ---- outer fit: rank on full outer-train, eval once on outer-test ----
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
        ranking_full, importances_full = rf_importance_ranking(X_tr_imp, y_tr)
        feats = ranking_full[:best_k]

        pred, fitted_clf = fit_predict(clf_name, X_tr[feats], y_tr, X_te[feats], needs_scaling)
        oof_pred.iloc[te_idx] = pred

        m = metrics_block(y_te, pred)
        fold_log.append({
            "fold": fold_i, "best_k": best_k, "n_train": len(tr_idx), "n_test": len(te_idx),
            "selected_features": feats, **m,
            "inner_scores_by_k": avg_inner,
        })
        if verbose:
            print(f"  [{clf_name}] fold {fold_i}: k*={best_k:>2d}  "
                  f"acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}  mae={m['mae']:.3f}")

    oof_pred = oof_pred.astype(int)
    overall = metrics_block(y, oof_pred)
    cm = confusion_matrix(y, oof_pred, labels=LABELS)
    return {
        "classifier": clf_name,
        "fold_log": fold_log,
        "oof_pred": oof_pred,
        "overall": overall,
        "confusion_matrix": cm,
        "mean_fold_metrics": {
            "accuracy": float(np.mean([f["accuracy"] for f in fold_log])),
            "macro_f1": float(np.mean([f["macro_f1"] for f in fold_log])),
            "mae": float(np.mean([f["mae"] for f in fold_log])),
            "accuracy_std": float(np.std([f["accuracy"] for f in fold_log])),
            "macro_f1_std": float(np.std([f["macro_f1"] for f in fold_log])),
            "mae_std": float(np.std([f["mae"] for f in fold_log])),
        },
    }


def topk_sweep(clf_name, X, y, groups, k_list, n_outer=5, select_metric="macro_f1"):
    """
    Simple (non-nested) sweep: for each fixed k, run the SAME 5-fold outer
    GroupKFold, rank features on each outer-train fold only, refit restricted
    to top-k, evaluate once on the matching outer-test fold. Average across
    folds per k. This directly answers 'what is the best top-k setting for
    this task', at the cost of a small optimistic bias from re-using the
    same outer folds to both pick k and report its performance (see the
    nested_cv_topk function for an unbiased estimate of the winning k).
    """
    needs_scaling = build_classifier(clf_name)[1]
    outer_cv = GroupKFold(n_splits=n_outer, shuffle=True, random_state=RNG)
    splits = list(outer_cv.split(X, y, groups))

    rows = []
    oof_preds_by_k = {k: pd.Series(index=X.index, dtype=int) for k in k_list}

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
        ranking, _ = rf_importance_ranking(X_tr_imp, y_tr)

        for k in k_list:
            feats = ranking[:k]
            pred, _ = fit_predict(clf_name, X_tr[feats], y_tr, X_te[feats], needs_scaling)
            oof_preds_by_k[k].iloc[te_idx] = pred
            m = metrics_block(y_te, pred)
            rows.append({"k": k, "fold": fold_i, **m})

    sweep_df = pd.DataFrame(rows)
    summary = sweep_df.groupby("k")[["accuracy", "macro_f1", "mae"]].agg(["mean", "std"])
    best_k = summary[("macro_f1", "mean")].idxmax() if select_metric == "macro_f1" \
        else (summary[("accuracy", "mean")].idxmax() if select_metric == "accuracy"
              else summary[("mae", "mean")].idxmin())

    oof_preds_by_k = {k: v.astype(int) for k, v in oof_preds_by_k.items()}
    pooled_overall = {k: metrics_block(y, oof_preds_by_k[k]) for k in k_list}
    pooled_cm = {k: confusion_matrix(y, oof_preds_by_k[k], labels=LABELS) for k in k_list}

    return {
        "classifier": clf_name, "sweep_df": sweep_df, "summary": summary,
        "best_k": best_k, "pooled_overall": pooled_overall, "pooled_cm": pooled_cm,
        "oof_preds_by_k": oof_preds_by_k,
    }


def dummy_cv(strategy, X, y, groups, n_outer=5):
    outer_cv = GroupKFold(n_splits=n_outer, shuffle=True, random_state=RNG)
    oof_pred = pd.Series(index=X.index, dtype=int)
    for tr_idx, te_idx in outer_cv.split(X, y, groups):
        clf = DummyClassifier(strategy=strategy, random_state=RNG)
        clf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_pred.iloc[te_idx] = clf.predict(X.iloc[te_idx])
    oof_pred = oof_pred.astype(int)
    overall = metrics_block(y, oof_pred)
    cm = confusion_matrix(y, oof_pred, labels=LABELS)
    return {"classifier": f"dummy_{strategy}", "oof_pred": oof_pred, "overall": overall, "confusion_matrix": cm}
