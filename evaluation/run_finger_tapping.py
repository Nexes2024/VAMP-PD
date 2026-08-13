"""
End-to-end driver: loads the finger-tapping dataset, runs the dummy baseline,
the Random-Forest top-k feature-selection sweep and nested CV.
Saves result tables and figures to the selected output directory.

Usage:
    python run_analysis.py /path/to/Fingertapping-final-bothsides.xlsx
"""
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

from finger_tapping_pipeline import (
    load_data, capped_k_list, nested_cv_topk, topk_sweep, dummy_cv,
    rf_importance_ranking, LABELS, RNG,
)

warnings.filterwarnings("ignore")

import os
OUT = None


def main(xlsx_path):
    df, X, y, groups, feature_cols = load_data(xlsx_path)
    k_list = capped_k_list(len(feature_cols))
    print(f"Loaded {len(df)} rows, {groups.nunique()} subjects, "
          f"{len(feature_cols)} candidate features.")
    print(f"Class counts: {y.value_counts().sort_index().to_dict()}")
    print(f"Top-k grid (capped at {len(feature_cols)} features): {k_list}")

    # ---------------------------------------------------------------- #
    # 1. Dummy baselines
    # ---------------------------------------------------------------- #
    print("\n=== Dummy baselines ===")
    dummies = {}
    for strat in ["most_frequent", "stratified"]:
        d = dummy_cv(strat, X, y, groups)
        dummies[strat] = d
        print(f"  {strat:14s}  acc={d['overall']['accuracy']:.3f}  "
              f"macroF1={d['overall']['macro_f1']:.3f}  mae={d['overall']['mae']:.3f}")

    # ---------------------------------------------------------------- #
    # 2. RandomForest: non-nested top-k sweep (headline "best setting")
    # ---------------------------------------------------------------- #
    print("\n=== Random Forest: top-k sweep (non-nested) ===")
    sweep_rf = topk_sweep("random_forest", X, y, groups, k_list)
    print(sweep_rf["summary"].round(3))
    best_k_naive = sweep_rf["best_k"]
    print(f"  Best k (by mean fold macro-F1): {best_k_naive}")
    sweep_rf["summary"].round(4).to_csv(f"{OUT}/rf_topk_sweep_summary.csv")

    # ---------------------------------------------------------------- #
    # 3. RandomForest: nested CV (unbiased generalization estimate)
    # ---------------------------------------------------------------- #
    print("\n=== Random Forest: nested CV (unbiased) ===")
    res_rf = nested_cv_topk("random_forest", X, y, groups, k_list, n_outer=5, n_inner=4)
    print("Pooled out-of-fold metrics:", res_rf["overall"])
    print("Mean +/- std across outer folds:", res_rf["mean_fold_metrics"])

    cls_report = classification_report(y, res_rf["oof_pred"], labels=LABELS,
                                        target_names=[f"score={l}" for l in LABELS],
                                        zero_division=0, output_dict=True)
    pd.DataFrame(cls_report).T.round(3).to_csv(f"{OUT}/rf_nested_classification_report.csv")

    fold_summary = pd.DataFrame([
        {"fold": f["fold"], "best_k": f["best_k"], "n_train": f["n_train"],
         "n_test": f["n_test"], "accuracy": f["accuracy"], "macro_f1": f["macro_f1"],
         "mae": f["mae"], "top_features": ", ".join(f["selected_features"][:8])}
        for f in res_rf["fold_log"]
    ])
    fold_summary.to_csv(f"{OUT}/rf_nested_fold_summary.csv", index=False)
    print(fold_summary[["fold", "best_k", "n_test", "accuracy", "macro_f1", "mae"]])

    # ---------------------------------------------------------------- #
    # 4. Gap check vs dummy -> decide whether other classifiers are required
    # ---------------------------------------------------------------- #
    best_dummy_acc = max(d["overall"]["accuracy"] for d in dummies.values())
    best_dummy_f1 = max(d["overall"]["macro_f1"] for d in dummies.values())
    gap_acc = res_rf["overall"]["accuracy"] - best_dummy_acc
    gap_f1 = res_rf["overall"]["macro_f1"] - best_dummy_f1
    print(f"\nRF vs best dummy: accuracy gap = {gap_acc*100:.1f}pp, "
          f"macro-F1 gap = {gap_f1*100:.1f}pp")
    needs_other_clf = (gap_acc < 0.10) or (gap_f1 < 0.10)
    print(f"Gap < 10pp on either metric -> {needs_other_clf} "
          f"(per spec, other classifiers are {'REQUIRED' if needs_other_clf else 'optional'})")

    # ---------------------------------------------------------------- #
    # 5. Full-data RF feature importance (for reporting / plot only)
    # ---------------------------------------------------------------- #
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    ranking_all, importances_all = rf_importance_ranking(X_imp, y)
    imp_df = pd.DataFrame({"feature": ranking_all, "importance": importances_all})
    imp_df.to_csv(f"{OUT}/feature_importance_full_data.csv", index=False)

    # ================================================================ #
    # FIGURES
    # ================================================================ #
    # Fig 1: macro-F1 (and accuracy) vs k
    fig, ax = plt.subplots(figsize=(6, 4))
    summ = sweep_rf["summary"]
    ax.errorbar(summ.index, summ[("macro_f1", "mean")], yerr=summ[("macro_f1", "std")],
                marker="o", capsize=3, label="Macro-F1", color="#2c5f8a")
    ax.errorbar(summ.index, summ[("accuracy", "mean")], yerr=summ[("accuracy", "std")],
                marker="s", capsize=3, label="Accuracy", color="#d97b29")
    ax.axhline(best_dummy_acc, ls="--", color="gray", lw=1, label="Dummy accuracy (best)")
    ax.set_xlabel("Top-k features")
    ax.set_ylabel("Score (mean ± std across 5 outer folds)")
    ax.set_title("Random Forest: performance vs. top-k features")
    ax.set_xticks(summ.index)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_topk_sweep.png", dpi=150)
    plt.close(fig)

    # Fig 2: confusion matrix (nested CV, pooled OOF)
    cm = res_rf["confusion_matrix"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            txt_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)", ha="center", va="center",
                    color=txt_color, fontsize=9)
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels([f"pred {l}" for l in LABELS])
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels([f"true {l}" for l in LABELS])
    ax.set_title("RF nested CV — out-of-fold confusion matrix\n(row-normalized %)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Fig 3: feature importance (full-data fit, top 15)
    top15 = imp_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top15["feature"], top15["importance"], color="#2c5f8a")
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title("RF feature importance — full dataset (top 15)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_feature_importance.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- #
    # Save a master JSON with everything needed for the report
    # ---------------------------------------------------------------- #
    master = {
        "n_rows": len(df), "n_subjects": int(groups.nunique()), "n_features": len(feature_cols),
        "feature_cols": feature_cols, "k_list": k_list,
        "class_counts": {int(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "dummy": {k: v["overall"] for k, v in dummies.items()},
        "rf_sweep_best_k": int(best_k_naive),
        "rf_sweep_pooled_at_best_k": sweep_rf["pooled_overall"][best_k_naive],
        "rf_nested_overall": res_rf["overall"],
        "rf_nested_mean_fold": res_rf["mean_fold_metrics"],
        "rf_nested_fold_ks": [f["best_k"] for f in res_rf["fold_log"]],
        "gap_acc_pp": gap_acc * 100, "gap_f1_pp": gap_f1 * 100,
        "needs_other_classifiers_per_spec": bool(needs_other_clf),
        "other_classifiers": other_results,
        "tenfold_classes_missing_folds": int((fold10_df["n_classes_present"] < 4).sum()),
        "top_features_full_data": imp_df.head(10).to_dict("records"),
    }
    with open(f"{OUT}/master_results.json", "w") as f:
        json.dump(master, f, indent=2, default=str)

    print("\nDone. Outputs written to ./outputs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input feature spreadsheet")
    parser.add_argument("--outdir", default="outputs/finger_tapping")
    args = parser.parse_args()

    OUT = args.outdir
    os.makedirs(OUT, exist_ok=True)
    main(args.data)
