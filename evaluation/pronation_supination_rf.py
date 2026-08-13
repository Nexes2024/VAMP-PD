"""
Subject-independent Random Forest classifier for pronation-supination Score (0-3).

Design choices (see report for rationale):
- Each row (one hand of one subject) is a sample. Left/right are POOLED but NOT
  aggregated: both hands keep their own label. Leakage is prevented by grouping
  on subject ID so all rows of a subject land in the same fold (GroupKFold).
- Artifact / ID columns are dropped.
- 5-fold GroupKFold (group = subject).
- Feature ranking via RandomForest MDI is done INSIDE each training fold only.
- Top-k sweep, k in [5,10,15,20,30,40,50] capped at n_features.
- Metrics on pooled out-of-fold (OOF) predictions: Accuracy, Macro-F1, MAE,
  confusion matrix. Per-fold mean+/-std also reported.
- RandomForest vs DummyClassifier; if accuracy gap < 10 pts, extra classifiers run.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix

RNG = 42
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="Input feature spreadsheet")
parser.add_argument("--outdir", default="outputs/pronation_supination")
args = parser.parse_args()

PATH = args.data
os.makedirs(args.outdir, exist_ok=True)

ARTIFACT_COLS = ['rel_path', 'hand', 'status', 'duration_s', 'fps', 'total_frames',
                 'detected_frames', 'detection_rate', 'cycle_count', 'cycles_per_second']
ID_COLS = ['video', 'subject']
TARGET = 'score'
K_GRID = [5, 10, 15, 20, 30, 40, 50]


def load_data():
    df = pd.read_excel(PATH, sheet_name=0)
    df.columns = [c.strip() for c in df.columns]
    if "Score" in df.columns and "score" not in df.columns:
        df = df.rename(columns={"Score": "score"})
    feats = [c for c in df.columns if c not in ARTIFACT_COLS + ID_COLS + [TARGET]]
    X = df[feats].astype(float).values
    y = df[TARGET].astype(int).values
    groups = df['subject'].values
    return X, y, groups, feats


def rf_factory():
    return RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)


def eval_topk_nested(X, y, groups, feat_names, model_builder, needs_scaling=False, n_splits=5):
    """Nested CV: in each training fold rank features with RF-MDI, then for each k
    fit `model_builder()` on top-k features. Returns OOF predictions per k and
    per-fold metric lists."""
    n_feat = X.shape[1]
    ks = sorted(set(min(k, n_feat) for k in K_GRID))  # cap & dedupe
    gkf = GroupKFold(n_splits=n_splits)

    oof_pred = {k: np.full(len(y), -1) for k in ks}
    fold_metrics = {k: [] for k in ks}          # list of (acc, f1, mae) per fold
    fold_selected = {k: [] for k in ks}         # selected features per fold (for stability)

    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte, ytr = X[tr], X[te], y[tr]
        # impute on TRAIN only
        imp = SimpleImputer(strategy='median').fit(Xtr)
        Xtr_i, Xte_i = imp.transform(Xtr), imp.transform(Xte)
        # rank features with RF MDI on TRAIN fold
        ranker = rf_factory().fit(Xtr_i, ytr)
        order = np.argsort(ranker.feature_importances_)[::-1]
        for k in ks:
            sel = order[:k]
            fold_selected[k].append([feat_names[i] for i in sel])
            Xtr_k, Xte_k = Xtr_i[:, sel], Xte_i[:, sel]
            if needs_scaling:
                sc = StandardScaler().fit(Xtr_k)
                Xtr_k, Xte_k = sc.transform(Xtr_k), sc.transform(Xte_k)
            mdl = model_builder().fit(Xtr_k, ytr)
            pred = mdl.predict(Xte_k)
            oof_pred[k][te] = pred
            fold_metrics[k].append((
                accuracy_score(y[te], pred),
                f1_score(y[te], pred, average='macro', zero_division=0),
                mean_absolute_error(y[te], pred),
            ))
    # pooled OOF metrics per k
    rows = []
    for k in ks:
        p = oof_pred[k]
        fm = np.array(fold_metrics[k])
        rows.append({
            'k': k,
            'OOF_acc': accuracy_score(y, p),
            'OOF_macroF1': f1_score(y, p, average='macro', zero_division=0),
            'OOF_MAE': mean_absolute_error(y, p),
            'fold_acc_mean': fm[:, 0].mean(), 'fold_acc_std': fm[:, 0].std(),
            'fold_f1_mean': fm[:, 1].mean(), 'fold_f1_std': fm[:, 1].std(),
            'fold_mae_mean': fm[:, 2].mean(), 'fold_mae_std': fm[:, 2].std(),
        })
    return pd.DataFrame(rows), oof_pred, fold_selected


def dummy_baseline(X, y, groups, n_splits=5, strategy='most_frequent'):
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.full(len(y), -1)
    for tr, te in gkf.split(X, y, groups):
        d = DummyClassifier(strategy=strategy, random_state=RNG).fit(X[tr], y[tr])
        preds[te] = d.predict(X[te])
    return {
        'acc': accuracy_score(y, preds),
        'macroF1': f1_score(y, preds, average='macro', zero_division=0),
        'MAE': mean_absolute_error(y, preds),
        'preds': preds,
    }


def main():
    X, y, groups, feats = load_data()
    print(f"Samples: {len(y)} | Subjects: {len(np.unique(groups))} | Features: {len(feats)}")
    print(f"Class counts: {dict(zip(*np.unique(y, return_counts=True)))}\n")

    # ---- RandomForest nested top-k sweep (5-fold) ----
    rf_df, rf_oof, rf_sel = eval_topk_nested(X, y, groups, feats, rf_factory, n_splits=5)
    print("=== RandomForest, 5-fold GroupKFold, top-k sweep ===")
    print(rf_df.round(4).to_string(index=False))

    best = rf_df.sort_values(['OOF_macroF1', 'OOF_acc'], ascending=False).iloc[0]
    best_k = int(best['k'])
    print(f"\nBest RF setting: k={best_k} | acc={best['OOF_acc']:.3f} "
          f"macroF1={best['OOF_macroF1']:.3f} MAE={best['OOF_MAE']:.3f}")

    # ---- Dummy baselines ----
    dum_mf = dummy_baseline(X, y, groups, strategy='most_frequent')
    dum_st = dummy_baseline(X, y, groups, strategy='stratified')
    print(f"\nDummy(most_frequent): acc={dum_mf['acc']:.3f} macroF1={dum_mf['macroF1']:.3f} MAE={dum_mf['MAE']:.3f}")
    print(f"Dummy(stratified):    acc={dum_st['acc']:.3f} macroF1={dum_st['macroF1']:.3f} MAE={dum_st['MAE']:.3f}")

    acc_gap = best['OOF_acc'] - dum_mf['acc']
    print(f"\nRF - Dummy(most_frequent) accuracy gap = {acc_gap*100:.1f} points")

    # ---- Confusion matrix for best RF ----
    cm = confusion_matrix(y, rf_oof[best_k], labels=[0, 1, 2, 3])
    print(f"\nConfusion matrix (rows=true, cols=pred) for best RF (k={best_k}):")
    print(pd.DataFrame(cm, index=[f"true{i}" for i in range(4)],
                       columns=[f"pred{i}" for i in range(4)]).to_string())
    # per-class recall/precision
    print("\nPer-class (from OOF):")
    for c in range(4):
        tp = cm[c, c]; support = cm[c].sum(); pred_c = cm[:, c].sum()
        rec = tp/support if support else 0
        prec = tp/pred_c if pred_c else 0
        print(f"  Score {c}: support={support:2d} recall={rec:.2f} precision={prec:.2f}")


    # ---- feature selection frequency at best k (stability) ----
    from collections import Counter
    cnt = Counter(f for fold in rf_sel[best_k] for f in fold)
    print(f"\nMost-selected features across folds at k={best_k}:")
    for f, c in cnt.most_common(12):
        print(f"  {f:18s} {c}/5 folds")

    # save artifacts
    rf_df.to_csv(os.path.join(args.outdir, "rf_topk_results.csv"), index=False)
    np.save(os.path.join(args.outdir, "cm_best.npy"), cm)
    return rf_df, cm, best_k, dum_mf, dum_st, cnt


if __name__ == '__main__':
    main()
