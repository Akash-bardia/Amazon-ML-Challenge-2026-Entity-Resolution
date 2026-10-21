"""Macro F0.5 exactly as the leaderboard defines it.

Per S1 entity: F0.5 = 1.25*tp / (n_pred + 0.25*n_true)   (equivalent to the P/R formula)
               singleton (n_true = 0): 1.0 if n_pred == 0 else 0.0
Averaged over ALL S1 entities in the evaluation set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_entity(pred: pd.DataFrame, truth: pd.DataFrame, q_all: np.ndarray) -> pd.DataFrame:
    """pred/truth: DataFrames with q, t. Returns per-q n_pred, n_true, tp, f05."""
    q_all = np.unique(q_all)
    pred = pred[np.isin(pred["q"].values, q_all)][["q", "t"]].drop_duplicates()
    truth = truth[np.isin(truth["q"].values, q_all)][["q", "t"]].drop_duplicates()
    out = pd.DataFrame(index=pd.Index(q_all, name="q"))
    out["n_pred"] = pred.groupby("q").size().reindex(out.index, fill_value=0)
    out["n_true"] = truth.groupby("q").size().reindex(out.index, fill_value=0)
    tp = pred.merge(truth, on=["q", "t"]).groupby("q").size()
    out["tp"] = tp.reindex(out.index, fill_value=0)
    f = 1.25 * out["tp"] / (out["n_pred"] + 0.25 * out["n_true"]).where(lambda s: s > 0, 1)
    f[out["n_true"] == 0] = (out.loc[out["n_true"] == 0, "n_pred"] == 0).astype(float)
    out["f05"] = f.astype(float)
    return out


def summarize(pe: pd.DataFrame) -> dict:
    n_pred, n_true, tp = pe["n_pred"].sum(), pe["n_true"].sum(), pe["tp"].sum()
    single = pe["n_true"] == 0
    return {
        "n_entities": int(len(pe)),
        "macro_f05": float(pe["f05"].mean()) if len(pe) else 0.0,
        "micro_precision": float(tp / n_pred) if n_pred else 1.0,
        "micro_recall": float(tp / n_true) if n_true else 1.0,
        "false_positives": int(n_pred - tp),
        "false_negatives": int(n_true - tp),
        "singleton_rate_true": float(single.mean()) if len(pe) else 0.0,
        "singleton_rate_pred": float((pe["n_pred"] == 0).mean()) if len(pe) else 0.0,
        "singleton_accuracy": float((pe.loc[single, "n_pred"] == 0).mean()) if single.any() else 1.0,
        "mean_pred_per_s1": float(pe["n_pred"].mean()) if len(pe) else 0.0,
        "mean_true_per_s1": float(pe["n_true"].mean()) if len(pe) else 0.0,
    }


def macro_f05(pred: pd.DataFrame, truth: pd.DataFrame, q_all: np.ndarray) -> float:
    return summarize(per_entity(pred, truth, q_all))["macro_f05"]


def candidate_recall(cands: pd.DataFrame, truth: pd.DataFrame, q_all: np.ndarray) -> float:
    tr = truth[np.isin(truth["q"].values, q_all)][["q", "t"]].drop_duplicates()
    if tr.empty:
        return 1.0
    hit = tr.merge(cands[["q", "t"]].drop_duplicates(), on=["q", "t"]).shape[0]
    return hit / len(tr)
