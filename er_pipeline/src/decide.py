"""Turn pair probabilities into final match lists.

1. calibration   : isotonic regression fitted on the val_tune half
2. 1-to-1        : every S2/S3 record goes to at most one S1 (highest probability wins)
3. set selection : per S1 keep the top-k maximising expected F0.5
                     E[F_k] ~= 1.25 * sum_{i<=k} p_i / (k + 0.25 * (sum_all p + lam))
                     E[F_0]  = prod(1 - p_i)     (probability the entity is a singleton)
   A plain threshold rule is tuned as well; whichever scores higher on val_tune is used.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .evaluate import macro_f05
from .utils import LOG


def fit_calibrator(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p, y)
    return iso


def one_to_one(df: pd.DataFrame, floor: float) -> pd.DataFrame:
    d = df[df["p"] >= floor].sort_values("p", ascending=False, kind="mergesort")
    return d.drop_duplicates("t", keep="first").reset_index(drop=True)


def select_expected_f(d: pd.DataFrame, lam: float) -> pd.DataFrame:
    if d.empty:
        return d[["q", "t", "p"]]
    d = d.sort_values(["q", "p"], ascending=[True, False], kind="mergesort")
    g = d.groupby("q", sort=False)
    k = g.cumcount().values + 1
    s = g["p"].cumsum().values
    et = np.maximum(g["p"].transform("sum").values + lam, 1e-6)
    fk = 1.25 * s / (k + 0.25 * et)
    logq = np.log1p(-np.clip(d["p"].values, 0, 1 - 1e-7))
    f0 = np.exp(pd.Series(logq).groupby(d["q"].values).transform("sum").values)
    d = d.assign(_k=k, _fk=fk, _f0=f0)
    best = d.loc[d.groupby("q", sort=False)["_fk"].idxmax(), ["q", "_k", "_fk", "_f0"]]
    best = best[best["_fk"] > best["_f0"]]
    kmap = pd.Series(best["_k"].values, index=best["q"].values)
    kb = kmap.reindex(d["q"].values).fillna(0).values
    return d.loc[d["_k"].values <= kb, ["q", "t", "p"]]


def select_threshold(d: pd.DataFrame, thr: float) -> pd.DataFrame:
    return d.loc[d["p"] >= thr, ["q", "t", "p"]]


def apply(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    d = one_to_one(df, params["floor"])
    if params["method"] == "expected_f":
        return select_expected_f(d, params["lam"])
    return select_threshold(d, params["thr"])


def tune(df: pd.DataFrame, truth: pd.DataFrame, q_all: np.ndarray, floor_default: float) -> dict:
    results = []
    for floor, lam in itertools.product([floor_default, 0.05, 0.1, 0.2, 0.3], [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]):
        prm = {"method": "expected_f", "floor": floor, "lam": lam}
        results.append((macro_f05(apply(df, prm), truth, q_all), prm))
    for thr in np.round(np.arange(0.2, 0.96, 0.05), 3):
        prm = {"method": "threshold", "floor": floor_default, "thr": float(thr)}
        results.append((macro_f05(apply(df, prm), truth, q_all), prm))
    results.sort(key=lambda r: -r[0])
    for sc, prm in results[:5]:
        LOG.info(f"  tune: {sc:.5f}  {json.dumps(prm)}")
    best_ef = max((r for r in results if r[1]["method"] == "expected_f"), key=lambda r: r[0])
    best_th = max((r for r in results if r[1]["method"] == "threshold"), key=lambda r: r[0])
    LOG.info(f"  best expected-F {best_ef[0]:.5f} | best threshold {best_th[0]:.5f}")
    return dict(results[0][1], tune_score=results[0][0])
