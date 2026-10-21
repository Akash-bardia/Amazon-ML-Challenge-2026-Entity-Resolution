"""Pair features (vectorised).

String similarities use rapidfuzz.process.cpdist (C++, multi-threaded, element-wise).
Token-overlap / IDF features use sparse binary matrices so there is no Python loop
over pairs. Country is never a feature (open-set country labels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

FEAT_REC_COLS = ["name_n", "core", "alt", "legal", "web", "addr_n", "nums", "cities", "state",
                 "script", "addr_empty"]


# --------------------------------------------------------------------------- #
# IDF tables
# --------------------------------------------------------------------------- #
def idf_table(strings: pd.Series) -> tuple[pd.Series, float]:
    """IDF over documents (records); returns (token -> idf, idf for unseen tokens)."""
    n = len(strings)
    df = None
    for i in range(0, n, 500_000):                         # chunked to bound memory
        s = pd.Series(strings.values[i:i + 500_000])
        toks = s.str.split().explode()
        toks = toks[toks.notna() & (toks != "")]
        d = pd.DataFrame({"r": toks.index.values, "tok": toks.values}).drop_duplicates()
        vc = d["tok"].value_counts()
        df = vc if df is None else df.add(vc, fill_value=0)
    if df is None:
        return pd.Series(dtype=np.float32), float(np.log(n + 1) + 1.0)
    idf = np.log((n + 1) / (df + 1)) + 1.0
    return idf.astype(np.float32), float(np.log(n + 1) + 1.0)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sim(a: np.ndarray, b: np.ndarray, scorer, mask: np.ndarray | None = None, scale=100.0) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=np.float32)
    if mask is None:
        mask = (a != "") & (b != "")
    idx = np.flatnonzero(mask)
    if len(idx):
        out[idx] = process.cpdist(a[idx].tolist(), b[idx].tolist(), scorer=scorer, workers=-1,
                                  dtype=np.float32) / scale
    return out


def _token_stats(a: np.ndarray, b: np.ndarray, idf: pd.Series | None = None, unseen: float = 1.0,
                 sep: str | None = None) -> dict:
    """Binary token overlap statistics between a[i] and b[i]."""
    n = len(a)
    sa = pd.Series(a).str.split(sep).explode()
    sb = pd.Series(b).str.split(sep).explode()
    sa = sa[sa.notna() & (sa != "")]
    sb = sb[sb.notna() & (sb != "")]
    codes, uniq = pd.factorize(pd.concat([sa, sb], ignore_index=True))
    V = max(len(uniq), 1)
    ca, cb = codes[: len(sa)], codes[len(sa):]
    A = sp.csr_matrix((np.ones(len(ca), np.float32), (sa.index.values, ca)), shape=(n, V))
    B = sp.csr_matrix((np.ones(len(cb), np.float32), (sb.index.values, cb)), shape=(n, V))
    A.sum_duplicates()
    B.sum_duplicates()
    A.data[:] = 1.0            # binary: a token repeated in one string counts once
    B.data[:] = 1.0
    C = A.multiply(B).tocsr()
    res = {
        "common": np.asarray(C.sum(1)).ravel(),
        "na": np.asarray(A.sum(1)).ravel(),
        "nb": np.asarray(B.sum(1)).ravel(),
    }
    if idf is not None:
        w = pd.Series(uniq).map(idf).fillna(unseen).values.astype(np.float32)
        res["ci"] = C @ w
        res["ai"] = A @ w
        res["bi"] = B @ w
        Cw = C.multiply(w[None, :]).tocsr()
        res["cmax"] = np.asarray(Cw.max(axis=1).todense()).ravel() if Cw.nnz else np.zeros(n, np.float32)
    return res


def _first_tok(s: np.ndarray) -> np.ndarray:
    return pd.Series(s).str.split(n=1).str[0].fillna("").values


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)
    return r.astype(np.float32)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build(pairs: pd.DataFrame, s1c: pd.DataFrame, poolc: pd.DataFrame, pool_off: int,
          idf_name: tuple, idf_addr: tuple) -> pd.DataFrame:
    """pairs: q, t (+ blocking columns). s1c/poolc indexed by global id with FEAT_REC_COLS."""
    qpos = s1c.index.get_indexer(pairs["q"].values)
    tpos = poolc.index.get_indexer(pairs["t"].values)
    assert (qpos >= 0).all() and (tpos >= 0).all(), "candidate ids missing from records"
    Q = {c: s1c[c].values[qpos] for c in FEAT_REC_COLS}
    T = {c: poolc[c].values[tpos] for c in FEAT_REC_COLS}
    f = {}

    # ---------------- names ----------------
    qc, tc = Q["core"].astype(object), T["core"].astype(object)
    f["n_exact"] = (qc == tc).astype(np.float32)
    f["n_ratio"] = _sim(qc, tc, fuzz.ratio)
    f["n_tsort"] = _sim(qc, tc, fuzz.token_sort_ratio)
    f["n_tset"] = _sim(qc, tc, fuzz.token_set_ratio)
    f["n_partial"] = _sim(qc, tc, fuzz.partial_ratio)
    f["n_jw"] = _sim(qc, tc, JaroWinkler.normalized_similarity, scale=1.0)
    f["n_full_ratio"] = _sim(Q["name_n"], T["name_n"], fuzz.ratio)

    qcc = pd.Series(qc).str.replace(" ", "", regex=False).values
    tcc = pd.Series(tc).str.replace(" ", "", regex=False).values
    talt = pd.Series(T["alt"]).str.replace(" ", "", regex=False).values
    qalt = pd.Series(Q["alt"]).str.replace(" ", "", regex=False).values
    alts = [
        _sim(qcc, talt, fuzz.ratio),
        _sim(qalt, tcc, fuzz.ratio),
        _sim(qcc, T["web"].astype(object), fuzz.ratio),
        _sim(Q["web"].astype(object), tcc, fuzz.ratio),
    ]
    with np.errstate(all="ignore"):
        f["n_alt"] = np.fmax.reduce(alts)
    f["n_compact_eq"] = ((qcc == tcc) | ((talt != "") & (qcc == talt)) | ((T["web"] != "") & (qcc == T["web"]))
                         ).astype(np.float32)
    f["n_first_eq"] = (_first_tok(qc) == _first_tok(tc)).astype(np.float32)

    ts = _token_stats(qc, tc, *idf_name)
    f["n_common"] = ts["common"].astype(np.float32)
    f["n_jacc"] = _safe_div(ts["common"], ts["na"] + ts["nb"] - ts["common"])
    f["n_idf_jacc"] = _safe_div(ts["ci"], ts["ai"] + ts["bi"] - ts["ci"])
    f["n_idf_max"] = ts["cmax"].astype(np.float32)
    f["n_q_unm_idf"] = (ts["ai"] - ts["ci"]).astype(np.float32)
    f["n_t_unm_idf"] = (ts["bi"] - ts["ci"]).astype(np.float32)
    f["n_ntok_q"] = ts["na"].astype(np.float32)
    f["n_ntok_t"] = ts["nb"].astype(np.float32)
    f["n_len_q"] = pd.Series(qc).str.len().values.astype(np.float32)
    f["n_len_t"] = pd.Series(tc).str.len().values.astype(np.float32)

    lg = _token_stats(Q["legal"].astype(object), T["legal"].astype(object))
    both = (lg["na"] > 0) & (lg["nb"] > 0)
    f["legal_match"] = (lg["common"] > 0).astype(np.float32)
    f["legal_conflict"] = (both & (lg["common"] == 0)).astype(np.float32)
    f["t_script"] = T["script"].astype(np.float32)

    # ---------------- addresses ----------------
    qa, ta = Q["addr_n"].astype(object), T["addr_n"].astype(object)
    f["t_addr_empty"] = T["addr_empty"].astype(np.float32)
    f["a_ratio"] = _sim(qa, ta, fuzz.ratio)
    f["a_tsort"] = _sim(qa, ta, fuzz.token_sort_ratio)
    f["a_tset"] = _sim(qa, ta, fuzz.token_set_ratio)
    f["a_partial"] = _sim(qa, ta, fuzz.partial_ratio)
    at = _token_stats(qa, ta, *idf_addr)
    has_a = (at["na"] > 0) & (at["nb"] > 0)
    f["a_common"] = np.where(has_a, at["common"], np.nan).astype(np.float32)
    f["a_idf_jacc"] = np.where(has_a, _safe_div(at["ci"], at["ai"] + at["bi"] - at["ci"]), np.nan).astype(np.float32)
    f["a_idf_max"] = np.where(has_a, at["cmax"], np.nan).astype(np.float32)
    f["a_q_unm_idf"] = np.where(has_a, at["ai"] - at["ci"], np.nan).astype(np.float32)
    f["a_t_unm_idf"] = np.where(has_a, at["bi"] - at["ci"], np.nan).astype(np.float32)

    qn, tn = Q["nums"].astype(object), T["nums"].astype(object)
    ns = _token_stats(qn, tn)
    bothn = (ns["na"] > 0) & (ns["nb"] > 0)
    f["num_common"] = ns["common"].astype(np.float32)
    f["num_q"] = ns["na"].astype(np.float32)
    f["num_t"] = ns["nb"].astype(np.float32)
    f["num_conflict"] = (bothn & (ns["common"] == 0)).astype(np.float32)
    f["num_all_q"] = np.where(ns["na"] > 0, ns["common"] == ns["na"], np.nan).astype(np.float32)
    f["num_first_eq"] = np.where(bothn, _first_tok(qn) == _first_tok(tn), np.nan).astype(np.float32)
    f["num_partial"] = _sim(qn, tn, fuzz.partial_ratio)

    cs = _token_stats(Q["cities"].astype(object), T["cities"].astype(object), sep="|")
    bothc = (cs["na"] > 0) & (cs["nb"] > 0)
    f["city_match"] = (cs["common"] > 0).astype(np.float32)
    f["city_conflict"] = (bothc & (cs["common"] == 0)).astype(np.float32)
    qs, tst = Q["state"], T["state"]
    boths = (qs != "") & (tst != "")
    f["state_match"] = (boths & (qs == tst)).astype(np.float32)
    f["state_conflict"] = (boths & (qs != tst)).astype(np.float32)

    # ---------------- blocking / competition ----------------
    f["cos_full"] = pairs["cos_full"].values.astype(np.float32)
    f["cos_addr"] = pairs["cos_addr"].values.astype(np.float32)
    f["n_keys"] = pairs["n_keys"].values.astype(np.float32)
    src = pairs["src"].values
    f["in_d1"] = (src & 1 > 0).astype(np.float32)
    f["in_d2"] = (src & 2 > 0).astype(np.float32)
    f["in_k"] = (src & 4 > 0).astype(np.float32)
    f["rank_pre"] = pairs["rank_pre"].values.astype(np.float32)
    f["rank_full"] = pairs["rank_full"].values.astype(np.float32)
    f["t_gap"] = (pairs["t_best"].values - pairs["cos_full"].values).astype(np.float32)
    f["q_gap"] = (pairs["q_best"].values - pairs["cos_full"].values).astype(np.float32)
    f["t_nq"] = pairs["t_nq"].values.astype(np.float32)
    f["q_nu"] = pairs["q_nu"].values.astype(np.float32)
    f["is_s2"] = (pairs["t"].values < pool_off).astype(np.float32)

    # ---------------- cross ----------------
    f["x_name_addr"] = (np.nan_to_num(f["n_tset"]) * np.where(np.isnan(f["a_tset"]), 1.0, f["a_tset"])
                        ).astype(np.float32)

    # ---------------- within-S1 context ----------------
    fd = pd.DataFrame(f)
    grp = pd.Series(pairs["q"].values)
    fd["n_ratio_rank"] = pd.Series(fd["n_ratio"].fillna(-1).values).groupby(grp.values).rank(
        ascending=False, method="min").values.astype(np.float32)
    fd["x_rank"] = pd.Series(fd["x_name_addr"].values).groupby(grp.values).rank(
        ascending=False, method="min").values.astype(np.float32)
    return fd


FEATURE_NAMES: list[str] | None = None   # filled at training time and stored with the model
