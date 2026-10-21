"""Stages `features`, `train`, `validate`.

features : builds feature tables for the train fold (role=train) and the validation
           halves (val_tune / val_report) from the SAME candidate files the test uses.
train    : LightGBM on the train fold (early stopping on a 5% slice of the train fold).
validate : calibrate + tune the decision on val_tune, report on val_report.
"""
from __future__ import annotations

import glob
import json
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import decide as D
from . import features as F
from .blocking import safe_name
from .config import Config
from .data import load_pool, load_s1, pool_offset, s1_countries
from .evaluate import candidate_recall, per_entity, summarize
from .utils import LOG, read_ground_truth, stable_hash01, timed


# --------------------------------------------------------------------------- #
# shared: truth pairs in id space, chunked feature iterator
# --------------------------------------------------------------------------- #
def truth_pairs(cfg: Config) -> pd.DataFrame:
    """Train ground truth as integer (q, t) = (S1 rid, pool pid)."""
    gt = read_ground_truth(os.path.join(cfg.data_dir, "train", "train_ground_truth.tsv"))
    s1 = load_s1(cfg, "train", ["id"])
    pool = load_pool(cfg, "train", ["id"])
    q = pd.Series(s1.index.values, index=s1["id"].values).reindex(gt["q_id"].values).values
    t = pd.Series(pool.index.values, index=pool["id"].values).reindex(gt["t_id"].values).values
    ok = ~(pd.isna(q) | pd.isna(t))
    return pd.DataFrame({"q": q[ok].astype(np.int64), "t": t[ok].astype(np.int64)})


def iter_features(cfg: Config, split: str, country: str, keep_q: np.ndarray | None):
    """Yield (pairs, features) chunks; chunks never split one S1 across two chunks."""
    path = cfg.path("cand", f"{split}_{safe_name(country)}.parquet")
    cand = pd.read_parquet(path)
    if keep_q is not None:
        cand = cand[np.isin(cand["q"].values, keep_q)]
    if cand.empty:
        return
    cand = cand.sort_values(["q", "rank_pre"]).reset_index(drop=True)
    s1c = load_s1(cfg, split, F.FEAT_REC_COLS, country)
    poolc = load_pool(cfg, split, F.FEAT_REC_COLS, country)
    idf_name = F.idf_table(poolc["core"])
    idf_addr = F.idf_table(poolc["addr_n"])
    off = pool_offset(cfg, split)
    qv = cand["q"].values
    starts = np.flatnonzero(np.r_[True, qv[1:] != qv[:-1]])
    i = 0
    while i < len(cand):
        j = min(i + cfg.feat_chunk_pairs, len(cand))
        if j < len(cand):                                  # move j forward to an S1 boundary
            b = np.searchsorted(starts, j)
            j = starts[b] if b < len(starts) else len(cand)
        pairs = cand.iloc[i:j].reset_index(drop=True)
        yield pairs, F.build(pairs, s1c, poolc, off, idf_name, idf_addr)
        i = j


def roles(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.path("prep", "splits.parquet"))


# --------------------------------------------------------------------------- #
# stage: features
# --------------------------------------------------------------------------- #
def run_features(cfg: Config) -> None:
    r = roles(cfg)
    truth = truth_pairs(cfg).assign(label=np.int8(1))
    role_of = pd.Series(r["role"].values, index=r["rid"].values)
    parts = {"train": r.loc[r["role"] == "train", "rid"].values,
             "val": r.loc[r["role"].str.startswith("val"), "rid"].values}
    for country in s1_countries(cfg, "train"):
        for part, keep in parts.items():
            out = cfg.path("feat", f"{part}_{safe_name(country)}.parquet")
            if os.path.exists(out):
                LOG.info(f"skip existing {out}")
                continue
            with timed(f"features {part}/{country}"):
                chunks = []
                for pairs, fd in iter_features(cfg, "train", country, keep):
                    fd.insert(0, "q", pairs["q"].values)
                    fd.insert(1, "t", pairs["t"].values)
                    lab = pairs[["q", "t"]].merge(truth, on=["q", "t"], how="left")["label"]
                    fd.insert(2, "label", lab.fillna(0).astype(np.int8).values)
                    fd.insert(3, "role", role_of.reindex(pairs["q"].values).values)
                    fd.insert(4, "country", country)
                    chunks.append(fd)
                if chunks:
                    df = pd.concat(chunks, ignore_index=True)
                    df.to_parquet(out, index=False)
                    LOG.info(f"  {part}/{country}: {len(df):,} pairs, positives {int(df['label'].sum()):,}")


def load_feats(cfg: Config, part: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(cfg.work_dir, "feat", f"{part}_*.parquet")))
    if not files:
        raise FileNotFoundError(f"no feature files for part={part}; run the features stage first")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


META_COLS = ["q", "t", "label", "role", "country"]


# --------------------------------------------------------------------------- #
# stage: train
# --------------------------------------------------------------------------- #
def run_train(cfg: Config) -> None:
    df = load_feats(cfg, "train")
    feat_names = [c for c in df.columns if c not in META_COLS]
    es = stable_hash01(pd.Series(df["q"].astype(str).values), cfg.seed + 7) < 0.05
    X, y = df[feat_names].astype(np.float32), df["label"].values
    LOG.info(f"train rows {len(df):,} (positives {int(y.sum()):,}), features {len(feat_names)}")
    dtr = lgb.Dataset(X[~es], y[~es], feature_name=feat_names, free_raw_data=True)
    dva = lgb.Dataset(X[es], y[es], reference=dtr)
    params = dict(cfg.lgb_params, seed=cfg.seed, num_threads=os.cpu_count() or 4)
    with timed("lightgbm fit"):
        model = lgb.train(params, dtr, num_boost_round=cfg.lgb_rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(cfg.lgb_early_stop, verbose=False),
                                     lgb.log_evaluation(100)])
    model.save_model(cfg.path("model", "lgb.txt"))
    with open(cfg.path("model", "features.json"), "w") as f:
        json.dump(feat_names, f)
    imp = pd.Series(model.feature_importance("gain"), index=feat_names).sort_values(ascending=False)
    LOG.info("top features by gain:\n" + imp.head(20).to_string())
    del df, X

    val = load_feats(cfg, "val")
    val["p_raw"] = model.predict(val[feat_names].astype(np.float32), num_iteration=model.best_iteration)
    val[["q", "t", "label", "role", "country", "p_raw"]].to_parquet(cfg.path("model", "val_pred.parquet"),
                                                                    index=False)
    LOG.info(f"best iteration {model.best_iteration}; validation predictions saved")


# --------------------------------------------------------------------------- #
# stage: validate
# --------------------------------------------------------------------------- #
def run_validate(cfg: Config, pred_file: str = "val_pred.parquet", p_col: str = "p_raw",
                 tag: str = "lgb") -> dict:
    vp = pd.read_parquet(cfg.path("model", pred_file))
    r = roles(cfg)
    truth = truth_pairs(cfg)
    q_tune = r.loc[r["role"] == "val_tune", "rid"].values
    q_rep = r.loc[r["role"] == "val_report", "rid"].values
    country_of = pd.Series(r["country"].values, index=r["rid"].values)

    tune_rows = vp["role"] == "val_tune"
    cal = D.fit_calibrator(vp.loc[tune_rows, p_col].values, vp.loc[tune_rows, "label"].values)
    vp["p"] = cal.predict(vp[p_col].values)
    with timed("tune decision on val_tune"):
        params = D.tune(vp.loc[tune_rows, ["q", "t", "p"]], truth, q_tune, cfg.p_floor)

    rep = vp[vp["role"] == "val_report"][["q", "t", "p"]]
    pred = D.apply(rep, params)
    pe = per_entity(pred, truth, q_rep)
    pe["country"] = country_of.reindex(pe.index).values
    report = {"decision": params, "overall": summarize(pe),
              "by_country": {c: summarize(g) for c, g in pe.groupby("country")}}

    # candidate recall (final top-k set) and the union ceiling
    cand = pd.concat([pd.read_parquet(f, columns=["q", "t"]) for f in
                      glob.glob(os.path.join(cfg.work_dir, "cand", "train_*.parquet"))], ignore_index=True)
    report["candidate_recall_topk"] = candidate_recall(cand, truth, q_rep)
    uf = glob.glob(os.path.join(cfg.work_dir, "cand", "union_train_*.parquet"))
    if uf:
        union = pd.concat([pd.read_parquet(f) for f in uf], ignore_index=True)
        report["candidate_recall_union"] = candidate_recall(union, truth, q_rep)

    with open(cfg.path("model", f"decision_{tag}.json"), "w") as f:
        json.dump(params, f, indent=2)
    with open(cfg.path("model", f"calibrator_{tag}.pkl"), "wb") as f:
        pickle.dump(cal, f)
    with open(cfg.path("reports", f"validation_{tag}.json"), "w") as f:
        json.dump(report, f, indent=2)
    LOG.info(f"VALIDATION ({tag}, val_report half)\n" + json.dumps(report, indent=2))
    return report
