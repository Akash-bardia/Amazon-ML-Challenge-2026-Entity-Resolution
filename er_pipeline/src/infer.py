"""Stage `infer`: score test candidates, decide, write + check the two submission files.

Uses exactly the same candidate files / feature code / model / calibrator / decision
as validation. Writes:
  output/matching_results.tsv, output/candidate_pairs.tsv
  work/model/test_pred.parquet      (pair probabilities, reused by the judge stage)
  work/reports/test_stats.json      (predicted-match statistics per country)
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import decide as D
from .config import Config
from .data import load_pool, load_s1, s1_countries
from .matcher import iter_features
from .utils import LOG, timed, write_id_list_tsv


def score_test(cfg: Config) -> pd.DataFrame:
    model = lgb.Booster(model_file=cfg.path("model", "lgb.txt"))
    with open(cfg.path("model", "features.json")) as f:
        feat_names = json.load(f)
    out = []
    for country in s1_countries(cfg, "test"):
        with timed(f"score test/{country}"):
            for pairs, fd in iter_features(cfg, "test", country, None):
                p = model.predict(fd[feat_names].astype(np.float32), num_iteration=model.best_iteration)
                out.append(pd.DataFrame({"q": pairs["q"].values, "t": pairs["t"].values,
                                         "p_raw": p.astype(np.float32), "country": country}))
    tp = pd.concat(out, ignore_index=True)
    tp.to_parquet(cfg.path("model", "test_pred.parquet"), index=False)
    return tp


def write_outputs(cfg: Config, tp: pd.DataFrame, p_col: str, tag: str) -> dict:
    with open(cfg.path("model", f"calibrator_{tag}.pkl"), "rb") as f:
        cal = pickle.load(f)
    with open(cfg.path("model", f"decision_{tag}.json")) as f:
        params = json.load(f)
    tp = tp.copy()
    tp["p"] = cal.predict(tp[p_col].values)
    pred = D.apply(tp[["q", "t", "p"]], params)

    s1 = load_s1(cfg, "test", ["id"])
    pool = load_pool(cfg, "test", ["id"])
    pid2id = pool["id"]
    rid2id = s1["id"]

    # candidate set = exactly the pairs the model scored
    cand = pd.concat([pd.read_parquet(f, columns=["q", "t", "rank_pre"]) for f in
                      glob.glob(os.path.join(cfg.work_dir, "cand", "test_*.parquet"))], ignore_index=True)

    # internal checks (vectorised)
    problems = []
    if pred.duplicated(["q", "t"]).any():
        problems.append("duplicate (S1, match) pairs")
    if pred.duplicated("t").any():
        problems.append("a S2/S3 record assigned to two S1 entities")
    if len(pred.merge(cand[["q", "t"]], on=["q", "t"])) != len(pred):
        problems.append("a matched id is not in the candidate set")
    if cand.duplicated(["q", "t"]).any():
        problems.append("duplicate candidate pairs")
    if (~pred["q"].isin(rid2id.index)).any() or (~pred["t"].isin(pid2id.index)).any():
        problems.append("unmapped ids in predictions")
    if problems:
        raise RuntimeError("output checks failed: " + "; ".join(problems))

    def joined(df: pd.DataFrame, order_col: str | None) -> pd.Series:
        if order_col:
            df = df.sort_values(["q", order_col])
        ids = pid2id.reindex(df["t"].values).values
        return pd.Series(ids, index=df["q"].values).groupby(level=0, sort=False).agg(",".join)

    match_str = joined(pred.assign(neg_p=-pred["p"]), "neg_p") if len(pred) else pd.Series(dtype=object)
    cand_str = joined(cand, "rank_pre")

    # order S1 exactly as in test_source1.tsv
    s1_sorted = s1.sort_index()
    s1_order = s1_sorted["id"].tolist()
    if len(set(s1_order)) != len(s1_order):
        raise RuntimeError("duplicate S1 ids in test_source1")
    m_map = dict(zip(rid2id.reindex(match_str.index).values, match_str.values))
    c_map = dict(zip(rid2id.reindex(cand_str.index).values, cand_str.values))
    os.makedirs(cfg.output_dir, exist_ok=True)
    m_path = os.path.join(cfg.output_dir, "matching_results.tsv")
    c_path = os.path.join(cfg.output_dir, "candidate_pairs.tsv")
    write_id_list_tsv(m_path, s1_order, m_map, "matched_entity_ids")
    write_id_list_tsv(c_path, s1_order, c_map, "candidate_entity_ids")
    LOG.info(f"wrote {m_path} and {c_path}")

    # statistics per country (compare with validation)
    n_pred = pred.groupby("q").size().reindex(s1_sorted.index, fill_value=0)
    ctry = s1_sorted["country"].values
    stats = {}
    for c in sorted(set(ctry)):
        m = ctry == c
        stats[c] = {"n_s1": int(m.sum()), "pred_singleton_rate": float((n_pred.values[m] == 0).mean()),
                    "mean_pred_per_s1": float(n_pred.values[m].mean())}
    stats["ALL"] = {"n_s1": len(s1_order), "pred_singleton_rate": float((n_pred.values == 0).mean()),
                    "mean_pred_per_s1": float(n_pred.values.mean())}
    with open(cfg.path("reports", f"test_stats_{tag}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    LOG.info("TEST prediction stats:\n" + json.dumps(stats, indent=2))
    return stats


def run_validator(cfg: Config) -> None:
    if not os.path.exists(cfg.validator_path):
        LOG.warning(f"validator not found at {cfg.validator_path}; skipping")
        return
    cmd = [sys.executable, cfg.validator_path,
           "--matching", os.path.join(cfg.output_dir, "matching_results.tsv"),
           "--candidate", os.path.join(cfg.output_dir, "candidate_pairs.tsv"),
           "--test-dir", os.path.join(cfg.data_dir, "test"), "--check-ids"]
    LOG.info("running official validator: " + " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    LOG.info(res.stdout + res.stderr)
    if res.returncode != 0:
        raise RuntimeError("official validator FAILED - do not submit")


def run(cfg: Config) -> None:
    tp = score_test(cfg)
    write_outputs(cfg, tp, "p_raw", "lgb")
    run_validator(cfg)
