import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from src.config import Config
from src.matcher import roles, truth_pairs
from src.evaluate import macro_f05, summarize, per_entity
import src.decide as D

def main():
    cfg = Config()
    vp = pd.read_parquet(cfg.path("model", "val_pred.parquet"))
    r = roles(cfg)
    truth = truth_pairs(cfg)
    q_tune = r.loc[r["role"] == "val_tune", "rid"].values
    q_rep = r.loc[r["role"] == "val_report", "rid"].values
    country_of = pd.Series(r["country"].values, index=r["rid"].values)

    print("Loading candidate tables to join candidate metadata...")
    cand_files = sorted(glob.glob(os.path.join(cfg.work_dir, "cand", "train_*.parquet")))
    cand_dfs = []
    cols_to_use = ["q", "t", "n_keys", "cos_full", "cos_addr", "t_best", "t_nq", "q_best", "rank_pre", "pre"]
    for f in cand_files:
        cand_dfs.append(pd.read_parquet(f, columns=cols_to_use))
    cands = pd.concat(cand_dfs, ignore_index=True)
    print(f"Loaded {len(cands):,} candidate rows.")

    vp = vp.merge(cands, on=["q", "t"], how="left")
    print("Merged candidate features into val_pred. Columns:", vp.columns.tolist())

    tune_m = vp["role"] == "val_tune"
    rep_m = vp["role"] == "val_report"

    # Baseline score with current decision:
    import pickle
    with open(cfg.path("model", "calibrator_lgb.pkl"), "rb") as f:
        cal = pickle.load(f)
    p_base = cal.predict(vp["p_raw"].values)
    d_rep_base = D.one_to_one(vp.assign(p=p_base).loc[rep_m, ["q", "t", "p"]], 0.02)
    base_sc = macro_f05(d_rep_base[d_rep_base["p"] >= 0.75], truth, q_rep)
    print(f"Baseline val_report macro_f05: {base_sc:.5f}")

    # Build a fast blender model on top of p_raw and candidate features
    feat_cols = ["p_raw", "cos_full", "cos_addr", "n_keys", "rank_pre", "t_best", "q_best", "pre"]
    # Fill NAs
    for col in feat_cols:
        vp[col] = vp[col].fillna(0.0)

    # Add logit(p_raw)
    eps = 1e-7
    p_clip = np.clip(vp["p_raw"].values, eps, 1 - eps)
    vp["logit_p"] = np.log(p_clip / (1 - p_clip))
    feat_cols.append("logit_p")

    X_tune = vp.loc[tune_m, feat_cols].values
    y_tune = vp.loc[tune_m, "label"].values
    X_rep = vp.loc[rep_m, feat_cols].values
    y_rep = vp.loc[rep_m, "label"].values

    print(f"Training Logistic Regression blender on {len(X_tune):,} rows...")
    blender = LogisticRegression(C=1.0, max_iter=200)
    blender.fit(X_tune, y_tune)
    p_tune_blend = blender.predict_proba(X_tune)[:, 1]
    p_rep_blend = blender.predict_proba(X_rep)[:, 1]

    # Evaluate blender:
    vp_tune_b = vp.loc[tune_m, ["q", "t"]].assign(p=p_tune_blend)
    vp_rep_b = vp.loc[rep_m, ["q", "t"]].assign(p=p_rep_blend)

    d_tune_b = D.one_to_one(vp_tune_b, 0.02)
    d_rep_b = D.one_to_one(vp_rep_b, 0.02)

    best_thr = 0.5
    best_sc = 0.0
    for thr in np.round(np.arange(0.2, 0.95, 0.02), 2):
        sc = macro_f05(d_tune_b[d_tune_b["p"] >= thr], truth, q_tune)
        if sc > best_sc:
            best_sc = sc
            best_thr = thr

    rep_sc = macro_f05(d_rep_b[d_rep_b["p"] >= best_thr], truth, q_rep)
    print(f"Blender (Logistic): best tune thr {best_thr:.2f} (tune {best_sc:.5f}) -> honest rep: {rep_sc:.5f}")

    # Also try LightGBM fast shallow booster
    dtr = lgb.Dataset(X_tune, y_tune, feature_name=feat_cols)
    dva = lgb.Dataset(X_rep, y_rep, feature_name=feat_cols, reference=dtr)
    lgb_params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "max_depth": 4, "verbose": -1, "n_jobs": 4}
    gbm = lgb.train(lgb_params, dtr, num_boost_round=100)
    p_tune_gbm = gbm.predict(X_tune)
    p_rep_gbm = gbm.predict(X_rep)

    d_tune_g = D.one_to_one(vp.loc[tune_m, ["q", "t"]].assign(p=p_tune_gbm), 0.02)
    d_rep_g = D.one_to_one(vp.loc[rep_m, ["q", "t"]].assign(p=p_rep_gbm), 0.02)

    best_thr_g = 0.5
    best_sc_g = 0.0
    for thr in np.round(np.arange(0.2, 0.95, 0.02), 2):
        sc = macro_f05(d_tune_g[d_tune_g["p"] >= thr], truth, q_tune)
        if sc > best_sc_g:
            best_sc_g = sc
            best_thr_g = thr

    rep_sc_g = macro_f05(d_rep_g[d_rep_g["p"] >= best_thr_g], truth, q_rep)
    print(f"Blender (LightGBM): best tune thr {best_thr_g:.2f} (tune {best_sc_g:.5f}) -> honest rep: {rep_sc_g:.5f}")

if __name__ == "__main__":
    main()
