import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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

    tune_m = vp["role"] == "val_tune"
    rep_m = vp["role"] == "val_report"

    eps = 1e-7
    p_clipped = np.clip(vp["p_raw"].values, eps, 1 - eps)
    logit = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)

    # 1. Global Platt
    lr_global = LogisticRegression(C=1.0)
    lr_global.fit(logit[tune_m], vp.loc[tune_m, "label"].values)

    # 2. Per-country Platt
    cals = {}
    for ctry in ["India", "US"]:
        lr_c = LogisticRegression(C=1.0)
        m = tune_m & (vp["country"] == ctry)
        lr_c.fit(logit[m], vp.loc[m, "label"].values)
        cals[ctry] = lr_c

    # Predict probabilities for tune and rep
    p_tune_c = np.zeros(tune_m.sum())
    p_rep_c = np.zeros(rep_m.sum())

    tune_indices = np.where(tune_m)[0]
    rep_indices = np.where(rep_m)[0]

    vp_tune = vp.loc[tune_m].copy()
    vp_rep = vp.loc[rep_m].copy()

    for ctry in ["India", "US"]:
        m_t = vp_tune["country"] == ctry
        m_r = vp_rep["country"] == ctry
        idx_t = logit[tune_indices[m_t]]
        idx_r = logit[rep_indices[m_r]]
        vp_tune.loc[m_t, "p"] = cals[ctry].predict_proba(idx_t)[:, 1]
        vp_rep.loc[m_r, "p"] = cals[ctry].predict_proba(idx_r)[:, 1]

    # Tune per-country thresholds
    best_thrs = {}
    for ctry in ["India", "US"]:
        q_tune_c = r.loc[(r["role"] == "val_tune") & (r["country"] == ctry), "rid"].values
        vp_tune_c = vp_tune[vp_tune["country"] == ctry]
        d_tune_c = D.one_to_one(vp_tune_c[["q", "t", "p"]], 0.02)
        best_c_thr = 0.70
        best_c_sc = 0.0
        for thr in np.round(np.arange(0.50, 0.90, 0.01), 3):
            pred = d_tune_c[d_tune_c["p"] >= thr]
            sc = macro_f05(pred, truth, q_tune_c)
            if sc > best_c_sc:
                best_c_sc = sc
                best_c_thr = thr
        best_thrs[ctry] = best_c_thr
        print(f"Country {ctry} Platt cal: best tune thr {best_c_thr:.3f} -> score {best_c_sc:.5f}")

    # Evaluate honestly on val_report
    preds_rep = []
    for ctry in ["India", "US"]:
        m_r = vp_rep["country"] == ctry
        d_r = D.one_to_one(vp_rep.loc[m_r, ["q", "t", "p"]], 0.02)
        preds_rep.append(d_r[d_r["p"] >= best_thrs[ctry]])
    combined_rep = pd.concat(preds_rep, ignore_index=True)
    pe_sep = per_entity(combined_rep, truth, q_rep)
    pe_sep["country"] = country_of.reindex(pe_sep.index).values
    s_sep = summarize(pe_sep)
    print(f"\nHonest val_report with Per-Country Platt Calibration + Per-Country Thrs:")
    print(f"  Overall macro_f05: {s_sep['macro_f05']:.5f} (baseline was 0.97495)")
    for c, g in pe_sep.groupby("country"):
        sg = summarize(g)
        print(f"  {c}: macro_f05={sg['macro_f05']:.5f}, P={sg['micro_precision']:.4f}, R={sg['micro_recall']:.4f}")

if __name__ == "__main__":
    main()
