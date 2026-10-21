import os
import json
import itertools
import pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

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

    tune_mask = vp["role"] == "val_tune"
    rep_mask = vp["role"] == "val_report"

    vp_tune = vp[tune_mask].copy()
    vp_rep = vp[rep_mask].copy()

    # 1. Global calibration
    cal_global = D.fit_calibrator(vp_tune["p_raw"].values, vp_tune["label"].values)
    vp_tune["p"] = cal_global.predict(vp_tune["p_raw"].values)
    vp_rep["p"] = cal_global.predict(vp_rep["p_raw"].values)

    print("=== 1. Fine-grained Global Threshold Scan ===")
    d_tune = D.one_to_one(vp_tune[["q", "t", "p"]], 0.02)
    d_rep = D.one_to_one(vp_rep[["q", "t", "p"]], 0.02)

    best_thr = 0.75
    best_tune_sc = 0.0
    for thr in np.round(np.arange(0.60, 0.85, 0.01), 3):
        pred = d_tune[d_tune["p"] >= thr]
        sc = macro_f05(pred, truth, q_tune)
        if sc > best_tune_sc:
            best_tune_sc = sc
            best_thr = thr
        if round(thr, 2) in [0.65, 0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80]:
            print(f"  tune thr {thr:.2f} -> {sc:.5f}")

    print(f"Best tune global thr: {best_thr} -> {best_tune_sc:.5f}")
    pred_rep = d_rep[d_rep["p"] >= best_thr]
    s_rep = summarize(per_entity(pred_rep, truth, q_rep))
    print(f"Honest val_report for thr {best_thr}: macro_f05={s_rep['macro_f05']:.5f}, P={s_rep['micro_precision']:.4f}, R={s_rep['micro_recall']:.4f}")

    print("\n=== 2. Per-Country Thresholds ===")
    # Split tune by country
    for ctry in ["India", "US"]:
        q_tune_c = r.loc[(r["role"] == "val_tune") & (r["country"] == ctry), "rid"].values
        vp_tune_c = vp_tune[vp_tune["country"] == ctry]
        d_tune_c = D.one_to_one(vp_tune_c[["q", "t", "p"]], 0.02)

        best_c_thr = 0.75
        best_c_sc = 0.0
        for thr in np.round(np.arange(0.55, 0.88, 0.01), 3):
            pred = d_tune_c[d_tune_c["p"] >= thr]
            sc = macro_f05(pred, truth, q_tune_c)
            if sc > best_c_sc:
                best_c_sc = sc
                best_c_thr = thr
        print(f"  Country {ctry}: best tune thr {best_c_thr:.3f} -> tune score {best_c_sc:.5f}")

    print("\n=== 3. Per-Country Calibration + Per-Country Thresholds ===")
    vp_tune_sep = vp_tune.copy()
    vp_rep_sep = vp_rep.copy()
    cals = {}
    for ctry in ["India", "US"]:
        m_t = vp_tune["country"] == ctry
        m_r = vp_rep["country"] == ctry
        cal_c = D.fit_calibrator(vp_tune.loc[m_t, "p_raw"].values, vp_tune.loc[m_t, "label"].values)
        cals[ctry] = cal_c
        vp_tune_sep.loc[m_t, "p"] = cal_c.predict(vp_tune.loc[m_t, "p_raw"].values)
        vp_rep_sep.loc[m_r, "p"] = cal_c.predict(vp_rep.loc[m_r, "p_raw"].values)

    best_thrs = {}
    for ctry in ["India", "US"]:
        q_tune_c = r.loc[(r["role"] == "val_tune") & (r["country"] == ctry), "rid"].values
        vp_tune_c = vp_tune_sep[vp_tune_sep["country"] == ctry]
        d_tune_c = D.one_to_one(vp_tune_c[["q", "t", "p"]], 0.02)
        best_c_thr = 0.75
        best_c_sc = 0.0
        for thr in np.round(np.arange(0.50, 0.88, 0.01), 3):
            pred = d_tune_c[d_tune_c["p"] >= thr]
            sc = macro_f05(pred, truth, q_tune_c)
            if sc > best_c_sc:
                best_c_sc = sc
                best_c_thr = thr
        best_thrs[ctry] = best_c_thr
        print(f"  Per-country cal {ctry}: best tune thr {best_c_thr:.3f} -> score {best_c_sc:.5f}")

    # Evaluate combined on val_report:
    preds_rep = []
    for ctry in ["India", "US"]:
        m_r = vp_rep_sep["country"] == ctry
        d_r = D.one_to_one(vp_rep_sep.loc[m_r, ["q", "t", "p"]], 0.02)
        preds_rep.append(d_r[d_r["p"] >= best_thrs[ctry]])
    combined_rep = pd.concat(preds_rep, ignore_index=True)
    pe_sep = per_entity(combined_rep, truth, q_rep)
    pe_sep["country"] = country_of.reindex(pe_sep.index).values
    s_sep = summarize(pe_sep)
    print(f"\nHonest val_report with Per-Country Calibration + Per-Country Thrs:")
    print(f"  Overall macro_f05: {s_sep['macro_f05']:.5f} (vs baseline 0.97495)")
    for c, g in pe_sep.groupby("country"):
        sg = summarize(g)
        print(f"  {c}: macro_f05={sg['macro_f05']:.5f}, P={sg['micro_precision']:.4f}, R={sg['micro_recall']:.4f}")

if __name__ == "__main__":
    main()
