import itertools
import numpy as np
import pandas as pd
import pickle
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

    with open(cfg.path("model", "calibrator_lgb.pkl"), "rb") as f:
        cal = pickle.load(f)
    vp["p"] = cal.predict(vp["p_raw"].values)

    tune = vp[vp["role"] == "val_tune"].copy()
    rep = vp[vp["role"] == "val_report"].copy()

    d_tune = D.one_to_one(tune[["q", "t", "p"]], 0.02)
    d_rep = D.one_to_one(rep[["q", "t", "p"]], 0.02)

    # Baseline on val_report:
    pred_base = d_rep[d_rep["p"] >= 0.75]
    base_score = macro_f05(pred_base, truth, q_rep)
    print(f"Current Baseline val_report macro_f05: {base_score:.5f}")

    # Function to apply two-tier rule:
    # An entity matches if its max p >= thr_main.
    # For entities that qualify, keep all its candidates with p >= thr_sub.
    def apply_two_tier(df, thr_main, thr_sub):
        g = df.groupby("q")["p"].max()
        qualifying_qs = g[g >= thr_main].index
        d_qual = df[df["q"].isin(qualifying_qs)]
        return d_qual[d_qual["p"] >= thr_sub]

    print("\n--- Grid Search on val_tune ---")
    best_params = None
    best_tune_sc = 0.0

    for thr_main in [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80]:
        for thr_sub in [0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70, thr_main]:
            if thr_sub > thr_main:
                continue
            pred_tune = apply_two_tier(d_tune, thr_main, thr_sub)
            sc = macro_f05(pred_tune, truth, q_tune)
            if sc > best_tune_sc:
                best_tune_sc = sc
                best_params = (thr_main, thr_sub)
            if round(thr_sub, 2) in [0.55, 0.60, 0.65, thr_main] and round(thr_main, 2) in [0.72, 0.75]:
                print(f"  tune (thr_main={thr_main:.2f}, thr_sub={thr_sub:.2f}) -> {sc:.5f}")

    print(f"\nBest val_tune params: thr_main={best_params[0]:.2f}, thr_sub={best_params[1]:.2f} -> tune score: {best_tune_sc:.5f}")

    # Evaluate honestly on val_report:
    pred_rep = apply_two_tier(d_rep, best_params[0], best_params[1])
    pe_rep = per_entity(pred_rep, truth, q_rep)
    pe_rep["country"] = country_of.reindex(pe_rep.index).values
    s_rep = summarize(pe_rep)
    print(f"\nHonest val_report with two-tier decision:")
    print(f"  Overall macro_f05: {s_rep['macro_f05']:.5f} (vs baseline {base_score:.5f})")
    print(f"  Micro Precision: {s_rep['micro_precision']*100:.2f}%, Micro Recall: {s_rep['micro_recall']*100:.2f}%")
    print(f"  False Positives: {s_rep['false_positives']}, False Negatives: {s_rep['false_negatives']}")
    print(f"  Singleton Accuracy: {s_rep['singleton_accuracy']*100:.2f}%")
    for c, g in pe_rep.groupby("country"):
        sg = summarize(g)
        print(f"  {c}: macro_f05={sg['macro_f05']:.5f}, P={sg['micro_precision']*100:.2f}%, R={sg['micro_recall']*100:.2f}%")

if __name__ == "__main__":
    main()
