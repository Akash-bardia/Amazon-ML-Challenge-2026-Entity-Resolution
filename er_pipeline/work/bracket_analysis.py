import pandas as pd
import numpy as np
import pickle
from src.config import Config
from src.matcher import roles, truth_pairs
import src.decide as D

def test():
    cfg = Config()
    vp = pd.read_parquet(cfg.path("model", "val_pred.parquet"))
    r = roles(cfg)
    truth = truth_pairs(cfg)
    q_tune = r.loc[r["role"] == "val_tune", "rid"].values
    
    with open(cfg.path("model", "calibrator_lgb.pkl"), "rb") as f:
        cal = pickle.load(f)
    vp["p"] = cal.predict(vp["p_raw"].values)
    
    tune = vp[vp["role"] == "val_tune"].copy()
    truth_tune = truth[np.isin(truth["q"].values, q_tune)]
    truth_qs = set(truth_tune["q"].values)
    
    d_tune = D.one_to_one(tune[["q", "t", "p"]], 0.02)
    max_p = d_tune.groupby("q")["p"].max().reindex(pd.Index(q_tune, name="q"), fill_value=0.0)
    
    df_ent = pd.DataFrame({"max_p": max_p.values, "has_truth": [q in truth_qs for q in q_tune]}, index=q_tune)
    
    # Inspect entities by max_p bracket
    brackets = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.75), (0.75, 1.01)]
    for low, high in brackets:
        m = (df_ent["max_p"] >= low) & (df_ent["max_p"] < high)
        n = m.sum()
        n_single = (~df_ent.loc[m, "has_truth"]).sum()
        pct_single = (n_single / n * 100) if n > 0 else 0
        print(f"max_p in [{low:.2f}, {high:.2f}): total={n:5d}, singletons={n_single:5d} ({pct_single:.1f}%), non-singletons={n - n_single:5d}")

if __name__ == "__main__":
    test()
