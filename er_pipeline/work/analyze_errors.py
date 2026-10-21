import json
import numpy as np
import pandas as pd
from src.config import Config
from src.matcher import roles, truth_pairs
from src.evaluate import macro_f05, summarize, per_entity
import src.decide as D

def analyze():
    cfg = Config()
    vp = pd.read_parquet(cfg.path("model", "val_pred.parquet"))
    r = roles(cfg)
    truth = truth_pairs(cfg)
    q_rep = r.loc[r["role"] == "val_report", "rid"].values
    country_of = pd.Series(r["country"].values, index=r["rid"].values)

    import pickle
    with open(cfg.path("model", "calibrator_lgb.pkl"), "rb") as f:
        cal = pickle.load(f)
    vp["p"] = cal.predict(vp["p_raw"].values)

    rep = vp[vp["role"] == "val_report"].copy()
    truth_val = truth[np.isin(truth["q"].values, q_rep)]
    truth_set = set(zip(truth_val["q"].values, truth_val["t"].values))

    # Current baseline:
    d_rep = D.one_to_one(rep[["q", "t", "p"]], 0.02)
    pred_base = d_rep[d_rep["p"] >= 0.75]

    # Look at pairs in d_rep
    d_rep["is_true"] = [x in truth_set for x in zip(d_rep["q"].values, d_rep["t"].values)]
    
    # Distribution of p for true matches and false matches
    print("=== Probability quantiles for TRUE matches in candidate set ===")
    true_pairs = d_rep[d_rep["is_true"]]
    print(true_pairs["p"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]))

    print("\n=== Probability quantiles for FALSE candidate pairs in candidate set ===")
    false_pairs = d_rep[~d_rep["is_true"]]
    print(false_pairs["p"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.999]))

    print(f"\nTotal true matches in d_rep: {len(true_pairs)}")
    print(f"True matches with p >= 0.75: {(true_pairs['p'] >= 0.75).sum()}")
    print(f"True matches with 0.50 <= p < 0.75: {((true_pairs['p'] >= 0.50) & (true_pairs['p'] < 0.75)).sum()}")
    print(f"True matches with 0.30 <= p < 0.50: {((true_pairs['p'] >= 0.30) & (true_pairs['p'] < 0.50)).sum()}")
    print(f"False pairs with p >= 0.75: {(false_pairs['p'] >= 0.75).sum()}")
    print(f"False pairs with 0.50 <= p < 0.75: {((false_pairs['p'] >= 0.50) & (false_pairs['p'] < 0.75)).sum()}")
    print(f"False pairs with 0.60 <= p < 0.75: {((false_pairs['p'] >= 0.60) & (false_pairs['p'] < 0.75)).sum()}")

if __name__ == "__main__":
    analyze()
