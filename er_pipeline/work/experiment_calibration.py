import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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

    tune_m = vp["role"] == "val_tune"
    rep_m = vp["role"] == "val_report"

    # Calibration 1: Current Isotonic
    cal_iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    cal_iso.fit(vp.loc[tune_m, "p_raw"].values, vp.loc[tune_m, "label"].values)

    # Calibration 2: Platt scaling (Logistic Regression on logit(p_raw))
    eps = 1e-7
    p_clipped = np.clip(vp["p_raw"].values, eps, 1 - eps)
    logit = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)
    lr = LogisticRegression(C=1.0)
    lr.fit(logit[tune_m], vp.loc[tune_m, "label"].values)

    # Test raw, isotonic, and Platt scaling
    for name, p_all in [
        ("Raw p_raw", vp["p_raw"].values),
        ("Isotonic", cal_iso.predict(vp["p_raw"].values)),
        ("Platt / Logit", lr.predict_proba(logit)[:, 1])
    ]:
        vp_temp = vp[["q", "t", "role"]].copy()
        vp_temp["p"] = p_all

        d_tune = D.one_to_one(vp_temp.loc[tune_m, ["q", "t", "p"]], 0.02)
        d_rep = D.one_to_one(vp_temp.loc[rep_m, ["q", "t", "p"]], 0.02)

        best_thr = 0.5
        best_sc = 0.0
        for thr in np.round(np.arange(0.2, 0.96, 0.02), 2):
            sc = macro_f05(d_tune[d_tune["p"] >= thr], truth, q_tune)
            if sc > best_sc:
                best_sc = sc
                best_thr = thr

        rep_sc = macro_f05(d_rep[d_rep["p"] >= best_thr], truth, q_rep)
        print(f"Method [{name:15s}]: best tune thr {best_thr:.2f} (tune {best_sc:.5f}) -> honest rep: {rep_sc:.5f}")

if __name__ == "__main__":
    main()
