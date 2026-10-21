"""Optional LLM judge (Qwen3-1.7B + LoRA, sequence classification).

judge_export  (PC)   : write text pairs for training / val / test uncertain band -> work/judge/*.parquet
judge_train   (A100) : fine-tune LoRA classifier on work/judge/train.parquet -> work/judge/model/
judge_score   (A100) : score work/judge/{val,test}.parquet -> work/judge/{val,test}_scores.parquet
judge_combine (PC)   : stack LightGBM + judge on val_tune, report on val_report, and write
                       submission v2 ONLY if the val_report score improves.
Only the work/judge folder has to be copied between the machines.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time

import numpy as np
import pandas as pd

from .config import Config
from .utils import LOG, timed

PROMPT = "Do these two records describe the same business?\nA: {a}\nB: {b}\nAnswer:"


def _text(name: pd.Series, addr: pd.Series, country: pd.Series) -> pd.Series:
    return name.fillna("") + " | " + addr.fillna("") + " | " + country.fillna("")


# --------------------------------------------------------------------------- #
# PC side
# --------------------------------------------------------------------------- #
def _attach_text(cfg: Config, split: str, pairs: pd.DataFrame) -> pd.DataFrame:
    from .data import load_pool, load_s1
    cols = ["name_raw", "addr_raw"]
    s1 = load_s1(cfg, split, cols)
    pool = load_pool(cfg, split, cols)
    a = s1.reindex(pairs["q"].values)
    b = pool.reindex(pairs["t"].values)
    pairs = pairs.reset_index(drop=True)
    pairs["text_a"] = _text(a["name_raw"], a["addr_raw"], a["country"]).values
    pairs["text_b"] = _text(b["name_raw"], b["addr_raw"], b["country"]).values
    return pairs


def run_export(cfg: Config) -> None:
    import lightgbm as lgb
    from .matcher import load_feats
    lo, hi = cfg.judge_band_low, cfg.judge_band_high
    model = lgb.Booster(model_file=cfg.path("model", "lgb.txt"))
    with open(cfg.path("model", "features.json")) as f:
        feat_names = json.load(f)

    tr = load_feats(cfg, "train")
    tr["p_raw"] = model.predict(tr[feat_names].astype(np.float32), num_iteration=model.best_iteration)
    tr = tr[["q", "t", "label", "p_raw"]]
    band = tr[(tr["p_raw"] >= lo) & (tr["p_raw"] <= hi)]
    rest = tr.drop(band.index)
    n_band = min(len(band), int(cfg.judge_n_train * 0.7))
    n_rest = min(len(rest), cfg.judge_n_train - n_band)
    sel = pd.concat([band.sample(n_band, random_state=cfg.seed), rest.sample(n_rest, random_state=cfg.seed)])
    sel = _attach_text(cfg, "train", sel[["q", "t", "label"]].sample(frac=1.0, random_state=cfg.seed))
    sel.to_parquet(cfg.path("judge", "train.parquet"), index=False)
    LOG.info(f"judge train pairs: {len(sel):,} (positives {int(sel['label'].sum()):,})")
    del tr

    vp = pd.read_parquet(cfg.path("model", "val_pred.parquet"))
    vb = vp[(vp["p_raw"] >= lo) & (vp["p_raw"] <= hi)][["q", "t", "label", "role"]]
    _attach_text(cfg, "train", vb).to_parquet(cfg.path("judge", "val.parquet"), index=False)
    tp = pd.read_parquet(cfg.path("model", "test_pred.parquet"))
    tb = tp[(tp["p_raw"] >= lo) & (tp["p_raw"] <= hi)][["q", "t"]]
    _attach_text(cfg, "test", tb).to_parquet(cfg.path("judge", "test.parquet"), index=False)
    LOG.info(f"judge band pairs: val {len(vb):,} / {len(vp):,}, test {len(tb):,} / {len(tp):,}")
    LOG.info(f"copy the folder {cfg.path('judge')} to the GPU machine (same relative path under work_dir)")


def run_combine(cfg: Config) -> None:
    from sklearn.linear_model import LogisticRegression
    from .infer import run_validator, write_outputs
    from .matcher import run_validate

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    vs = pd.read_parquet(cfg.path("judge", "val_scores.parquet"))
    vp = pd.read_parquet(cfg.path("model", "val_pred.parquet")).merge(vs, on=["q", "t"], how="left")
    inb = vp["judge_logit"].notna()
    tune = inb & (vp["role"] == "val_tune")
    X = np.c_[logit(vp.loc[tune, "p_raw"].values), vp.loc[tune, "judge_logit"].values]
    lr = LogisticRegression(C=1.0).fit(X, vp.loc[tune, "label"].values)
    LOG.info(f"stacker weights lgb={lr.coef_[0][0]:.3f} judge={lr.coef_[0][1]:.3f} b={lr.intercept_[0]:.3f}")
    vp["p_comb"] = vp["p_raw"]
    Xa = np.c_[logit(vp.loc[inb, "p_raw"].values), vp.loc[inb, "judge_logit"].values]
    vp.loc[inb, "p_comb"] = lr.predict_proba(Xa)[:, 1]
    vp.to_parquet(cfg.path("model", "val_pred_judge.parquet"), index=False)

    base = json.load(open(cfg.path("reports", "validation_lgb.json")))["overall"]["macro_f05"]
    rep = run_validate(cfg, "val_pred_judge.parquet", "p_comb", "judge")
    new = rep["overall"]["macro_f05"]
    LOG.info(f"val_report macro F0.5: LightGBM {base:.5f} -> with judge {new:.5f}")
    if new <= base:
        LOG.info("judge does not improve validation; keeping submission v1")
        return

    ts = pd.read_parquet(cfg.path("judge", "test_scores.parquet"))
    tp = pd.read_parquet(cfg.path("model", "test_pred.parquet")).merge(ts, on=["q", "t"], how="left")
    tinb = tp["judge_logit"].notna()
    tp["p_comb"] = tp["p_raw"]
    Xt = np.c_[logit(tp.loc[tinb, "p_raw"].values), tp.loc[tinb, "judge_logit"].values]
    tp.loc[tinb, "p_comb"] = lr.predict_proba(Xt)[:, 1]
    for fn in ("matching_results.tsv", "candidate_pairs.tsv"):
        src = os.path.join(cfg.output_dir, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(cfg.output_dir, "v1_" + fn))
    write_outputs(cfg, tp, "p_comb", "judge")
    run_validator(cfg)
    LOG.info("submission v2 (with judge) written; v1 kept as output/v1_*.tsv")


# --------------------------------------------------------------------------- #
# GPU side (only needs torch / transformers / peft / pandas / pyarrow)
# --------------------------------------------------------------------------- #
def _load_model(cfg: Config, train: bool):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.judge_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(cfg.judge_model, num_labels=1, torch_dtype=dtype)
    model.config.pad_token_id = tok.pad_token_id
    return tok, model


def _batches(df: pd.DataFrame, bs: int, shuffle: bool, seed: int):
    idx = np.arange(len(df))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


def _encode(tok, df, idx, max_len, device):
    texts = [PROMPT.format(a=a, b=b) for a, b in zip(df["text_a"].values[idx], df["text_b"].values[idx])]
    return tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)


def run_train(cfg: Config) -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    df = pd.read_parquet(cfg.path("judge", "train.parquet"))
    tok, model = _load_model(cfg, True)
    lcfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=cfg.judge_lora_r, lora_alpha=2 * cfg.judge_lora_r,
                      lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    for n, p in model.named_parameters():          # train LoRA + head in fp32 for stability
        if p.requires_grad:
            p.data = p.data.float()
    model.to(device)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.judge_lr, weight_decay=0.0)
    steps = cfg.judge_epochs * math.ceil(len(df) / cfg.judge_batch)
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
    lossf = torch.nn.BCEWithLogitsLoss()
    labels = torch.tensor(df["label"].values, dtype=torch.float32)
    model.train()
    step, t0 = 0, time.time()
    for ep in range(cfg.judge_epochs):
        for idx in _batches(df, cfg.judge_batch, True, cfg.seed + ep):
            enc = _encode(tok, df, idx, cfg.judge_max_len, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**enc).logits.squeeze(-1).float()
            loss = lossf(logits, labels[idx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0 or step == steps:
                LOG.info(f"judge step {step}/{steps} loss {loss.item():.4f} "
                         f"({(time.time() - t0) / step:.2f}s/step)")
    out = cfg.path("judge", "model")
    model.save_pretrained(out)
    tok.save_pretrained(out)
    LOG.info(f"judge adapter saved to {out}")


def run_score(cfg: Config) -> None:
    import torch
    from peft import PeftModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, base = _load_model(cfg, False)
    model = PeftModel.from_pretrained(base, cfg.path("judge", "model")).to(device).eval()
    for part in ("val", "test"):
        path = cfg.path("judge", f"{part}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        lens = (df["text_a"].str.len() + df["text_b"].str.len()).values
        order = np.argsort(-lens)
        out = np.zeros(len(df), dtype=np.float32)
        bs = cfg.judge_batch * 4
        with timed(f"judge score {part} ({len(df):,} pairs)"), torch.inference_mode():
            for i in range(0, len(order), bs):
                idx = order[i:i + bs]
                enc = _encode(tok, df, idx, cfg.judge_max_len, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    out[idx] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
                if (i // bs) % 200 == 0:
                    LOG.info(f"  {min(i + bs, len(df)):,}/{len(df):,}")
        pd.DataFrame({"q": df["q"].values, "t": df["t"].values, "judge_logit": out}).to_parquet(
            cfg.path("judge", f"{part}_scores.parquet"), index=False)
