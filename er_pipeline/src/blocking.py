"""Stage `block`: candidate generation per split and country.

Channels (union):
  D1  dense k-NN on "name | address"            (k_dense_full)
  D2  dense k-NN on address only                 (k_dense_addr, records with an address)
  K   exact keys: compact name / alias / website,  house-number+city, house-number+street word
Then every union pair gets cos_full / cos_addr, "competition" statistics are computed
over ALL Source-1 queries of the country, and the union is pre-ranked down to
k_final candidates per S1. That final set is what the matcher scores and what is
written to candidate_pairs.tsv.

Output: cand/{split}_{country}.parquet with q (S1 row id), t (pool row id) + columns.
Pool row id = row in concat(source2, source3) of that split.
"""
from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd

from .config import Config
from .data import load_pool, load_s1, s1_countries
from .embed import Encoder, knn, pair_cosine, pick_device
from .utils import LOG, timed

REC_COLS = ["name_n", "alt", "addr_n", "addr_empty", "keys"]


def full_text(df: pd.DataFrame) -> list[str]:
    alt = df["alt"].where(df["alt"] == "", " " + df["alt"])
    return (df["name_n"] + alt + " | " + df["addr_n"]).tolist()


def safe_name(country: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in (country or "unknown"))


# --------------------------------------------------------------------------- #
# key channel
# --------------------------------------------------------------------------- #
def explode_keys(keys: pd.Series, local_idx: np.ndarray, chunk: int = 500_000) -> pd.DataFrame:
    parts = []
    for i in range(0, len(keys), chunk):
        s = keys.iloc[i:i + chunk]
        s = pd.Series(s.values, index=local_idx[i:i + chunk])
        e = s.str.split("\x1f").explode()
        e = e[e.notna() & (e != "")]
        if e.empty:
            continue
        h = pd.util.hash_pandas_object(e, index=False).values.view(np.int64)
        typ = (e.str[0] == "n").values.astype(np.int8)          # 1 = name key, 0 = address key
        parts.append(pd.DataFrame({"h": h, "i": e.index.values.astype(np.int32), "typ": typ}))
    if not parts:
        return pd.DataFrame({"h": pd.Series(dtype=np.int64), "i": pd.Series(dtype=np.int32),
                             "typ": pd.Series(dtype=np.int8)})
    return pd.concat(parts, ignore_index=True)


def key_candidates(cfg: Config, q_keys: pd.Series, p_keys: pd.Series) -> pd.DataFrame:
    kp = explode_keys(p_keys, np.arange(len(p_keys), dtype=np.int32))
    cnt = kp.groupby("h")["i"].transform("size")
    cap = np.where(kp["typ"].values == 1, cfg.key_cap_name, cfg.key_cap_addr)
    kp = kp[cnt.values <= cap][["h", "i"]].rename(columns={"i": "t"})
    kq = explode_keys(q_keys, np.arange(len(q_keys), dtype=np.int32))[["h", "i"]].rename(columns={"i": "q"})
    out = []
    for q0 in range(0, len(q_keys), cfg.block_q_chunk):
        sub = kq[(kq["q"] >= q0) & (kq["q"] < q0 + cfg.block_q_chunk)]
        m = sub.merge(kp, on="h", how="inner")[["q", "t"]]
        if m.empty:
            continue
        g = m.groupby(["q", "t"], sort=False).size().rename("n_keys").reset_index()
        g = g.sort_values(["q", "n_keys"], ascending=[True, False])
        g = g[g.groupby("q").cumcount() < cfg.k_keys]
        out.append(g)
        del m, g
    if not out:
        return pd.DataFrame({"q": pd.Series(dtype=np.int32), "t": pd.Series(dtype=np.int32),
                             "n_keys": pd.Series(dtype=np.int16)})
    res = pd.concat(out, ignore_index=True)
    res["n_keys"] = res["n_keys"].clip(upper=100).astype(np.int16)
    return res


# --------------------------------------------------------------------------- #
# dense channel helpers
# --------------------------------------------------------------------------- #
def dense_pairs(idx: np.ndarray, score: np.ndarray, q_map: np.ndarray, t_map: np.ndarray) -> pd.DataFrame:
    nq, k = idx.shape
    q = np.repeat(q_map, k)
    t_local = idx.ravel()
    ok = t_local >= 0
    return pd.DataFrame({"q": q[ok].astype(np.int32), "t": t_map[t_local[ok]].astype(np.int32)})


def get_embeddings(cfg: Config, enc: Encoder, name: str, texts: list[str]) -> np.ndarray:
    """Encode (or reuse cached) embeddings; always returned memory-mapped to keep RAM free."""
    path = cfg.path("emb", f"{name}.npy")
    if os.path.exists(path):
        arr = np.load(path, mmap_mode="r")
        if len(arr) == len(texts):
            LOG.info(f"  reuse cached embeddings {name} ({len(arr):,})")
            return arr
        del arr
    arr = enc.encode(texts, cfg.embed_batch, desc=name)
    np.save(path, arr)
    del arr
    return np.load(path, mmap_mode="r")


# --------------------------------------------------------------------------- #
# one split x country
# --------------------------------------------------------------------------- #
def block_country(cfg: Config, enc: Encoder, device: str, split: str, country: str,
                  s1c: pd.DataFrame, poolc: pd.DataFrame, keep_q: np.ndarray | None,
                  val_q: np.ndarray | None) -> None:
    tag = f"{split}_{safe_name(country)}"
    q_ids = s1c.index.values.astype(np.int32)        # global S1 row ids
    t_ids = poolc.index.values.astype(np.int32)      # global pool row ids
    nq, npool = len(q_ids), len(t_ids)
    LOG.info(f"[{tag}] {nq:,} S1 queries vs {npool:,} pool records")
    q_local = np.arange(nq, dtype=np.int32)
    t_local = np.arange(npool, dtype=np.int32)

    # ---- D1: full text ------------------------------------------------------
    Eq_f = get_embeddings(cfg, enc, f"{tag}_full_q", full_text(s1c))
    Ep_f = get_embeddings(cfg, enc, f"{tag}_full_p", full_text(poolc))
    idx, _ = knn(Eq_f, Ep_f, cfg.k_dense_full, device, cfg.knn_q_chunk, cfg.knn_p_chunk)
    parts = [dense_pairs(idx, None, q_local, t_local).assign(src=np.int8(1))]
    del idx

    # ---- D2: address only ---------------------------------------------------
    Eq_a = Ep_a = None
    if cfg.use_addr_embedding:
        qa = np.flatnonzero(s1c["addr_empty"].values == 0)
        pa = np.flatnonzero(poolc["addr_empty"].values == 0)
        Eq_a = get_embeddings(cfg, enc, f"{tag}_addr_q", s1c["addr_n"].tolist())
        Ep_a = get_embeddings(cfg, enc, f"{tag}_addr_p", poolc["addr_n"].tolist())
        idx, _ = knn(Eq_a[qa], Ep_a[pa], cfg.k_dense_addr, device, cfg.knn_q_chunk, cfg.knn_p_chunk)
        parts.append(dense_pairs(idx, None, qa.astype(np.int32), pa.astype(np.int32)).assign(src=np.int8(2)))
        del idx

    # ---- K: exact keys ------------------------------------------------------
    kc = key_candidates(cfg, s1c["keys"].reset_index(drop=True), poolc["keys"].reset_index(drop=True))
    parts.append(kc[["q", "t"]].assign(src=np.int8(4)))

    # ---- union ----------------------------------------------------------------
    u = pd.concat(parts, ignore_index=True)
    del parts
    # each channel emits a pair at most once, so summing the channel bits == bitwise OR
    u = u.groupby(["q", "t"], sort=False)["src"].sum().reset_index()
    u["src"] = u["src"].astype(np.int8)
    u = u.merge(kc, on=["q", "t"], how="left")
    u["n_keys"] = u["n_keys"].fillna(0).astype(np.int16)
    del kc
    LOG.info(f"[{tag}] union candidates: {len(u):,} ({len(u) / max(nq, 1):.1f} per S1)")

    qa_ = u["q"].values
    ta_ = u["t"].values
    u["cos_full"] = pair_cosine(Eq_f, Ep_f, qa_, ta_)
    if Eq_a is not None:
        ca = pair_cosine(Eq_a, Ep_a, qa_, ta_)
        empty = (s1c["addr_empty"].values[qa_] == 1) | (poolc["addr_empty"].values[ta_] == 1)
        ca[empty] = np.nan
        u["cos_addr"] = ca
    else:
        u["cos_addr"] = np.float32(np.nan)
    del Eq_f, Ep_f, Eq_a, Ep_a
    gc.collect()

    # ---- competition statistics over ALL queries -----------------------------------
    u["t_best"] = u.groupby("t")["cos_full"].transform("max").astype(np.float32)
    u["t_nq"] = u.groupby("t")["q"].transform("size").astype(np.int32)
    u["q_best"] = u.groupby("q")["cos_full"].transform("max").astype(np.float32)
    u["q_nu"] = u.groupby("q")["t"].transform("size").astype(np.int16)

    # ---- pre-rank to k_final ------------------------------------------------------
    ca_f = u["cos_addr"].fillna(u["cos_full"])
    u["pre"] = ((1 - cfg.prerank_w_addr) * u["cos_full"] + cfg.prerank_w_addr * ca_f
                + cfg.key_bonus * u["n_keys"].clip(upper=3)).astype(np.float32)

    if val_q is not None:                      # union recall ceiling bookkeeping (val only)
        vmask = np.isin(q_ids[u["q"].values], val_q)
        u.loc[vmask, ["q", "t"]].assign(q=lambda d: q_ids[d["q"].values],
                                        t=lambda d: t_ids[d["t"].values]) \
            .to_parquet(cfg.path("cand", f"union_{tag}.parquet"), index=False)

    u = u.sort_values(["q", "pre"], ascending=[True, False])
    u["rank_pre"] = u.groupby("q").cumcount().astype(np.int16)
    u = u[u["rank_pre"] < cfg.k_final]
    u["rank_full"] = u.groupby("q")["cos_full"].rank(ascending=False, method="first").astype(np.int16)

    u["q"] = q_ids[u["q"].values]
    u["t"] = t_ids[u["t"].values]
    if keep_q is not None:
        u = u[np.isin(u["q"].values, keep_q)]
    u.reset_index(drop=True).to_parquet(cfg.path("cand", f"{tag}.parquet"), index=False)
    LOG.info(f"[{tag}] saved {len(u):,} final candidates")

    if not cfg.keep_embeddings:
        gc.collect()                                   # close memmaps before deleting (Windows)
        for side in ("full_q", "full_p", "addr_q", "addr_p"):
            p = cfg.path("emb", f"{tag}_{side}.npy")
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError as e:
                    LOG.warning(f"could not delete {p} ({e}); delete work/emb manually later")


def run(cfg: Config, splits=("train", "test")) -> None:
    device = pick_device(cfg.device)
    LOG.info(f"blocking device: {device}")
    enc = Encoder(cfg.embed_model, device, cfg.embed_max_len, cfg.embed_prefix)
    roles = pd.read_parquet(cfg.path("prep", "splits.parquet"))
    keep_train = roles.loc[roles["role"] != "unused", "rid"].values.astype(np.int32)
    val_q = roles.loc[roles["role"].str.startswith("val"), "rid"].values.astype(np.int32)
    for split in splits:
        for country in s1_countries(cfg, split):
            out = cfg.path("cand", f"{split}_{safe_name(country)}.parquet")
            if os.path.exists(out):
                LOG.info(f"skip existing {out}")
                continue
            with timed(f"block {split}/{country}"):
                s1c = load_s1(cfg, split, REC_COLS, country)
                poolc = load_pool(cfg, split, REC_COLS, country)
                block_country(cfg, enc, device, split, country, s1c, poolc,
                              keep_train if split == "train" else None,
                              val_q if split == "train" else None)
                del s1c, poolc
            gc.collect()
