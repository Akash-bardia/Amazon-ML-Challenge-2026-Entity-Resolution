"""Stage `prep`: learn dictionaries, build gazetteer, normalise all 6 files, make splits.

Outputs (under work_dir):
  prep/dictionary.json        learned non-Latin -> Latin token / component maps
  prep/gazetteer.json         per-country city list learned from Source 1
  prep/{split}_s{k}.parquet   normalised records (row order == file order)
  prep/splits.parquet         train S1 roles: val_tune / val_report / train / unused
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import normalize as N
from .config import Config
from .utils import LOG, read_ground_truth, read_source_tsv, stable_hash01, timed

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)


def src_path(cfg: Config, split: str, k: int) -> str:
    return os.path.join(cfg.data_dir, split, f"{split}_source{k}.tsv")


def prep_path(cfg: Config, split: str, k: int) -> str:
    return cfg.path("prep", f"{split}_s{k}.parquet")


# --------------------------------------------------------------------------- #
# 1. dictionary learned from aligned training pairs
# --------------------------------------------------------------------------- #
def learn_dictionary(cfg: Config) -> dict:
    gt = read_ground_truth(os.path.join(cfg.data_dir, "train", "train_ground_truth.tsv"))
    s1 = read_source_tsv(src_path(cfg, "train", 1)).set_index("entity_id")
    tok_counts: dict[str, Counter] = defaultdict(Counter)
    comp_counts: dict[str, Counter] = defaultdict(Counter)
    for k in (2, 3):
        df = read_source_tsv(src_path(cfg, "train", k))
        mask = df["business_name"].map(N.has_non_latin) | df["business_address"].map(N.has_non_latin)
        df = df[mask]
        df = df.merge(gt, left_on="entity_id", right_on="t_id", how="inner")
        df = df.join(s1[["business_name", "business_address"]], on="q_id", rsuffix="_s1")
        for tn, ta, sn, sa in zip(df["business_name"], df["business_address"],
                                  df["business_name_s1"], df["business_address_s1"]):
            # names: align token-by-token when the token counts agree
            if N.has_non_latin(tn):
                tt, st = N.raw_tokens(tn), N.raw_tokens(sn)
                if len(tt) == len(st):
                    for a, b in zip(tt, st):
                        if N.has_non_latin(a) and not N.has_non_latin(b):
                            tok_counts[a][b.lower()] += 1
            # addresses: a single non-Latin component usually is the state -> S1 last component
            if N.has_non_latin(ta):
                comps = [c.strip() for c in ta.split(",") if c.strip()]
                nl = [c for c in comps if N.has_non_latin(c)]
                s_comps = [c.strip() for c in sa.split(",") if c.strip()]
                if len(nl) == 1 and s_comps and not N.has_non_latin(s_comps[-1]):
                    comp_counts[nl[0]][s_comps[-1].lower()] += 1
        del df
    def select(counts):
        out = {}
        for src, c in counts.items():
            (best, n), total = c.most_common(1)[0], sum(c.values())
            if n >= cfg.dict_min_count and n / total >= cfg.dict_min_share:
                out[src] = best
        return out
    d = {"token": select(tok_counts), "component": select(comp_counts)}
    LOG.info(f"dictionary: {len(d['token']):,} token maps, {len(d['component']):,} component maps")
    return d


# --------------------------------------------------------------------------- #
# 2. city gazetteer from Source 1 (train + test; both are provided data)
# --------------------------------------------------------------------------- #
def _run_parallel(cfg: Config, fn, rows: list, init_args) -> list:
    size = cfg.prep_chunk_rows
    chunks = [rows[i:i + size] for i in range(0, len(rows), size)]
    if cfg.n_workers <= 1 or len(chunks) == 1:
        N.init_worker(*init_args)
        return [x for ch in chunks for x in fn(ch)]
    out = []
    with ProcessPoolExecutor(max_workers=cfg.n_workers, initializer=N.init_worker,
                             initargs=init_args) as ex:
        for res in ex.map(fn, chunks):
            out.extend(res)
    return out


def build_gazetteer(cfg: Config, d: dict) -> dict:
    counts: dict[str, Counter] = defaultdict(Counter)
    for split in SPLITS:
        s1 = read_source_tsv(src_path(cfg, split, 1), usecols=["business_address", "country"])
        rows = list(zip(s1["business_address"], s1["country"]))
        comps = _run_parallel(cfg, N.components_chunk, rows, (d["token"], d["component"], {}))
        for (_, country), cl in zip(rows, comps):
            ck = N.country_key(country)
            states = N.STATE_TABLES.get(ck, {})
            for c in cl:
                c = N.CITY_PREFIX_RE.sub("", c)
                toks = c.split()
                if not (1 <= len(toks) <= 4) or any(not t.isalpha() for t in toks):
                    continue
                if c in states or any(t in N.STREET_WORDS for t in toks) or len(c) < 3:
                    continue
                counts[ck][c] += 1
    gz = {ck: sorted(c for c, n in cnt.items() if n >= cfg.gazetteer_min_count) for ck, cnt in counts.items()}
    LOG.info("gazetteer sizes: " + ", ".join(f"{k}={len(v):,}" for k, v in gz.items()))
    return gz


# --------------------------------------------------------------------------- #
# 3. normalise every file
# --------------------------------------------------------------------------- #
def _norm_block(df: pd.DataFrame, ex, size: int) -> pd.DataFrame:
    rows = list(zip(df["name_raw"], df["addr_raw"], df["country"]))
    chunks = [rows[i:i + size] for i in range(0, len(rows), size)]
    results = ex.map(N.normalize_chunk, chunks) if ex is not None else map(N.normalize_chunk, chunks)
    cols = {c: [] for c in N.NORM_COLS}
    for res in results:
        for c in N.NORM_COLS:
            cols[c].extend(res[c])
    norm = pd.DataFrame(cols)
    for c in ("script", "addr_empty"):
        norm[c] = norm[c].astype("int8")
    return pd.concat([df.reset_index(drop=True), norm], axis=1)


def normalize_file(cfg: Config, split: str, k: int, d: dict, gz: dict, block_rows: int = 1_000_000) -> None:
    """Stream the TSV in blocks, normalise in parallel, append to one parquet file."""
    out = prep_path(cfg, split, k)
    tmp = out + ".tmp"
    init_args = (d["token"], d["component"], gz)
    ex = None
    if cfg.n_workers > 1:
        ex = ProcessPoolExecutor(max_workers=cfg.n_workers, initializer=N.init_worker, initargs=init_args)
    else:
        N.init_worker(*init_args)
    writer, schema, n = None, None, 0
    try:
        reader = pd.read_csv(src_path(cfg, split, k), sep="\t", dtype=str, keep_default_na=False,
                             quoting=csv.QUOTE_NONE, encoding="utf-8", chunksize=block_rows)
        for block in reader:
            block.columns = ["id", "name_raw", "addr_raw", "country"]
            df = _norm_block(block, ex, cfg.prep_chunk_rows)
            df.insert(0, "rid", np.arange(n, n + len(df), dtype=np.int32))
            for c in df.columns:
                if df[c].dtype != object and pd.api.types.is_string_dtype(df[c]):
                    df[c] = df[c].astype(object)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp, schema)
            writer.write_table(table.cast(schema))
            n += len(df)
            LOG.info(f"  {split}_s{k}: {n:,} rows normalised")
    finally:
        if writer is not None:
            writer.close()
        if ex is not None:
            ex.shutdown()
    os.replace(tmp, out)
    LOG.info(f"{split}_s{k}: {n:,} rows -> {out}")


# --------------------------------------------------------------------------- #
# 4. validation split (train S1 roles)
# --------------------------------------------------------------------------- #
def make_splits(cfg: Config) -> None:
    s1 = pd.read_parquet(prep_path(cfg, "train", 1), columns=["id", "country"])
    h = stable_hash01(s1["id"], cfg.seed)
    role = pd.Series("unused", index=s1.index)
    val = h < cfg.val_fraction
    role[val] = "val_tune"
    role[val & (stable_hash01(s1["id"], cfg.seed + 1) < 0.5)] = "val_report"
    rest = ~val
    n_rest = int(rest.sum())
    if cfg.n_train_s1 >= n_rest:
        role[rest] = "train"
    else:
        order = pd.Series(h, index=s1.index)[rest].sort_values()
        role[order.index[: cfg.n_train_s1]] = "train"
    out = pd.DataFrame({"rid": s1.index.astype("int32"), "id": s1["id"], "country": s1["country"],
                        "role": role.values})
    out.to_parquet(cfg.path("prep", "splits.parquet"), index=False)
    LOG.info("split roles: " + out["role"].value_counts().to_dict().__repr__())


def run(cfg: Config) -> None:
    with timed("learn dictionary"):
        d = learn_dictionary(cfg)
        with open(cfg.path("prep", "dictionary.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
    with timed("build gazetteer"):
        gz = build_gazetteer(cfg, d)
        with open(cfg.path("prep", "gazetteer.json"), "w", encoding="utf-8") as f:
            json.dump(gz, f, ensure_ascii=False)
    for split in SPLITS:
        for k in SOURCES:
            with timed(f"normalise {split}_s{k}"):
                normalize_file(cfg, split, k, d, gz)
    with timed("validation split"):
        make_splits(cfg)
