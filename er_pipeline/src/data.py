"""Loading normalised records per split / country with global row ids.

Global ids:
  S1   : rid = row number in {split}_source1
  pool : pid = row number in source2, then len(source2) + row number in source3
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import Config
from .prep import prep_path


def pool_offset(cfg: Config, split: str) -> int:
    return pq.ParquetFile(prep_path(cfg, split, 2)).metadata.num_rows


def s1_countries(cfg: Config, split: str) -> list[str]:
    return sorted(pd.read_parquet(prep_path(cfg, split, 1), columns=["country"])["country"].unique())


def _read(path: str, cols: list[str], country: str | None) -> pd.DataFrame:
    filters = [("country", "==", country)] if country is not None else None
    cols = list(dict.fromkeys(["rid", "country"] + cols))
    df = pd.read_parquet(path, columns=cols, filters=filters)
    for c in df.columns:                     # plain object strings -> numpy object arrays everywhere
        if pd.api.types.is_string_dtype(df[c]) and df[c].dtype != object:
            df[c] = df[c].astype(object)
    return df


def load_s1(cfg: Config, split: str, cols: list[str], country: str | None = None) -> pd.DataFrame:
    df = _read(prep_path(cfg, split, 1), cols, country)
    return df.set_index(df["rid"].astype(np.int64)).drop(columns="rid")


def load_pool(cfg: Config, split: str, cols: list[str], country: str | None = None) -> pd.DataFrame:
    off = pool_offset(cfg, split)
    a = _read(prep_path(cfg, split, 2), cols, country)
    b = _read(prep_path(cfg, split, 3), cols, country)
    a.index = a["rid"].astype(np.int64)
    b.index = b["rid"].astype(np.int64) + off
    return pd.concat([a, b]).drop(columns="rid")
