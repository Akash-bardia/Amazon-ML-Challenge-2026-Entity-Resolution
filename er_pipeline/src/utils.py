"""Shared helpers: logging, timing, TSV/ground-truth IO, stable hashing."""
from __future__ import annotations

import csv
import hashlib
import logging
import os
import sys
import time
import zlib
from contextlib import contextmanager

import numpy as np
import pandas as pd

LOG = logging.getLogger("er")


def setup_logging(log_path: str | None = None) -> None:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    LOG.addHandler(h)
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        LOG.addHandler(fh)


@contextmanager
def timed(what: str):
    t0 = time.time()
    LOG.info(f"START {what}")
    yield
    LOG.info(f"DONE  {what} in {time.time() - t0:,.1f}s")


def read_source_tsv(path: str, usecols=None) -> pd.DataFrame:
    """Read a source TSV exactly as-is: tab separated, no quoting, empty -> ''."""
    csv.field_size_limit(10**8)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                     usecols=usecols, encoding="utf-8", engine="c")
    return df


def read_ground_truth(path: str) -> pd.DataFrame:
    """Return long-format truth pairs: columns q_id, t_id (singletons have no rows)."""
    gt = read_source_tsv(path)
    gt.columns = ["q_id", "ids"]
    gt = gt[gt["ids"] != ""]
    out = gt.assign(t_id=gt["ids"].str.split(",")).explode("t_id")[["q_id", "t_id"]]
    out = out[out["t_id"] != ""]
    return out.reset_index(drop=True)


def stable_hash01(ids: pd.Series, seed: int) -> np.ndarray:
    """Deterministic pseudo-random number in [0,1) per id (independent of Python hash seed).

    BLAKE2b, not CRC32: CRC is linear, so differently-salted CRCs are correlated.
    """
    salt = str(seed).encode()
    return np.fromiter(
        (int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8, salt=salt[:16]).digest(), "big") / 2**64
         for s in ids), dtype=np.float64, count=len(ids))


def write_id_list_tsv(path: str, s1_ids, mapping: dict, col: str) -> None:
    """Write matching_results.tsv / candidate_pairs.tsv. mapping: S1 id -> 'id1,id2' string."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        f.writelines(f"{sid}\t{mapping.get(sid, '')}\n" for sid in s1_ids)


def countries_in(df: pd.DataFrame) -> list[str]:
    return sorted(df["country"].unique().tolist())
