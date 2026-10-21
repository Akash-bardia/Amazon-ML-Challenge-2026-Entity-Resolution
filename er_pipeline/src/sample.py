"""Stage `make_sample`: write a small copy of the dataset for smoke tests.

Train: a fraction of S1 entities, ALL their true matches, plus the same fraction of
other S2/S3 records as distractors. Test: the same fraction of every file.
"""
from __future__ import annotations

import os
import zlib

from .utils import LOG


def _keep(eid: str, frac: float) -> bool:
    return zlib.crc32(eid.encode()) / 2**32 < frac


def _filter_file(src: str, dst: str, pred) -> int:
    n = 0
    with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8", newline="") as fo:
        fo.write(fi.readline())
        for line in fi:
            if pred(line.split("\t", 1)[0]):
                fo.write(line)
                n += 1
    return n


def run(src_dir: str, dst_dir: str, frac: float) -> None:
    for split in ("train", "test"):
        os.makedirs(os.path.join(dst_dir, split), exist_ok=True)
    gt_src = os.path.join(src_dir, "train", "train_ground_truth.tsv")
    targets, s1_keep = set(), set()
    with open(gt_src, encoding="utf-8") as fi, \
            open(os.path.join(dst_dir, "train", "train_ground_truth.tsv"), "w", encoding="utf-8", newline="") as fo:
        fo.write(fi.readline())
        for line in fi:
            sid, ids = line.rstrip("\r\n").split("\t") if "\t" in line else (line.strip(), "")
            if _keep(sid, frac):
                s1_keep.add(sid)
                targets.update(i for i in ids.split(",") if i)
                fo.write(line)
    n = _filter_file(os.path.join(src_dir, "train", "train_source1.tsv"),
                     os.path.join(dst_dir, "train", "train_source1.tsv"), lambda i: i in s1_keep)
    LOG.info(f"sample train S1: {n:,}")
    for k in (2, 3):
        n = _filter_file(os.path.join(src_dir, "train", f"train_source{k}.tsv"),
                         os.path.join(dst_dir, "train", f"train_source{k}.tsv"),
                         lambda i: i in targets or _keep(i, frac))
        LOG.info(f"sample train S{k}: {n:,}")
    for k in (1, 2, 3):
        n = _filter_file(os.path.join(src_dir, "test", f"test_source{k}.tsv"),
                         os.path.join(dst_dir, "test", f"test_source{k}.tsv"), lambda i: _keep(i, frac))
        LOG.info(f"sample test S{k}: {n:,}")
