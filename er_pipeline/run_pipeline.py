"""Business Entity Resolution pipeline - single entry point.

Examples
  python run_pipeline.py --stage all
  python run_pipeline.py --stage prep,block
  python run_pipeline.py --stage block --set embed_batch=256 --set knn_q_chunk=256
  python run_pipeline.py --stage make_sample --sample-frac 0.01 --sample-dir sample_data
  python run_pipeline.py --stage judge_train,judge_score        (on the GPU machine)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Ensure all caching and temp storage stays strictly on the D drive
_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache"))
os.environ.setdefault("HF_HOME", os.path.join(_CACHE_DIR, "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(_CACHE_DIR, "torch"))
os.environ.setdefault("UV_CACHE_DIR", os.path.join(_CACHE_DIR, "uv"))
os.environ.setdefault("TEMP", os.path.join(_CACHE_DIR, "tmp"))
os.environ.setdefault("TMP", os.path.join(_CACHE_DIR, "tmp"))

MAIN_STAGES = ["prep", "block", "features", "train", "validate", "infer", "package"]
ALL_STAGES = ["make_sample"] + MAIN_STAGES + ["judge_export", "judge_train", "judge_score", "judge_combine"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    help="comma-separated stages, or 'all' = " + ",".join(MAIN_STAGES) +
                         "; available: " + ",".join(ALL_STAGES))
    ap.add_argument("--config", default=None, help="optional JSON file with config overrides")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="override one config field")
    ap.add_argument("--team", default="team", help="team name used for the submission zip")
    ap.add_argument("--sample-frac", type=float, default=0.01, help="make_sample: fraction of entities to keep")
    ap.add_argument("--sample-dir", default="sample_data", help="make_sample: output dataset directory")
    args = ap.parse_args()

    from src.config import Config
    from src.utils import LOG, setup_logging, timed

    cfg = Config.load(args.config, args.set)
    os.makedirs(cfg.work_dir, exist_ok=True)
    setup_logging(os.path.join(cfg.work_dir, "logs", f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    stages = MAIN_STAGES if args.stage == "all" else [s.strip() for s in args.stage.split(",") if s.strip()]
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        ap.error(f"unknown stage(s): {bad}")
    cfg.save(os.path.join(cfg.work_dir, "config_used.json"))
    LOG.info(f"stages: {stages}")
    LOG.info(f"data_dir={cfg.data_dir}  work_dir={cfg.work_dir}  output_dir={cfg.output_dir}")

    for st in stages:
        with timed(f"STAGE {st}"):
            if st == "make_sample":
                from src import sample
                sample.run(cfg.data_dir, args.sample_dir, args.sample_frac)
            elif st == "prep":
                from src import prep
                prep.run(cfg)
            elif st == "block":
                from src import blocking
                blocking.run(cfg)
            elif st == "features":
                from src import matcher
                matcher.run_features(cfg)
            elif st == "train":
                from src import matcher
                matcher.run_train(cfg)
            elif st == "validate":
                from src import matcher
                matcher.run_validate(cfg)
            elif st == "infer":
                from src import infer
                infer.run(cfg)
            elif st == "package":
                from src import package
                package.run(cfg, args.team)
            elif st == "judge_export":
                from src import judge
                judge.run_export(cfg)
            elif st == "judge_train":
                from src import judge
                judge.run_train(cfg)
            elif st == "judge_score":
                from src import judge
                judge.run_score(cfg)
            elif st == "judge_combine":
                from src import judge
                judge.run_combine(cfg)


if __name__ == "__main__":
    main()
