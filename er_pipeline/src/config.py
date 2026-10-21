"""Pipeline configuration.

All tunable settings live in one dataclass. Defaults are sized for an 8-core PC
with 16 GB RAM and a 6 GB GPU. Override any field from the command line with
``--set name=value`` (see run_pipeline.py) or by passing a JSON file with ``--config``.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields


@dataclass
class Config:
    # ---- paths -------------------------------------------------------------
    data_dir: str = "../student_resource/dataset"
    work_dir: str = "work"
    output_dir: str = "output"
    validator_path: str = "../student_resource/utils/validate_submission.py"

    # ---- general -----------------------------------------------------------
    seed: int = 42
    n_workers: int = 6                                    # CPU processes for normalisation (safe for 16 GB RAM)
    prep_chunk_rows: int = 50_000                         # rows per normalisation task

    # ---- validation split --------------------------------------------------
    val_fraction: float = 0.05        # share of train S1 held out (split into tune/report halves)
    n_train_s1: int = 300_000         # train-fold S1 entities used to fit the matcher

    # ---- dictionary learning (local scripts -> Latin) -----------------------
    dict_min_count: int = 3
    dict_min_share: float = 0.5
    gazetteer_min_count: int = 5

    # ---- embeddings / dense blocking ---------------------------------------
    embed_model: str = "intfloat/multilingual-e5-small"
    embed_prefix: str = "query: "
    embed_max_len: int = 64
    embed_batch: int = 512
    use_addr_embedding: bool = True   # second dense channel on address text only
    device: str = "auto"              # auto | cuda | cpu
    knn_q_chunk: int = 512            # query rows per GPU matmul
    knn_p_chunk: int = 1_000_000      # pool rows resident on GPU at once
    k_dense_full: int = 30
    k_dense_addr: int = 15
    keep_embeddings: bool = False     # keep .npy embedding files after a country is done

    # ---- key blocking --------------------------------------------------------
    key_cap_name: int = 1000          # skip name keys shared by more pool records than this
    key_cap_addr: int = 500           # skip address keys shared by more pool records than this
    k_keys: int = 30                  # keep top-k key candidates per S1 (by #shared keys)
    block_q_chunk: int = 200_000      # S1 rows per key-join chunk

    # ---- candidate pre-ranking ----------------------------------------------
    k_final: int = 25                 # candidates per S1 fed to the matcher (= candidate_pairs.tsv)
    key_bonus: float = 0.03           # pre-rank bonus per shared key (max 3 keys)
    prerank_w_addr: float = 0.4       # pre-rank score = 0.6*cos_full + 0.4*cos_addr + key bonus

    # ---- features / model ---------------------------------------------------
    feat_chunk_pairs: int = 1_000_000
    lgb_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "verbose": -1,
    })
    lgb_rounds: int = 3000
    lgb_early_stop: int = 100

    # ---- decision -------------------------------------------------------------
    p_floor: float = 0.02             # pairs below this probability are never matched

    # ---- LLM judge (optional, runs on the A100) -------------------------------
    judge_model: str = "Qwen/Qwen3-1.7B"
    judge_band_low: float = 0.02      # LightGBM probability band sent to the judge
    judge_band_high: float = 0.98
    judge_n_train: int = 300_000
    judge_max_len: int = 128
    judge_batch: int = 64
    judge_lr: float = 2e-4
    judge_epochs: int = 1
    judge_lora_r: int = 16

    # ---- helpers ------------------------------------------------------------
    def path(self, *parts: str) -> str:
        p = os.path.join(self.work_dir, *parts)
        os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
        return p

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | None, overrides: list[str] | None = None) -> "Config":
        cfg = cls()
        if path:
            with open(path, encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    setattr(cfg, k, v)
        types = {f.name: f.type for f in fields(cls)}
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"--set expects name=value, got {item!r}")
            k, v = item.split("=", 1)
            if k not in types:
                raise ValueError(f"unknown config field {k!r}")
            cur = getattr(cfg, k)
            if isinstance(cur, bool):
                v = v.lower() in ("1", "true", "yes", "y")
            elif isinstance(cur, int):
                v = int(v)
            elif isinstance(cur, float):
                v = float(v)
            elif isinstance(cur, dict):
                v = json.loads(v)
            setattr(cfg, k, v)
        return cfg
