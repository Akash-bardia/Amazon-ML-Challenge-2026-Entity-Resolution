# Business Entity Resolution Pipeline (Amazon ML Challenge 2026)

End-to-end pipeline: **normalise → block (dense + keys) → features → LightGBM → calibrated 1-to-1
expected-F0.5 decision → submission files → package**, plus an optional **Qwen3-1.7B judge** that runs on
a separate GPU machine.

```
TSVs ─► prep ─► block ─► features ─► train ─► validate ─► infer ─► package        (RTX 4050 PC)
                                                   │
                                                   └► judge_export ─► [judge_train ─► judge_score] ─► judge_combine
                                                                        (A100 / RunPod, optional)
```

---

## 1. What each stage does

| Stage | Machine | What happens | Main outputs |
|---|---|---|---|
| `prep` | CPU | Learns a local-script→Latin word dictionary from training pairs, learns city lists from Source 1, normalises all 6 files (names, legal forms, aliases/F.K.A., websites, addresses, states, house numbers, keys). Makes the validation split: 5% of train S1 held out, split into **val_tune** / **val_report**; 300k train-fold S1 for fitting. | `work/prep/*.parquet`, `dictionary.json`, `gazetteer.json`, `splits.parquet` |
| `block` | **GPU** | Per country: multilingual-e5-small embeddings of "name \| address" and of address only, exact GPU k-NN (chunked, fits 6 GB), exact name / house-number keys; union → cosine + competition stats over **all** S1 → top-25 per S1. Validation S1 are searched against the **full** training pool (same as test). | `work/cand/{split}_{country}.parquet` |
| `features` | CPU | ~60 features per candidate pair (fuzzy name/address scores, IDF token overlap, numbers, city/state, embeddings, competition gaps, ranks). | `work/feat/{train,val}_*.parquet` |
| `train` | CPU | LightGBM on the train fold; predictions for the validation pairs. | `work/model/lgb.txt`, `val_pred.parquet` |
| `validate` | CPU | Isotonic calibration + decision tuning on **val_tune**; honest score on **val_report**; candidate recall. | `work/reports/validation_lgb.json` |
| `infer` | CPU | Same features/model/decision on test; writes both TSVs; internal checks + official validator (`--check-ids`). | `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `work/reports/test_stats_lgb.json` |
| `package` | CPU | Builds `<team>_submission.zip` (output/, code/, filled Documentation_template.md). | `work/submission/<team>_submission.zip` |
| `judge_*` | A100 | Optional Qwen3-1.7B LoRA judge for uncertain pairs; used only if it improves val_report. | `output/` v2 (v1 kept as `output/v1_*.tsv`) |

---

## 2. Prerequisites

**Main PC (RTX 4050 laptop):**
- Windows, NVIDIA driver with CUDA ≥ 13.0 (yours shows 13.3 ✔), 16 GB RAM, **≥ 40 GB free disk**.
- **Python 3.12** (the system Python 3.14 is too new for some wheels; we create a separate 3.12 env).
- Internet on first run (downloads `intfloat/multilingual-e5-small`, ~470 MB). No business data is looked up anywhere.
- Plugged in, sleep disabled during long stages.

**Optional judge machine (RunPod A100):** RunPod PyTorch template, container disk 30 GB, volume 20 GB at `/workspace`.

---

## 3. Setup (one time, PowerShell, inside this folder)

```powershell
pip install uv
python -m uv venv --python 3.12 .venv
.venv\Scripts\activate
python -m uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
python -m uv pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The last line must print `True` and the RTX 4050 name.

**Data location.** By default the pipeline expects this folder next to the dataset:
```
amazon_ml_challenge/
├── er_pipeline/                          <- this folder (run commands here)
└── 6ab10eb3b23ba_student_resource/student_resource/dataset/{train,test}/*.tsv
```
If your data is elsewhere add `--set data_dir=D:/path/to/dataset` (folder that contains `train/` and `test/`)
and `--set validator_path=D:/path/to/utils/validate_submission.py` to every command.

---

## 4. Run

### 4.1 Dry run on 1% of the data first (≈10–15 min, catches environment problems)
```powershell
python run_pipeline.py --stage make_sample --sample-frac 0.01 --sample-dir sample_data
python run_pipeline.py --stage all --set data_dir=sample_data --set work_dir=work_sample --set output_dir=output_sample --set val_fraction=0.2
```
It must print the validator's `PASS` and finish with `DONE STAGE package`. Scores on the sample are
meaningless (the sample pool is tiny) — it only proves the environment and code work. Delete
`sample_data/`, `work_sample/`, `output_sample/` afterwards.

### 4.2 Full run
```powershell
python run_pipeline.py --stage prep
python run_pipeline.py --stage block
python run_pipeline.py --stage features,train,validate
python run_pipeline.py --stage infer
python run_pipeline.py --stage package --team YOUR_TEAM_NAME
```
(or `python run_pipeline.py --stage all --team YOUR_TEAM_NAME`). Expected times on the RTX 4050 PC:

| Stage | Time (approx.) |
|---|---|
| prep | 30–40 min |
| block | 1.5–2.5 h |
| features + train + validate | 45–75 min |
| infer + package | 20–40 min |

Every stage saves its outputs; if something stops, re-run the same command — finished countries/files are
skipped. To force a stage to redo its work, delete its folder under `work/` (e.g. `work/cand`).

### 4.3 What to check before submitting
1. `work/reports/validation_lgb.json` → `overall.macro_f05` (the honest estimate of the leaderboard),
   `candidate_recall_topk`, and `by_country`.
2. `work/reports/test_stats_lgb.json` → predicted singleton rate and matches per S1 per country should look
   similar to the validation ones (`singleton_rate_pred`, `mean_pred_per_s1`). France is the one to watch.
3. The log ends with the official validator printing **PASS**.
4. Upload `output/matching_results.tsv` to the portal.

---

## 5. Optional: Qwen3-1.7B judge on RunPod A100 (~45–60 min of GPU)

1. **PC:** `python run_pipeline.py --stage judge_export` → creates `work/judge/{train,val,test}.parquet` (~0.5–1 GB).
2. Copy `er_pipeline/` **code only** (`run_pipeline.py`, `src/`, `requirements-judge.txt`) and the folder
   `work/judge/` to the pod under `/workspace/er_pipeline/` (keep the `work/judge/` path).
3. **Pod:**
   ```bash
   cd /workspace/er_pipeline
   pip install -r requirements-judge.txt
   python run_pipeline.py --stage judge_train,judge_score --set judge_batch=64
   ```
   Produces `work/judge/model/`, `work/judge/val_scores.parquet`, `work/judge/test_scores.parquet`.
   **Stop the pod right after** (billing is per second).
4. Copy `work/judge/val_scores.parquet` and `work/judge/test_scores.parquet` back into `work/judge/` on the PC.
5. **PC:** `python run_pipeline.py --stage judge_combine` → writes submission v2 **only if** val_report improves
   (v1 is kept as `output/v1_*.tsv`), then `python run_pipeline.py --stage package --team YOUR_TEAM_NAME`.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` in `block` | `--set embed_batch=256 --set knn_q_chunk=256 --set knn_p_chunk=500000` |
| RAM runs out in `block` / `features` | `--set block_q_chunk=50000 --set feat_chunk_pairs=500000`; close other apps |
| `block` too slow | `--set use_addr_embedding=false` (drops the address-only dense channel, ~40% faster) |
| prep uses too much RAM | `--set n_workers=6` |
| Validator not found | `--set validator_path=PATH/TO/validate_submission.py` |
| Judge OOM on A100 40 GB | `--set judge_batch=32` |

All settings live in `src/config.py`; any field can be overridden with `--set name=value`. The config
actually used is saved to `work/config_used.json`, logs to `work/logs/`.

## 7. Models and licences
- `intfloat/multilingual-e5-small` (118M, MIT) — embeddings for blocking.
- LightGBM (MIT) — matcher.
- `Qwen/Qwen3-1.7B` (1.7B, Apache-2.0) — optional judge.
All under 8B parameters and MIT/Apache-2.0, as the rules require. No external data or lookups are used.
