# HA-ASIDE — ICGHIT 2027 research code

**HA-ASIDE: Role-Conditioned Embedding Rotation for Instruction–Data Separation**  
Phan Cong Danh and Wooguil Pak  
School of Computer Science and Engineering, Yeungnam University

This repository provides the source, exact data selections, training recipe,
inference implementation, and recorded generated tokens used by the six-page
manuscript in [`paper/main.pdf`](paper/main.pdf). The paper evaluates a
single-turn feasibility configuration. Multi-turn agent evaluation and utility
optimization remain future work. Conference acceptance is not claimed.

There are three reproducibility paths:

1. **Reanalyze recorded experiments on CPU:** decode saved token IDs and recompute
   Tables II/III, encoding diagnostics, and the paired confidence interval.
2. **Train HA-ASIDE from the pinned base model:** reconstruct byte-identical
   ChatQA training/validation files and run the reported three-epoch configuration.
3. **Run fresh experiments:** use a trained HA-ASIDE checkpoint and the exact
   published baseline weights on the same frozen inputs and metrics.

The HA-ASIDE 7B weights are not included or hosted by this repository. Fresh
HA-ASIDE inference requires training first. Baseline weights are downloaded
from the original authors. Their winning training recipes are not fully exposed
by those checkpoints; we do not invent baseline hyperparameters or present a
newly trained baseline as the exact published model.

## Setup

Run commands from the repository root. Linux with Python 3.10/3.11 is the intended
GPU environment. CPU reanalysis also works on Windows.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU reanalysis and smoke tests:
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/runtime.txt
```

For GPU training, install the recorded PyTorch CUDA build instead of the CPU
wheel, then install DeepSpeed:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/training.txt
```

The historical inference environment recorded Python 3.10.13, PyTorch
2.5.1+cu121, Transformers 5.17.0, NumPy 2.2.6, and an RTX 6000 Ada GPU.
See `reproducibility/environment.json`. Not every historical training dependency
was recorded. Release dependency pins make setup explicit; they do not establish
bitwise identity to an undocumented historical environment. The release's actual
CPU checks are recorded separately in `reproducibility/release_verification.json`.

Read [source and model terms](DATA_AND_MODEL_SOURCES.md). Generated data, caches,
weights, and experiment outputs are ignored by Git.

## Recompute the reported results

```bash
python scripts/prepare_evaluation.py
python scripts/reproduce_results.py
python scripts/plot_results.py
```

Preparation downloads immutable upstream sources and reconstructs 1,690 selected
inputs: 800 archived role variants and 890 clean utility controls. Every case ID
and content hash must match `reproducibility/evaluation_cases.json`; no generated
question or reference answer is substituted. The prepared texts stay in `work/`.

Reanalysis uses the recorded token IDs, the fixed form-aware matcher, the original
IHEval scoring functions, and BIPIA QA token F1. It checks all 8,360 archived
outputs (4,800 separation/intervention outputs and 3,560 utility outputs).
The answer is cut before the first `endoftext` or `im_end` token for every model.
Figures, CSV tables, and verification reports are written under
`outputs/reproduced/`. No model weights or API judge are needed for this path.

Expected paper scores, on a 0–100 scale:

| Method | Paired separation | Language accuracy | Translation ROUGE-L F1 | Verb F1 | QA F1 |
|---|---:|---:|---:|---:|---:|
| Vanilla | 4.0 | 83.75 | 51.56 | 30.42 | 5.62 |
| ISE-2 | 2.5 | 47.92 | 43.02 | 38.67 | 7.23 |
| ASIDE | 21.0 | 57.50 | 52.86 | 36.06 | 3.91 |
| HA-ASIDE | 55.5 | 21.25 | 27.58 | 8.81 | 1.21 |

The HA-ASIDE–ASIDE paired-separation difference is 34.5 percentage points. A
1,000-resample paired percentile bootstrap (seed 42) over 50 question families
gives the unrounded 95% interval `[25.9875, 43.0]` percentage points, reported
as `[26.0, 43.0]` in the manuscript. Reanalysis is not fresh model inference.

## Reproduce HA-ASIDE training

```bash
python scripts/prepare_training.py
python scripts/train_paper.py                    # print the exact command
python scripts/train_paper.py --verify-data      # historical trainer data/config dry run
python scripts/train_paper.py --run              # GPU training
```

Preparation downloads ChatQA revision `164f018...`, verifies its checksum, and
reconstructs **32,046 training / 3,552 validation** conversations from the frozen
source-row index. Both output files must match the original full-file SHA-256
values. Documents, first questions, connected-group assignments, exclusions,
and token counts are preserved. `semantic_screening.json` documents the limited
screening decisions; it is not a full-corpus semantic certification.

The launcher uses the **historical training implementation**, restored byte for
byte from its deployment archive and checked against the reported checkpoint's
source hashes. It is kept under `reproducibility/historical_training/` because
the general training code subsequently gained additional features. Training
uses the pinned Qwen2.5-7B base and Instruct tokenizer, full-sequence next-token
loss, no truncation beyond the prepared 2,048-token bound, AdamW, cosine decay,
learning rate `1e-6`, zero warmup and weight decay, effective batch 64, microbatch
1, accumulation 64, three epochs, and seed 42. RGAG is disabled. Role angles are
SYSTEM=0, USER=π/8, DATA=π/2; other input components retain identity.

The original run used one RTX 6000 Ada 48 GB GPU with gradient checkpointing and
DeepSpeed ZeRO-2 CPU optimizer offload. The exact offload config is in `configs/`.
The historical CLI flag `--single` selects the first configuration for **three
full epochs**; it does not impose a step limit. Only `--smoke` adds a two-step
limit and changes the effective batch to 2. A smoke run is not paper training:

```bash
python scripts/train_paper.py --smoke --run --output outputs/training-smoke
```

Checkpoints are saved after each epoch. The reported model is the third-epoch
checkpoint at step 1503, not a completed hyperparameter search winner. Use
`outputs/training/lr1e-06_warm0_batch64/checkpoint-1503` for the corresponding
training run. Model-only checkpoint saving does not support resuming optimizer
state. New training output directories must be empty; no existing checkpoint
is silently overwritten. Hardware and numerical-library differences can change
newly trained weights and downstream scores; recorded-output checks remain the
exact numerical reference.

## Run fresh paper experiments

```bash
python scripts/download_baselines.py             # inspect pinned downloads
python scripts/download_baselines.py --download

python scripts/run_experiments.py --method vanilla --checkpoint checkpoints/baselines/vanilla --run
python scripts/run_experiments.py --method ise --checkpoint checkpoints/baselines/ise --run
python scripts/run_experiments.py --method aside_orig --checkpoint checkpoints/baselines/aside_orig --run
python scripts/run_experiments.py --method ha_aside --checkpoint outputs/training/lr1e-06_warm0_batch64/checkpoint-1503 --run

python scripts/reproduce_results.py --inference-dir outputs/inference --output outputs/fresh-results
python scripts/plot_results.py --analysis outputs/fresh-results
```

Omit `--run` for input and checkpoint validation without loading model weights.
Use `--smoke` with a separate `--output` directory for a short inference check.
The result analyzer refuses incomplete cohorts. Each method is loaded in its
own process. Resume is permitted only when the stored plan and checkpoint
metadata agree. Inputs are never truncated to fit the context budget.

**Counts:** 50 underlying probe questions × two clause forms × two contexts =
200 contexts. A paired USER–DATA score uses two responses per context, hence 400
responses per model. The paper additionally includes 200 descriptive SYSTEM
responses and 400 encoding-only interventions: **1,000 separation generations
plus 890 utility generations per method**. The archived output bundle also
contains 200 auxiliary history-placement responses per method; they are not
included in the paper tables. `--include-history-diagnostic` regenerates that
archived auxiliary condition explicitly. Bootstrap resampling uses the 50
question families, not 200 independent questions or 800 independent contexts.

Fresh generation calls the same shared `generate_transcript` implementation as
the archived run, with greedy decoding, fixed reference history, 16,384 total
context budget, 1,024 new tokens for separation/QA, and 2,048 for IHEval controls.
All models use identical inputs and response boundaries. Source-relabeling
interventions require application control of role IDs; they are diagnostics of
encoding sensitivity. Low clean-task utility is preserved in the results.

## Tests and paper build

```bash
python scripts/smoke_training.py
python -m unittest discover -s tests -v
python scripts/verify_release.py
# With Tectonic installed:
cd paper && tectonic main.tex
```

The CPU smoke test trains a tiny random model using the historical training
modules. It verifies gradients and fixed role rotations, not 7B effectiveness.
The PDF's figures are already embedded and need no font installation to compile.
`plot_results.py` uses Times New Roman when installed, with a serif fallback for
portable replotting. The supplied final PDF retains its venue fonts.
