# PROFIT: current model and portable training

This repository contains the PROFIT model core and a portable version of its training entrypoint. The core model and inference files are byte-identical to the supplied `PROFIT_method_core` export; their SHA-256 values are recorded in `PROVENANCE.json`.

PROFIT forecasts every observed process and quality channel. It applies RevIN, selects phase-correlated leader histories, aligns them within the observed window, decomposes them into fixed DCT bands, encodes patches separately per band, and combines same-band leader information. The final model also supports the summary residual readout and centered low-rank channel adapter. `models/PAFNet.py` is the core; `models/PROFITAux.py` provides the evaluated auxiliary readouts. Input and output shapes are `[batch, context, channels]` and `[batch, horizon, channels]`.

## Installation and implementation checks

Clone the repository and run from its root directory so the `models` and `layers` imports resolve to this implementation:

```powershell
git clone https://github.com/StevenLuan904/PROFIT.git
Set-Location .\PROFIT
python -m pip install -r requirements.txt
python tests/test_method_core.py
```

Use Python 3.9 or later. Requirements describe direct dependencies, not a complete environment lock. Install the Torch build appropriate for your CPU or CUDA system. The implementation checks use synthetic inputs and do not need a dataset or checkpoint. They cover phase alignment, DCT reconstruction, patch/readout behavior, TRAIN-only thresholds and equivalence of quality-query execution.

## Data contract

No datasets or trained weights are included in this repository. Supply chronologically ordered CSV files in `data/raw/`, or set `data_root` in a configuration file. The first column must be `date`; every remaining column must be a finite numeric channel, in the original benchmark order. The loader excludes `date`, fits StandardScaler on TRAIN rows only, retains all forecast origins and tail batches, and reports equally weighted MSE/MAE in training-standardized units.

| Configuration | File | Numeric channels | TRAIN / validation / test rows |
|---|---|---:|---:|
| `configs/mp.json` | `MP_new_date.csv` | 11 | 3063 / 438 / 875 |
| `configs/production.json` | `Production_small.csv` | 18 | 6935 / 1000 / 2000 |
| `configs/mining.json` | `Mining_small.csv` | 8 | 7000 / 1000 / 2000 |

The Production profile expects the 9,935-row causal cohort. Its quality value at a process timestamp comes from the latest earlier laboratory sample, with the unavailable prefix removed. Obtain and prepare benchmark data separately before using a named profile.

The three configurations retain the supplied H=24 model/training settings and use a 360-sample context. Change `horizon` and `seed` for additional runs. Preserve the dataset row boundaries and original channel order. Default device is CUDA; for a CPU execution check, copy a config and change `device` to `cpu`.

## Train, select and evaluate

Use fresh output paths for every run:

```powershell
python scripts/train.py smoke --config configs/mp.json --output-dir runs/mp_smoke
python scripts/train.py fit --config configs/mp.json --output-dir runs/mp_fit
python scripts/train.py freeze --runs runs/mp_fit --output runs/mp_selection.json
python scripts/train.py evaluate --selection runs/mp_selection.json
```

`smoke` executes one optimization step and checks validation prediction shapes. `fit` selects a checkpoint by validation MSE and computes no held-out test scores. `freeze` selects the minimum-validation-MSE run among candidates with the same model, dataset, horizon, seed and data identity. `evaluate` consumes that frozen selection and writes a held-out evaluation once. Multiple seeds/horizons require separate run and selection paths. The trainer refuses to overwrite completed or uncertain artifacts; it does not implement checkpoint resume.

For MP, the fixed mixed MSE-Huber thresholds are calibrated once from the actual standardized TRAIN rows. The original FP64 prefix-sum and exact equal-horizon weighted-MAD rule over horizons 12, 24, 36 and 48 is preserved. Validation/test values do not determine these thresholds. Production and Mining use MSE. The centered-adapter penalty and optimizer grouping retain the supplied implementation. Physical microbatch size is explicit because BatchNorm statistics depend on it.

The training data/window/scaler code lives in `data_provider/data_loader.py`; it is the canonical `prepare()` implementation moved out of the launcher without changing its arithmetic. The registry supports only the bundled PROFIT/PAFNet model family; baseline implementations are not bundled in this repository.

## Quality-query execution

`quality_query_execution.py` predicts designated output channels while retaining the full candidate-sensor universe. `quality_row_execution.py` reuses standardization of overlapping observed rows for consecutive requests with common frozen weights and an already fitted TRAIN scaler. It does not cache FFT, DCT or attention outputs across different forecast windows. The synthetic implementation checks compare both paths against the full forward pass.

## Scope and provenance

This release provides executable source for fresh runs. It does not redistribute raw data, historical checkpoints, server manifests, paper files or result tables, and does not claim that implementation checks reproduce the published forecasting scores. The supplied model core is unchanged. Portability changes are limited to local data paths, module packaging, core-only registry branches and rebuilding the same fixed TRAIN thresholds instead of reading private campaign artifacts.

`LICENSE` is retained verbatim from the accompanying source snapshot, including its original copyright notice, for the inherited source. It is not a new licensing decision for newly added packaging files. Third-party baseline source and its licenses are outside this independent release.
