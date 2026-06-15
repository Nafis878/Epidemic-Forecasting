# Training on Google Colab Pro (GPU)

The heavy step in this benchmark is fitting the neural base models (MIST, TFT,
PatchTST) across **flu + COVID-19 + RSV** and the two genuine-vintage seasons.
Run that on a Colab Pro **GPU** runtime with `scripts/colab_train.py`; then bring
the resulting `quantiles_long_genuine.parquet` back and run the cheap
post-processing (ensembles, hybrid, bootstrap, paper numbers) locally.

> Everything genuine-vintage is sourced live from the Delphi Epidata **NHSN
> weekly** signals. A free **`DELPHI_API_KEY` is effectively required**: the
> ingestion pulls the full revision history in chunks (~150+ small calls), and
> the *keyless* endpoint throttles so hard it takes hours and silently drops
> chunks. With a key it is fast and complete. Register a free key in seconds at
> <https://api.delphi.cmu.edu/epidata/admin/registration_form>.

## 0. Runtime
`Runtime → Change runtime type → GPU` (T4/L4/A100 all work; Pro gives the better ones).

## 1. Get the code onto Colab
Because this branch has local changes, **zip your repo folder and upload it**
(simplest). Locally, zip the project directory, then in a Colab cell:

```python
from google.colab import files
import zipfile, os
up = files.upload()                       # choose Epidemic-Forecasting.zip
name = next(iter(up))
zipfile.ZipFile(name).extractall('/content/repo')
# cd into the folder that contains requirements.txt
root = next(d for d, _, fs in os.walk('/content/repo') if 'requirements.txt' in fs)
os.chdir(root); print('cwd =', os.getcwd())
```

*(Alternatively, if you have pushed this branch to GitHub:
`!git clone -b <branch> https://github.com/<you>/Epidemic-Forecasting.git && cd Epidemic-Forecasting`.)*

## 2. Install dependencies (keep Colab's CUDA-enabled torch)
Install everything **except** torch, so Colab's preinstalled GPU torch is kept:

```python
!grep -v '^torch' requirements.txt > /tmp/reqs.txt && pip install -q -r /tmp/reqs.txt
import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())
```

## 3. API key (REQUIRED — paste your free Delphi key)
```python
import os
os.environ['DELPHI_API_KEY'] = 'PASTE_YOUR_KEY'   # https://api.delphi.cmu.edu/epidata/admin/registration_form
assert os.environ['DELPHI_API_KEY'], "Get a free key first — keyless ingestion is throttled to hours."
```

## 4. Train on the GPU (ingest → prove vintage → train → zip)
```python
!python scripts/colab_train.py            # add --skip-ingest to reuse an uploaded store
```
This runs, in order: genuine ingestion (full week range), the vintage-authenticity
report (must print `genuine_vintage = True`), GPU training + the per-quantile dump
to `results/quantiles_long_genuine.parquet` (with a `disease` column), and zips the
artifacts to `results/genuine_artifacts.zip`.

Smoke-test the whole path quickly first (NOT reportable numbers):
```python
!python scripts/colab_train.py --profile quick
```

## 5. Download the artifacts
```python
from google.colab import files
files.download('results/genuine_artifacts.zip')
```
*(Or persist to Drive: `from google.colab import drive; drive.mount('/content/drive')`
then copy the zip there.)*

## 6. Back on your machine
Unzip into `results/`, then run the cheap, CPU-only post-processing:

```bash
unzip -o genuine_artifacts.zip -d results/
# (the genuine, multi-disease aggregation / hybrid / bootstrap / paper numbers
#  run on quantiles_long_genuine.parquet — minutes, no GPU)
```

## Expected time
- Ingestion: a few minutes (≈156 light API calls with backoff).
- Training: the GPU-bound part — roughly tens of minutes for `--full` across 3
  diseases × 2 seasons × the full panel (far faster than CPU). `--quick` is ~1–2 min.
- Post-processing (local): minutes.

## Notes
- Only `--profile full` artifacts are reportable; `--quick` validates the pipeline.
- The genuine seasons (2024-25, 2025-26) and per-disease signals live in
  `configs/experiment.yaml` under the `genuine:` block — edit there, not in code.
