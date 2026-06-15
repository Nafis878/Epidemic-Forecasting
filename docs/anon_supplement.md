# Anonymised Supplement — Packaging Instructions

This note explains how to produce a **double-blind-safe** code-and-data archive for submission.
The manuscript (`paper/mist_neurips.tex`) is already anonymised: `\author{Anonymous}` and no
author-identifying URLs appear in it (the reproducibility statement points reviewers to this
supplement, not to a public repository).

The repository, however, still contains author-identifying strings outside the paper. Strip them
before zipping. Do **not** commit the anonymised zip into the public repo; build it on the side.

## 1. Identifying strings to remove or replace

| File | Line (approx.) | Content | Action |
|---|---|---|---|
| `LICENSE` | 3 | `Copyright (c) 2025 Nafis878` | Replace name with `Anonymous Authors` (or an MIT "the authors" form) |
| `README.md` | ~170 | BibTeX `author = {Nafis878}` | Replace with `author = {Anonymous}`; drop any repo URL |
| `CHANGES.md` | ~181 | `https://github.com/Nafis878/Epidemic-Forecasting` | Remove the URL |
| `paper/mist_neurips.tex` | — | already `\author{Anonymous}`, no URL | none |

Search the whole tree before zipping (case-insensitive):

```bash
grep -rniE "nafis|github\.com/Nafis|your-name-here" . --exclude-dir=.git
```

The command should return **nothing** after edits.

## 2. Strip version-control and local-identity metadata

`.git/` history embeds author name and email in every commit and must not ship.

```bash
# Work on a copy, never the live repo.
cp -r "Epidemic Forecasting" anon_pkg && cd anon_pkg
rm -rf .git .gitignore                 # drop history + any local ignores
rm -rf **/__pycache__ .pytest_cache    # caches can hold absolute user paths
rm -f results/run_profile.txt          # ephemeral marker, not a result
```

Also scrub absolute paths that leak a username (e.g. `C:\Users\<name>\...`): these appear only in
transient logs, not in tracked source, but grep to be sure:

```bash
grep -rniE "users[\\/][^\\/]+" . --exclude-dir=.git | grep -vi anonymous
```

## 3. What to include

Include everything needed to reproduce the numbers:

- `models/`, `evaluation/`, `features/`, `ingestion/`, `configs/`, `scripts/`, `tests/`
- `paper/` (the `.tex`, `macros.tex`, `numbers.json`, auto-generated tables)
- `results/` and `tables/` (committed artifacts the gate checks)
- `data/store/*.parquet` (the versioned store) **if** licensing permits redistribution; otherwise
  include `ingestion/` and document how to rebuild the store from public sources.
- `requirements.txt`, `README.md` (anonymised), `docs/`

Exclude: `.git/`, caches, any private credentials, and `figures/` source PSDs if any.

## 4. Verify the package reproduces and stays anonymous

```bash
cd anon_pkg
python -m pip install -r requirements.txt
python scripts/reproduce.py --quick           # pipeline + artifacts in ~1-2 min
python -m pytest -q                            # all tests pass
python -m evaluation.assert_reproducibility    # gate exits 0
grep -rniE "nafis|github\.com/Nafis" . --exclude-dir=.git    # must be empty
```

For the camera-ready / rebuttal, regenerate the reportable numbers with the full profile
(`python scripts/reproduce.py --full`) so `paper/tables_auto.tex` and the hybrid macros are
populated from full-run artifacts rather than the `--quick` placeholders.

## 5. Zip

```bash
cd .. && zip -r anon_supplement.zip anon_pkg -x '*/.git/*' '*/__pycache__/*'
```

Upload `anon_supplement.zip` as the supplementary material. After acceptance, restore author
identity in `LICENSE`, `README.md`, `CHANGES.md`, and `paper/mist_neurips.tex`.
