"""One-command Colab runner for the genuine-vintage RAF benchmark.

Gives a "clone the repo and run one script" experience **without ever putting the
Delphi API key in the repo**. The key is read at runtime from, in order:

  1. the ``DELPHI_API_KEY`` environment variable, or
  2. Colab's Secrets store (``google.colab.userdata`` — the key icon in the left
     sidebar), which persists across sessions and is never committed.

It never reads the key from source, so nothing secret is pushed to GitHub.

One-time Colab setup
--------------------
  1. Runtime -> Change runtime type -> GPU.
  2. Click the key icon (Secrets) in the left sidebar; add a secret named
     ``DELPHI_API_KEY`` with your registered key and enable notebook access.
     (Register a free key at
     https://api.delphi.cmu.edu/epidata/admin/registration_form)
  3. Then, in a cell::

        !git clone https://github.com/Nafis878/Epidemic-Forecasting.git
        %cd Epidemic-Forecasting
        !pip install -q -r requirements.txt
        !python scripts/colab_run.py

Steps run: ingest genuine multi-disease vintages -> verify they are genuine ->
full genuine benchmark + RAF verdict. Read ``results/raf_verdict.json`` at the end.
"""

from __future__ import annotations

import os
import subprocess
import sys

REGISTER_URL = "https://api.delphi.cmu.edu/epidata/admin/registration_form"


def load_api_key() -> str | None:
    """Return the Delphi key from the env var or Colab Secrets; never from source."""
    key = os.environ.get("DELPHI_API_KEY")
    if key:
        return key
    try:  # Colab Secrets (google.colab.userdata) — persists, never committed.
        from google.colab import userdata  # type: ignore

        key = userdata.get("DELPHI_API_KEY")
        if key:
            os.environ["DELPHI_API_KEY"] = key
            return key
    except Exception:
        pass
    return None


def main() -> int:
    key = load_api_key()
    if not key:
        print(
            "ERROR: DELPHI_API_KEY not found.\n"
            "  - In Colab: add it via the Secrets panel (key icon) as 'DELPHI_API_KEY'\n"
            "    and enable notebook access, then re-run.\n"
            "  - Elsewhere: export DELPHI_API_KEY=<your-key> before running.\n"
            f"  - No key yet? Register a free one at {REGISTER_URL}",
            file=sys.stderr,
        )
        return 2

    py = sys.executable
    steps = [
        [py, "-m", "ingestion.vintage_delphi", "--time-values", "202001-202622"],
        [py, "evaluation/vintage_report.py"],
        [py, "scripts/reproduce.py", "--full", "--genuine"],
    ]
    for cmd in steps:
        print(f"\n>>> {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)  # env carries DELPHI_API_KEY to children

    print("\n[colab_run] complete. Verdict -> results/raf_verdict.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
