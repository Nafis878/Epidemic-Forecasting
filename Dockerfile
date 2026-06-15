# Reproducible environment for the epidemic-forecasting benchmark.
# Pinned to the versions in requirements.txt (Python 3.10).
#
#   docker build -t epi-bench .
#   docker run --rm epi-bench                      # runs the quick pipeline + tests
#   docker run --rm epi-bench python scripts/reproduce.py --full   # paper numbers
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /work

# Install dependencies first for layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Default: validate the pipeline end-to-end (quick profile) and run the tests.
CMD ["bash", "-lc", "python scripts/reproduce.py --quick --n-boot 200 && python -m pytest -q"]
