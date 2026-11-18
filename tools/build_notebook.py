"""Combine the four analysis scripts into ONE notebook: CW1_analysis.ipynb.

Each script is a list of '# %%' cells. This tool turns them into notebook cells:
  '# %% [markdown]'  -> markdown cell (leading '# ' removed)
  '# %% <title>'     -> code cell (title kept as the first comment line)
Run from the repo root:  python tools/build_notebook.py
The notebook is a convenience copy; the scripts remain the source of truth.
"""
import json
import re
from pathlib import Path

SCRIPTS = [
    ("01_task1_data_features.py", "Task 1: Data preparation and feature engineering"),
    ("02_task2_clustering.py", "Task 2: Stock segmentation and sensitivity analysis"),
    ("03_task3_review.py", "Task 3.2: Evidence for the business review"),
    ("04_build_datamart.py", "Task 3.1: Data mart"),
]

INTRO = """# S&P 500 share segmentation: risk-return vs adapted RFM
6BUIS017W CRM & CM with Business Intelligence, individual coursework 1

**How to use this notebook**
- Run the cells in order. Parts: Task 1 (data and features), Task 2 (clustering), Task 3.2 (review evidence), Task 3.1 (data mart).
- The first run needs internet (Wikipedia and Yahoo Finance). To rerun offline from the saved snapshot in `data/raw/`, set `CW1_USE_CACHE=1` before running the Task 1 cells.
- In Colab: clone/upload the repository first so `datamart/schema.sql` and `data/` are available, then run the setup cell below.
- Tunable choices (windows, seed, k range, k-selection rule) are in the CONFIG cell of each part."""

SETUP = '''# Setup: install packages if running in Google Colab (locally, use: pip install -r requirements.txt)
import sys, subprocess
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yfinance", "pyarrow", "lxml", "html5lib"], check=True)'''


import itertools
_ids = itertools.count()


def cell(kind, text):
    lines = text.rstrip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    base = {"cell_type": kind, "metadata": {}, "source": src, "id": f"c{next(_ids):03d}"}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def split_script(path):
    """Yield (kind, text) cells from a '# %%' formatted script."""
    kind, buf = None, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"# %%(.*)$", line)
        if m:
            if kind and "".join(buf).strip():
                yield kind, buf
            tag = m.group(1).strip()
            kind, buf = ("markdown", []) if tag == "[markdown]" else ("code", [f"# {tag}" if tag else ""])
            if kind == "code" and not tag:
                buf = []
        else:
            buf.append(line)
    if kind and "".join(buf).strip():
        yield kind, buf


cells = [cell("markdown", INTRO), cell("code", SETUP)]
for path, title in SCRIPTS:
    cells.append(cell("markdown", f"---\n# {title}\n`source: {path}`"))
    for kind, lines in split_script(path):
        if kind == "markdown":
            text = "\n".join(re.sub(r"^# ?", "", l) for l in lines)
        else:
            text = "\n".join(lines)
        cells.append(cell(kind, text.strip("\n")))

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}}
Path("CW1_analysis.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote CW1_analysis.ipynb with {len(cells)} cells")
