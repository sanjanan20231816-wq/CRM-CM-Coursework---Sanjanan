# S&P 500 share segmentation: risk-return vs adapted RFM

6BUIS017W CRM & CM with Business Intelligence, individual coursework 1.
Two segmentations of S&P 500 shares (weekly return + weekly beta vs adapted Recency/Frequency/Monetary
trading-activity features), compared with Ward and K-means clustering.

## Pipeline (run in this order)
| Script | Brief task | What it does | Main outputs |
|---|---|---|---|
| `01_task1_data_features.py` | Task 1 | Downloads and validates data, weekly return/beta, R/F/M | `data/processed/feature_table.csv`, `tables/`, worked-example figure |
| `02_task2_clustering.py` | Task 2 | Feature prep, Ward + K-means (k = 2..8), profiles, correlations, Monetary removal | `data/processed/cluster_assignments.csv`, `feature_table_with_clusters.csv`, `figures/f2_*`, `tables/t2_*` |
| `03_task3_review.py` | Task 3.2 | Sector mix, risk-return vs RFM agreement, return dependence, adjustment and survivorship checks | `figures/f3_1`-`f3_3`, `tables/t3_*` |
| `04_build_datamart.py` | Task 3.1 | Builds the star-schema SQLite data mart from `datamart/schema.sql`, draws the diagram | `figures/f3_4_star_schema.png`, `tables/t3_datamart_*` |

**Single notebook:** `CW1_analysis.ipynb` contains all four parts in order (41 code cells, outputs and figures
included). It is generated from the scripts by `python tools/build_notebook.py`; the scripts are the source of truth.
The saved outputs come from a run using the cached snapshot (`CW1_USE_CACHE=1`), and reproduce the script outputs byte for byte.

Every script is a sequence of `# %%` cells: open in Colab/VS Code/Jupyter, or run with `python <script>`.
Each cell has a header comment saying which part of the brief it answers. Tunable choices
(windows, seed, k range, k-selection rule) sit in the CONFIG cell at the top of each script.

## How to run
```bash
pip install -r requirements.txt
python 01_task1_data_features.py       # needs internet the first time (Wikipedia + Yahoo Finance)
python 02_task2_clustering.py
python 03_task3_review.py
python 04_build_datamart.py
```
Offline rerun from the saved snapshot: set the environment variable `CW1_USE_CACHE=1` before script 01.
In Colab use `%run 01_task1_data_features.py` etc. after uploading/cloning the repo.

## Key parameters
Study window 2021-01-01 to 2025-01-01 (end exclusive), baseline 20 sessions, window 60 sessions,
volume multiplier 2, seed 42 (K-means `n_init=50`), k range 2-8.
k-selection rule (script 02 CONFIG): highest silhouette with k >= 3 and smallest cluster >= 5% of shares.

## Folder layout
- `data/raw/` constituent list, daily prices (parquet), benchmark, retrieval metadata (the preserved snapshot)
- `data/processed/` weekly returns, event flags, feature table, cluster assignments, exclusion log, run metadata, results JSON
- `data/datamart/` SQLite mart (built by script 04, not committed: large)
- `datamart/schema.sql` star-schema DDL
- `tables/`, `figures/` outputs cited in the report
- `report/` Word template and structure notes

## Provenance of the reported run
- Constituent list retrieved: see `data/raw/retrieval_metadata.json` (2026-11-17 UTC)
- Observation date: 2024-12-31 (see `data/processed/run_metadata.json`)
- Python 3.12.10, pandas 3.1.1, numpy 2.5.5, yfinance 1.8.0, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2
  (full list: `tables/package_versions.txt`)
- Public GitHub address: ____

## Status
- [x] Task 1 script (run)
- [x] Task 2 script (run)
- [x] Task 3 scripts (run)
- [ ] Report written by the student
