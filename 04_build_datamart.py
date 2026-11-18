# %% [markdown]
# # Task 3.1: Build the data mart (star schema) in SQLite and draw its diagram
# Input : outputs of scripts 01-03 and datamart/schema.sql
# Output: data/datamart/sp500_mart.db (large, git-ignored), figures/f3_4_star_schema.png,
#         tables/t3_datamart_rowcounts.csv, tables/t3_datamart_example_query.csv
# Purpose: shows the design in the report is real, reproducible and queryable. Re-running
# creates a fresh database; in production every run would ADD a dim_run row instead.

# %% CONFIG
import json, sqlite3, subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(".")
RAW, PROC, TAB, FIG = ROOT / "data/raw", ROOT / "data/processed", ROOT / "tables", ROOT / "figures"
DB_DIR = ROOT / "data/datamart"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB = DB_DIR / "sp500_mart.db"
CALC_VERSION = "1.0"          # bump when feature definitions change
BASELINE, MULT = 20, 2.0

# %% Load inputs
meta = json.loads((PROC / "run_metadata.json").read_text())
feat = pd.read_csv(PROC / "feature_table_with_clusters.csv", index_col="ticker")
feat = feat[feat["common_set"]]
sel = json.loads((PROC / "task2_results.json").read_text())["selected_k"]
cons = pd.read_csv(RAW / "sp500_constituents.csv")
cons["ticker"] = cons["Symbol"].str.replace(".", "-", regex=False)
prices = pd.read_parquet(RAW / "prices_daily.parquet")
prices["date"] = pd.to_datetime(prices["date"])
prices = prices[prices.Ticker.isin(feat.index)]
ret = pd.read_parquet(PROC / "weekly_returns.parquet")[feat.index]
ret_m = pd.read_parquet(PROC / "weekly_returns_benchmark.parquet")["^GSPC"]
events = pd.read_parquet(PROC / "event_flags.parquet")[feat.index]
bench = pd.read_csv(RAW / "gspc_daily.csv", index_col=0, parse_dates=True)

try:
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    git_commit = None          # not inside a git repo yet

# %% Create the database from schema.sql
if DB.exists():
    DB.unlink()
con = sqlite3.connect(DB)
con.executescript((ROOT / "datamart/schema.sql").read_text())

# %% Dimensions
# dim_run: one row describing THIS calculation (sources, windows, versions, seed)
con.execute("INSERT INTO dim_run VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
    datetime.now(timezone.utc).isoformat(timespec="seconds"), CALC_VERSION, git_commit,
    "yfinance (Yahoo Finance) + Wikipedia S&P 500 list", meta["constituents_retrieved_utc"], "2021-01-01", "2025-01-01",
    meta["observation_date"], meta["baseline"], meta["window"], meta["mult"], meta["seed"],
    meta["python"], meta["pandas"], meta["yfinance"], meta["scikit-learn"]))
con.execute("INSERT INTO dim_benchmark VALUES (1,'^GSPC','S&P 500 index','price index (no dividends)')")

dates = bench.index
week_end = dates.to_series().dt.to_period("W-FRI").dt.end_time.dt.normalize()
complete_weeks = set(ret_m.index) | {ret_m.index[0] - pd.Timedelta(days=7)}   # weeks used in the return series
dim_date = pd.DataFrame({
    "date_key": dates.strftime("%Y%m%d").astype(int), "full_date": dates.strftime("%Y-%m-%d"),
    "year": dates.year, "month": dates.month, "day_of_week": dates.day_name(),
    "week_ending_friday": (dates + pd.to_timedelta((4 - dates.day_of_week) % 7, unit="D")).strftime("%Y-%m-%d"),
    "is_trading_day": 1,
    "is_complete_week": [int(pd.Timestamp(w) in complete_weeks) for w in
                         (dates + pd.to_timedelta((4 - dates.day_of_week) % 7, unit="D"))]})
dim_date.to_sql("dim_date", con, if_exists="append", index=False)

sec = feat[["sector", "sub_industry"]].drop_duplicates().reset_index(drop=True)
sec.insert(0, "sector_key", range(1, len(sec) + 1)); sec.columns = ["sector_key", "gics_sector", "gics_sub_industry"]
sec.to_sql("dim_sector", con, if_exists="append", index=False)

share = feat.reset_index()[["ticker", "company", "sector", "sub_industry"]].merge(
    cons[["ticker", "CIK", "Date added"]], on="ticker", how="left").merge(
    sec, left_on=["sector", "sub_industry"], right_on=["gics_sector", "gics_sub_industry"], how="left")
dim_share = pd.DataFrame({"share_key": range(1, len(share) + 1), "ticker": share.ticker, "company": share.company,
                          "cik": share.CIK.astype(str), "sector_key": share.sector_key,
                          "date_added_to_index": share["Date added"], "valid_from": "2021-01-01",
                          "valid_to": None, "is_current": 1})
dim_share.to_sql("dim_share", con, if_exists="append", index=False)
share_key = dict(zip(dim_share.ticker, dim_share.share_key))

# %% Facts
# fact_daily_price: prices, volume, derived traded value and the event flag with its threshold
vol = prices.pivot(index="date", columns="Ticker", values="Volume")[feat.index]
baseline = vol.shift(1).rolling(BASELINE, min_periods=BASELINE).mean()
assert ((vol > MULT * baseline) & baseline.notna()).equals(events.astype(bool)), "event flags differ from script 01"
daily = prices.copy()
daily["share_key"] = daily.Ticker.map(share_key)
daily["date_key"] = daily.date.dt.strftime("%Y%m%d").astype(int)
daily["traded_value_usd"] = daily.Close * daily.Volume
base_long = baseline.stack().rename("baseline_mean_vol_20").reset_index().rename(columns={"level_0": "date", "date": "date"})
base_long.columns = ["date", "Ticker", "baseline_mean_vol_20"]
daily = daily.merge(base_long, on=["date", "Ticker"], how="left")
daily["event_threshold"] = MULT * daily.baseline_mean_vol_20
ev_long = events.stack().rename("is_high_volume_event").reset_index()
ev_long.columns = ["date", "Ticker", "is_high_volume_event"]
daily = daily.merge(ev_long, on=["date", "Ticker"], how="left")
daily["is_high_volume_event"] = daily["is_high_volume_event"].astype(int)
daily["run_key"] = 1
cols = ["run_key", "date_key", "share_key", "Open", "High", "Low", "Close", "Adj Close", "Volume",
        "traded_value_usd", "baseline_mean_vol_20", "event_threshold", "is_high_volume_event"]
out = daily[cols]; out.columns = ["run_key", "date_key", "share_key", "open", "high", "low", "close", "adj_close",
                                  "volume", "traded_value_usd", "baseline_mean_vol_20", "event_threshold",
                                  "is_high_volume_event"]
out.to_sql("fact_daily_price", con, if_exists="append", index=False, chunksize=50000)

# fact_weekly_return
wk = ret.stack().rename("weekly_return").reset_index()
wk.columns = ["week_end", "Ticker", "weekly_return"]
wk["week_end_date_key"] = wk.week_end.dt.strftime("%Y%m%d").astype(int)
wk["share_key"] = wk.Ticker.map(share_key)
wk["benchmark_weekly_return"] = wk.week_end.map(ret_m)
wk = wk.assign(run_key=1, benchmark_key=1)[["run_key", "week_end_date_key", "share_key", "benchmark_key",
                                            "weekly_return", "benchmark_weekly_return"]]
# week-ending Fridays can be holidays and so are not in dim_date: store them anyway (dates are in the key)
wk.to_sql("fact_weekly_return", con, if_exists="append", index=False)

# fact_share_feature
f = feat.reset_index()
obs_key = int(meta["observation_date"].replace("-", ""))
ft = pd.DataFrame({"run_key": 1, "observation_date_key": obs_key, "share_key": f.ticker.map(share_key),
                   "benchmark_key": 1, "mean_weekly_return": f.mean_weekly_return, "weekly_beta": f.beta,
                   "beta_std_error": f.beta_se, "n_weeks": f.n_weeks, "recency_days": f.R, "frequency_events": f.F,
                   "monetary_usd": f.M_usd, "last_event_date": f.last_event_date})
ft.to_sql("fact_share_feature", con, if_exists="append", index=False)

# fact_cluster_assignment: all five solutions (four baselines + R,F-only sensitivity run)
rows = []
spec = {"A_ward": ("A_risk_return", "ward"), "A_kmeans": ("A_risk_return", "kmeans"),
        "B_ward": ("B_rfm", "ward"), "B_kmeans": ("B_rfm", "kmeans"), "B_ward_RF_only": ("B_rfm_RF_only", "ward")}
for col, (matrix, algo) in spec.items():
    k = sel["B_ward"]["k"] if col == "B_ward_RF_only" else sel[col]["k"]
    rows.append(pd.DataFrame({"run_key": 1, "share_key": f.ticker.map(share_key), "feature_matrix": matrix,
                              "algorithm": algo, "k": k, "cluster_id": f[col].values}))
pd.concat(rows).to_sql("fact_cluster_assignment", con, if_exists="append", index=False)
con.commit()

# %% Verify: row counts and a stakeholder-style query
counts = pd.DataFrame([(t, con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in
                       ["dim_run", "dim_date", "dim_sector", "dim_share", "dim_benchmark", "fact_daily_price",
                        "fact_weekly_return", "fact_share_feature", "fact_cluster_assignment"]],
                      columns=["table", "rows"])
counts.to_csv(TAB / "t3_datamart_rowcounts.csv", index=False)
print(counts.to_string(index=False))
assert counts.set_index("table").loc["fact_daily_price", "rows"] == len(feat) * len(bench)
assert counts.set_index("table").loc["fact_weekly_return", "rows"] == ret.size

# Example: sector mix and median traded value of each RFM (Ward) cluster, straight from the mart
q = """
SELECT c.cluster_id, s.gics_sector, COUNT(*) AS n_shares,
       ROUND(AVG(f.monetary_usd)/1e9, 1) AS mean_traded_value_usd_bn
FROM fact_cluster_assignment c
JOIN fact_share_feature f ON f.share_key = c.share_key AND f.run_key = c.run_key
JOIN dim_share d ON d.share_key = c.share_key
JOIN dim_sector s ON s.sector_key = d.sector_key
WHERE c.feature_matrix = 'B_rfm' AND c.algorithm = 'ward' AND c.run_key = 1
GROUP BY c.cluster_id, s.gics_sector ORDER BY c.cluster_id, n_shares DESC
"""
example = pd.read_sql(q, con)
example.to_csv(TAB / "t3_datamart_example_query.csv", index=False)
print(example.head(8).to_string(index=False))
con.close()

# %% Star-schema diagram for the report
fig, ax = plt.subplots(figsize=(13, 7.5)); ax.set_xlim(0, 13); ax.set_ylim(0, 7.5); ax.axis("off")


def box(x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="#444", lw=1))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9.5, fontweight="bold")
    ax.text(x + 0.12, y + h - 0.6, "\n".join(lines), ha="left", va="top", fontsize=7.5, family="monospace")


F, D = "#cfe2f3", "#fff2cc"
box(4.6, 5.1, 3.8, 2.1, "fact_daily_price", ["grain: share x trading day", "close, adj_close, volume,", "traded_value_usd,", "baseline_mean_vol_20,", "event_threshold, is_event"], F)
box(0.2, 5.1, 3.4, 2.1, "dim_date", ["date_key, full_date", "week_ending_friday", "is_trading_day", "is_complete_week"], D)
box(9.4, 5.1, 3.4, 2.1, "dim_share (SCD Type 2)", ["ticker, company, cik", "date_added_to_index", "valid_from / valid_to", "-> dim_sector"], D)
box(9.4, 2.6, 3.4, 1.7, "dim_sector", ["gics_sector", "gics_sub_industry"], D)
box(0.2, 2.6, 3.4, 1.7, "dim_benchmark", ["symbol (^GSPC)", "price_basis: price index"], D)
box(4.6, 2.9, 3.8, 1.5, "fact_weekly_return", ["grain: share x Friday week", "weekly_return, benchmark_return"], F)
box(0.2, 0.2, 4.2, 2.0, "fact_share_feature", ["grain: share x obs. date x run", "mean_weekly_return, beta, R, F, M", "n_weeks, last_event_date"], F)
box(4.8, 0.2, 4.1, 2.0, "fact_cluster_assignment", ["grain: share x run x matrix x algo", "k, cluster_id"], F)
box(9.4, 0.2, 3.4, 2.0, "dim_run (provenance)", ["source, retrieved_utc, versions", "windows 20/60, multiplier,", "observation_date, seed,", "calc_version, git_commit"], "#d9ead3")
# arrows = foreign keys (from -> to); only the distinctive ones are drawn to keep the picture readable
for (x1, y1, x2, y2) in [(4.6, 6.1, 3.6, 6.1),      # fact_daily_price -> dim_date
                         (8.4, 6.1, 9.4, 6.1),      # fact_daily_price -> dim_share
                         (11.1, 5.1, 11.1, 4.3),    # dim_share -> dim_sector
                         (4.6, 3.6, 3.6, 3.6),      # fact_weekly_return -> dim_benchmark
                         (2.0, 2.2, 2.0, 2.6),      # fact_share_feature -> dim_benchmark
                         (9.4, 1.2, 8.9, 1.2)]:     # dim_run -> fact_cluster_assignment
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#555"))
ax.text(6.5, 4.75, "every fact table also carries share_key -> dim_share, a date key -> dim_date and run_key -> dim_run",
        ha="center", fontsize=8, style="italic", color="#444")
ax.text(6.5, 7.35, "Data mart star schema (blue = facts, yellow = dimensions, green = provenance)",
        ha="center", fontsize=11, fontweight="bold")
fig.savefig(FIG / "f3_4_star_schema.png", dpi=200, bbox_inches="tight")
if "inline" in plt.get_backend().lower():   # show the figure when running as a notebook
    plt.show()
plt.close(fig)
print("Data mart built:", DB)
