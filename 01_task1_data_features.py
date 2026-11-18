# %% [markdown]
# # Task 1: Data preparation and feature engineering
# Sections follow the brief: 1.1 acquisition, 1.2 validation, 1.3 weekly return/beta,
# 1.4 adapted RFM, 1.5 evidence (worked example, summaries).

# %% CONFIG (change parameters here only)
import io, json, sys, time, platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

START = "2021-01-01"
END = "2025-01-01"          # exclusive (yfinance convention)
BASELINE = 20               # sessions in the rolling volume baseline
WINDOW = 60                 # sessions for Frequency and Monetary
MULT = 2.0                  # high-volume multiplier
SEED = 42
K_RANGE = range(2, 9)
import os
USE_CACHE = os.environ.get("CW1_USE_CACHE", "0") == "1"   # True = reload data/raw snapshot instead of downloading
CHUNK = 50                  # tickers per yfinance request

ROOT = Path(".")
RAW, PROC, TAB, FIG = ROOT/"data/raw", ROOT/"data/processed", ROOT/"tables", ROOT/"figures"
for d in (RAW, PROC, TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

import yfinance as yf
import sklearn, scipy, matplotlib
versions = {"python": sys.version.split()[0], "platform": platform.platform(),
            "pandas": pd.__version__, "numpy": np.__version__, "yfinance": yf.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__}
(TAB/"package_versions.txt").write_text("\n".join(f"{k}=={v}" for k, v in versions.items()))
print(versions)

excl_log = []   # every exclusion and warning goes here
def log(ticker, stage, rule, detail, action):
    excl_log.append(dict(ticker=ticker, stage=stage, rule=rule, detail=detail, action=action))

# %% 1.1a Constituent list from Wikipedia
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CONS_FILE = RAW/"sp500_constituents.csv"
META_FILE = RAW/"retrieval_metadata.json"

if USE_CACHE:
    cons = pd.read_csv(CONS_FILE)
    retrieval = json.loads(META_FILE.read_text())
else:
    html = requests.get(WIKI, headers={"User-Agent": "Mozilla/5.0 (university coursework, pandas read_html)"},
                        timeout=30)
    html.raise_for_status()
    cons = pd.read_html(io.StringIO(html.text))[0]
    retrieval = {"wikipedia_url": WIKI,
                 "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "n_rows": int(len(cons))}
    cons.to_csv(CONS_FILE, index=False)
    META_FILE.write_text(json.dumps(retrieval, indent=2))

cons = cons.rename(columns={"Symbol": "wiki_symbol", "Security": "company", "GICS Sector": "sector",
                            "GICS Sub-Industry": "sub_industry", "Date added": "date_added", "CIK": "cik"})
cons["ticker"] = cons["wiki_symbol"].str.replace(".", "-", regex=False)   # BRK.B -> BRK-B
changed = cons[cons.ticker != cons.wiki_symbol]
print("Retrieved:", retrieval["retrieved_utc"], "| constituents:", len(cons))
print("Ticker symbol changes for Yahoo:\n", changed[["wiki_symbol", "ticker"]].to_string(index=False))
for _, r in changed.iterrows():
    log(r.ticker, "1.1", "ticker_format", f"{r.wiki_symbol} -> {r.ticker}", "INFO")

tickers = cons["ticker"].tolist()

# %% 1.1b Download daily prices (auto_adjust=False keeps Close AND Adj Close)
PRICE_FILE = RAW/"prices_daily.parquet"
BENCH_FILE = RAW/"gspc_daily.csv"

def download_chunk(tk, retries=3):
    for a in range(retries):
        try:
            d = yf.download(tk, start=START, end=END, auto_adjust=False, actions=True,
                            group_by="ticker", progress=False, threads=True)
            if d is not None and not d.empty:
                return d
        except Exception as e:
            print("retry", a, e)
        time.sleep(5 * (a + 1))
    return None

def to_long(d, tk):
    if not isinstance(d.columns, pd.MultiIndex):
        d.columns = pd.MultiIndex.from_product([[tk[0]], d.columns])
    out = []
    for t in tk:
        if t in d.columns.get_level_values(0):
            x = d[t].copy()
            x["Ticker"] = t
            out.append(x.reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

if USE_CACHE:
    prices = pd.read_parquet(PRICE_FILE)
    bench = pd.read_csv(BENCH_FILE, index_col=0, parse_dates=True)
else:
    parts, failed = [], []
    for i in range(0, len(tickers), CHUNK):
        tk = tickers[i:i + CHUNK]
        d = download_chunk(tk)
        if d is None:
            failed += tk
            continue
        parts.append(to_long(d, tk))
        print(f"downloaded {i + len(tk)}/{len(tickers)}")
    prices = pd.concat(parts, ignore_index=True)
    prices = prices.rename(columns={"Date": "date"})
    prices["date"] = pd.to_datetime(prices["date"])
    for t in failed:
        log(t, "1.1", "download_failed", "no data returned after retries", "EXCLUDE")
    b = yf.download("^GSPC", start=START, end=END, auto_adjust=False, progress=False)
    if isinstance(b.columns, pd.MultiIndex):
        b.columns = b.columns.get_level_values(0)
    bench = b
    prices.to_parquet(PRICE_FILE, index=False)
    bench.to_csv(BENCH_FILE)

prices["date"] = pd.to_datetime(prices["date"])
print(prices.shape, "| benchmark rows:", len(bench), "| range:", prices.date.min().date(), "->", prices.date.max().date())

# %% 1.2 Validation
bench_idx = bench.index
assert bench_idx.is_unique and bench_idx.min() >= pd.Timestamp(START) and bench_idx.max() < pd.Timestamp(END)
FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
valid, by_ticker = [], {t: g for t, g in prices.groupby("Ticker")}

for t in tickers:
    g = by_ticker.get(t)
    if g is None:
        log(t, "1.2", "no_data", "ticker absent from download", "EXCLUDE")
        continue
    n_raw = len(g)
    g = g.dropna(subset=FIELDS, how="all")                    # drops pre-listing padding only; no filling
    if n_raw != len(g):
        log(t, "1.2", "padding_rows_dropped", f"{n_raw - len(g)} all-NaN rows", "INFO")
    if g.empty:
        log(t, "1.2", "no_data", "all rows empty", "EXCLUDE"); continue
    ok = True
    dups = int(g.date.duplicated().sum())
    if dups:
        log(t, "1.2", "duplicate_dates", f"{dups} duplicates", "EXCLUDE"); ok = False
    g = g.set_index("date").sort_index()
    g = g[~g.index.duplicated(keep="first")]
    nmiss = int(g[["Close", "Adj Close", "Volume"]].isna().any(axis=1).sum())
    if nmiss:
        log(t, "1.2", "missing_values", f"{nmiss} days with NaN Close/AdjClose/Volume", "EXCLUDE"); ok = False
    npos = int(((g[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any(axis=1)).sum())
    if npos:
        log(t, "1.2", "nonpositive_price", f"{npos} days", "EXCLUDE"); ok = False
    nvol = int(((g.Volume < 0) | (g.Volume == 0)).sum())
    if nvol:
        log(t, "1.2", "volume_error", f"{nvol} days with zero/negative volume", "EXCLUDE"); ok = False
    missing_dates = bench_idx.difference(g.index)
    if len(missing_dates):
        first = g.index.min().date()
        late = first > bench_idx.min().date()
        log(t, "1.2", "unequal_history",
            f"{len(missing_dates)} benchmark days absent; first obs {first}" + (" (late listing)" if late else " (gaps)"),
            "EXCLUDE"); ok = False
    extra = g.index.difference(bench_idx)
    if len(extra):
        log(t, "1.2", "dates_not_in_benchmark", f"{len(extra)} extra dates", "WARN")
    # corporate actions
    if ok:
        splits = g.index[g["Stock Splits"].fillna(0) > 0]
        r = g.Close.pct_change()
        for sd in splits:
            win = r.loc[sd - pd.Timedelta(days=5): sd + pd.Timedelta(days=5)]
            worst = win.abs().max()
            ratio = float(g.loc[sd, "Stock Splits"])
            if worst > 0.40:
                log(t, "1.2", "split_price_jump", f"split {ratio}:1 on {sd.date()}, max |ret| {worst:.2f}: Close not split-adjusted?", "EXCLUDE"); ok = False
            else:
                log(t, "1.2", "split_checked", f"split {ratio}:1 on {sd.date()}, max |ret| {worst:.2f}", "INFO")
        # Adj Close vs Close should only diverge on dividend/split events
        rr = g["Adj Close"].pct_change() - r
        evdays = g.index[(g["Dividends"].fillna(0) > 0) | (g["Stock Splits"].fillna(0) > 0)]
        # adjustment shows on the day BEFORE ex-date in some data; allow +-1 session
        near = set()
        for e in evdays:
            loc = g.index.get_loc(e)
            near.update(g.index[max(0, loc - 1): loc + 2])
        unexplained = rr[(rr.abs() > 0.01) & (~rr.index.isin(near))]
        if len(unexplained):
            log(t, "1.2", "adjustment_inconsistency",
                f"{len(unexplained)} days where AdjClose/Close returns differ >1% away from dividend/split dates", "WARN")
    if ok:
        valid.append(t)

print("Valid after 1.2:", len(valid), "of", len(tickers))

# %% 1.2b Build aligned panels (dates x tickers) for retained shares
sub = prices[prices.Ticker.isin(valid)]
close = sub.pivot(index="date", columns="Ticker", values="Close").reindex(bench_idx)[valid]
adj = sub.pivot(index="date", columns="Ticker", values="Adj Close").reindex(bench_idx)[valid]
vol = sub.pivot(index="date", columns="Ticker", values="Volume").reindex(bench_idx)[valid]
assert not close.isna().any().any() and not adj.isna().any().any() and not vol.isna().any().any()
bench_adj = bench["Adj Close"] if "Adj Close" in bench else bench["Close"]
bench_note = "Adj Close" if "Adj Close" in bench else "Close"
print("Benchmark price basis:", bench_note, "(^GSPC is a price index; Adj Close == Close expected)")
print("Max |Adj-Close| on benchmark:", float((bench["Adj Close"] - bench["Close"]).abs().max()) if "Adj Close" in bench else "n/a")

# %% 1.3 Weekly returns (Friday-ending weeks, last available trading price)
def weekly_prices(df):
    w = df.resample("W-FRI").last()
    # drop incomplete first/last calendar weeks, keep holiday-shortened ones
    week_monday = w.index - pd.Timedelta(days=4)
    keep = (week_monday >= pd.Timestamp(START)) & (w.index <= pd.Timestamp(END) - pd.Timedelta(days=1))
    return w[keep]

w_adj = weekly_prices(adj)
w_bench = weekly_prices(bench_adj.to_frame("^GSPC"))["^GSPC"]
assert w_adj.index.equals(w_bench.index), "weekly labels differ between shares and benchmark"
assert not w_adj.isna().any().any()
ret = w_adj.pct_change().iloc[1:]                    # first week has no preceding price
ret_m = w_bench.pct_change().iloc[1:]
assert ret.index.equals(ret_m.index)
n_weeks = len(ret)
print(f"Weekly prices: {len(w_adj)} ({w_adj.index[0].date()} -> {w_adj.index[-1].date()}); weekly returns: {n_weeks}")

mean_ret = ret.mean()                                 # arithmetic mean weekly return

# %% 1.3.2 Weekly beta = cov(r_i, r_m) / var(r_m), plus independent OLS check
beta = ret.apply(lambda s: s.cov(ret_m)) / ret_m.var()          # ddof=1 in both
slope = pd.Series({t: np.polyfit(ret_m.values, ret[t].values, 1)[0] for t in ret.columns})
assert np.allclose(beta.values, slope.values), "beta formula disagrees with OLS slope"
corr_m = ret.apply(lambda s: s.corr(ret_m))
beta_se = np.sqrt((1 - corr_m**2) / (n_weeks - 2)) * ret.std() / ret_m.std()
print("Beta check vs OLS slope: OK | median beta", round(beta.median(), 3))

# %% 1.4 Adapted RFM
obs_date = close.index.max()
print("Observation date:", obs_date.date())

def event_flags(v, baseline=BASELINE, mult=MULT):
    """v: volume DataFrame/Series. Threshold = mult * mean of previous `baseline` sessions
    (current day excluded via shift(1)); strictly greater. NaN baseline -> not testable (False)."""
    base = v.shift(1).rolling(baseline, min_periods=baseline).mean()
    return (v > mult * base) & base.notna(), base

def recency_days(event_dates, obs):
    if len(event_dates) == 0:
        return np.nan
    return (obs - max(event_dates)).days

# --- unit tests (brief's examples)
assert recency_days([pd.Timestamp("2024-09-15")], pd.Timestamp("2024-09-20")) == 5
assert recency_days([pd.Timestamp("2024-09-20")], pd.Timestamp("2024-09-20")) == 0
assert np.isnan(recency_days([], pd.Timestamp("2024-09-20")))
tv = pd.Series([1e6] * 20 + [2.5e6, 2.0e6, 1.8e6], index=pd.bdate_range("2024-01-01", periods=23))
tf, _ = event_flags(tv)
assert tf.iloc[20] and not tf.iloc[21] and not tf.iloc[22], "threshold test failed (strict >2x, 2.0M not an event)"
print("Unit tests passed")

events, baseline_mean = event_flags(vol)
sessions = len(vol)
assert sessions >= BASELINE + WINDOW

R = pd.Series({t: recency_days(events.index[events[t]], obs_date) for t in vol.columns})
last_event = pd.Series({t: (events.index[events[t]].max() if events[t].any() else pd.NaT) for t in vol.columns})
win = slice(-WINDOW, None)
base_ok = bool(baseline_mean.iloc[win].notna().all().all())        # every window day has a full 20-session baseline
F = events.iloc[win].sum()
M = (close * vol).iloc[win].sum()

# independent slow check of F for a few shares
for t in vol.columns[:5]:
    v = vol[t].values; cnt = 0
    for i in range(sessions - WINDOW, sessions):
        if v[i] > MULT * v[i - BASELINE:i].mean():
            cnt += 1
    assert cnt == F[t], f"F mismatch for {t}"
print("Independent loop check of F passed")

noevent = R[R.isna()].index
for t in noevent:
    log(t, "1.4", "no_event", "no high-volume day in valid history: R missing", "EXCLUDE_RFM")

# %% Feature table (one row per included share)
feat = pd.DataFrame({
    "mean_weekly_return": mean_ret, "beta": beta, "beta_se": beta_se, "n_weeks": n_weeks,
    "R": R, "F": F, "M_usd": M, "last_event_date": last_event})
feat = feat.join(cons.set_index("ticker")[["company", "sector", "sub_industry"]])
feat["valid_riskreturn"] = feat[["mean_weekly_return", "beta"]].notna().all(axis=1)
feat["valid_rfm"] = feat[["R", "F", "M_usd"]].notna().all(axis=1)
feat["common_set"] = feat.valid_riskreturn & feat.valid_rfm
feat.index.name = "ticker"
feat.to_csv(PROC/"feature_table.csv")
ret.to_parquet(PROC/"weekly_returns.parquet"); ret_m.to_frame("^GSPC").to_parquet(PROC/"weekly_returns_benchmark.parquet")
events.to_parquet(PROC/"event_flags.parquet")
pd.DataFrame(excl_log).to_csv(PROC/"exclusion_log.csv", index=False)

run_meta = {"observation_date": str(obs_date.date()), "n_trading_sessions": sessions, "n_weekly_returns": n_weeks,
            "first_week": str(ret.index[0].date()), "last_week": str(ret.index[-1].date()),
            "n_constituents_listed": len(tickers), "n_valid_after_validation": len(valid),
            "n_common_set": int(feat.common_set.sum()), "baseline": BASELINE, "window": WINDOW, "mult": MULT,
            "seed": SEED, "constituents_retrieved_utc": retrieval["retrieved_utc"], **versions}
(PROC/"run_metadata.json").write_text(json.dumps(run_meta, indent=2))
print(json.dumps(run_meta, indent=2))
print("Window days all have full baseline:", bool(base_ok))

# %% 1.5 Evidence: summaries, exclusion summary
cols = ["mean_weekly_return", "beta", "R", "F", "M_usd"]
summ = feat.loc[feat.common_set, cols].agg(["count", "mean", "median", "std", "min", "max", "skew"]).T
summ.to_csv(TAB/"feature_summary.csv"); print(summ.round(4))
log_df = pd.DataFrame(excl_log)
print(log_df.groupby(["rule", "action"]).size())
log_df.groupby(["stage", "rule", "action"]).size().rename("n").to_csv(TAB/"exclusion_summary.csv")
print("R value counts (top):\n", feat.R.value_counts().head(8))

# %% 1.5 Worked example (default AAPL)
EX = "AAPL" if "AAPL" in vol.columns else vol.columns[0]
ex = pd.DataFrame({"Close": close[EX], "Volume": vol[EX], "mean_prev20": baseline_mean[EX],
                   "threshold_2x": MULT * baseline_mean[EX], "event": events[EX]})
last60 = ex.iloc[-WINDOW:]
ev_table = ex[ex.event].tail(8).copy()
ev_table.to_csv(TAB/f"worked_example_{EX}_events.csv")
print(f"\nWorked example {EX}: obs date {obs_date.date()}")
print(ev_table)
le = ex.index[ex.event].max()
print(f"R = ({obs_date.date()} - {le.date()}) = {R[EX]} calendar days")
print(f"F = {int(F[EX])} events in last {WINDOW} sessions ({last60.index[0].date()} -> {last60.index[-1].date()})")
print(f"M = sum(Close*Volume) over those {WINDOW} sessions = ${M[EX]:,.0f} (= ${M[EX]/1e9:,.1f}bn)")

fig, ax = plt.subplots(figsize=(10, 4))
plot = ex.iloc[-120:]
ax.bar(plot.index, plot.Volume / 1e6, color="lightgrey", label="Volume")
ax.plot(plot.index, plot.threshold_2x / 1e6, color="tab:red", lw=1.2, label="2x mean of previous 20 sessions")
ax.scatter(plot.index[plot.event], plot.Volume[plot.event] / 1e6, color="tab:blue", zorder=3, label="High-volume event")
ax.axvline(last60.index[0], ls="--", color="k", lw=0.8); ax.text(last60.index[0], ax.get_ylim()[1]*0.95, " 60-session window", fontsize=8)
ax.set_ylabel("Shares (millions)"); ax.set_title(f"{EX}: high-volume event rule (observation date {obs_date.date()})"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG/f"worked_example_{EX}.png", dpi=200)
if "inline" in plt.get_backend().lower():   # show the figure when running as a notebook
    plt.show()

# %% Quick sanity: beta vs sector sanity and known benchmarks
print(feat.loc[[t for t in ["AAPL", "MSFT", "NVDA", "JNJ", "XOM"] if t in feat.index],
               ["mean_weekly_return", "beta", "R", "F", "M_usd"]].round(4))
print("Task 1 complete. Next: Task 2 clustering (02_task2_clustering.py).")
