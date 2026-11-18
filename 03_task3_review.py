# %% [markdown]
# # Task 3.2: Evidence for the business review and critical evaluation
# Input : outputs of scripts 01 and 02 (data/processed/*, data/raw/*)
# Output: tables/t3_*.csv, figures/f3_*.png, data/processed/task3_results.json
#
# Each section produces the numbers behind one sub-question of the brief:
#   3.2.1 cluster profiles + sector composition (risk-return vs RFM clusters)
#   3.2.2 sensitivity findings are produced in script 02 (correlations, Monetary removal)
#   3.2.3 limitations: return dependencies, price-adjustment basis, survivorship bias
# (3.1, the data mart, is built by 04_build_datamart.py.)

# %% CONFIG
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.metrics import adjusted_rand_score

SEED = 42
N_PERMUTATIONS = 500       # random relabellings used as the "no structure" baseline
START, END = "2021-01-01", "2025-01-01"
SOLUTIONS = ["A_ward", "A_kmeans", "B_ward", "B_kmeans"]

ROOT = Path(".")
RAW, PROC, TAB, FIG = ROOT / "data/raw", ROOT / "data/processed", ROOT / "tables", ROOT / "figures"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
COLORS = plt.get_cmap("tab10").colors
rng = np.random.default_rng(SEED)
results = {}


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    if "inline" in plt.get_backend().lower():   # show the figure when running as a notebook
        plt.show()
    plt.close(fig)


# %% Load everything produced so far
feat = pd.read_csv(PROC / "feature_table_with_clusters.csv", index_col="ticker")
feat = feat[feat["common_set"]]                                    # the same 486 shares as Task 2
ret = pd.read_parquet(PROC / "weekly_returns.parquet")[feat.index]   # weeks x shares
ret_m = pd.read_parquet(PROC / "weekly_returns_benchmark.parquet")["^GSPC"]
print(f"{len(feat)} shares, {len(ret)} weekly returns")

# %% 3.2.1a Sector composition of every cluster
def cramers_v(table):
    """Association between two categorical variables (0 = none, 1 = perfect)."""
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


sector_rows = []
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
sectors = sorted(feat["sector"].unique())
sector_colors = dict(zip(sectors, plt.get_cmap("tab20").colors))
for ax, s in zip(axes.ravel(), SOLUTIONS):
    ct = pd.crosstab(feat[s], feat["sector"])
    share = ct.div(ct.sum(axis=1), axis=0) * 100               # % of each cluster in each sector
    share.round(1).to_csv(TAB / f"t3_sector_share_{s}.csv")
    left = np.zeros(len(share))
    for sec in sectors:
        vals = share[sec].values if sec in share else np.zeros(len(share))
        ax.barh([f"C{c} (n={ct.loc[c].sum()})" for c in share.index], vals, left=left,
                color=sector_colors[sec], label=sec)
        left += vals
    ax.set_xlabel("% of cluster"); ax.set_title(f"{s}: sector mix per cluster")
    chi2, p, _, _ = chi2_contingency(ct, correction=False)
    sector_rows.append({"solution": s, "chi2": chi2, "p_value": p, "cramers_V": cramers_v(ct)})
axes[0, 0].legend(fontsize=6, ncol=2, loc="lower right")
save_fig(fig, "f3_1_sector_composition.png")
sector_assoc = pd.DataFrame(sector_rows).round(4)
sector_assoc.to_csv(TAB / "t3_sector_association.csv", index=False)
print(sector_assoc.to_string(index=False))
results["sector_association"] = sector_assoc.set_index("solution").to_dict("index")

# %% 3.2.1b Risk-return clusters vs RFM clusters (same shares)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
cross_rows = []
for ax, algo in zip(axes, ("ward", "kmeans")):
    a, b = feat[f"A_{algo}"], feat[f"B_{algo}"]
    ct = pd.crosstab(a.rename("risk-return cluster"), b.rename("RFM cluster"))
    ct.to_csv(TAB / f"t3_crosstab_A_vs_B_{algo}.csv")
    ari, v = adjusted_rand_score(a, b), cramers_v(ct)
    cross_rows.append({"algorithm": algo, "ARI_A_vs_B": ari, "cramers_V": v})
    ax.imshow(ct.values, cmap="Blues")
    ax.set_xticks(range(ct.shape[1]), ct.columns); ax.set_yticks(range(ct.shape[0]), ct.index)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, ct.values[i, j], ha="center", va="center", fontsize=9,
                    color="white" if ct.values[i, j] > ct.values.max() * 0.6 else "black")
    ax.set_xlabel("RFM cluster"); ax.set_ylabel("Risk-return cluster")
    ax.set_title(f"{algo}: ARI {ari:.2f}, Cramer's V {v:.2f}")
save_fig(fig, "f3_2_risk_vs_rfm_crosstab.png")
cross = pd.DataFrame(cross_rows).round(4)
cross.to_csv(TAB / "t3_A_vs_B_agreement.csv", index=False)
print(cross.to_string(index=False))
results["A_vs_B_agreement"] = cross.set_index("algorithm").to_dict("index")

# %% 3.2.3a Return dependencies: do clusters group shares that move together?
# Similar beta does not mean similar co-movement. Compare average pairwise correlation of
# weekly returns for pairs INSIDE the same cluster vs pairs in DIFFERENT clusters, and
# against random labelling (permutation) with identical cluster sizes.
C = ret.corr().values
iu = np.triu_indices_from(C, k=1)                        # each pair counted once
corr_all = C[iu]
tickers = list(ret.columns)


def within_across(labels):
    labels = np.asarray(labels)
    same = (labels[:, None] == labels[None, :])[iu]
    return corr_all[same].mean(), corr_all[~same].mean()


dep_rows = []
for s in SOLUTIONS:
    lab = feat.loc[tickers, s].values
    w, a = within_across(lab)
    perm = np.array([within_across(rng.permutation(lab))[0] for _ in range(N_PERMUTATIONS)])
    dep_rows.append({"solution": s, "within_cluster_mean_corr": w, "across_cluster_mean_corr": a,
                     "random_label_within_mean": perm.mean(), "random_label_within_sd": perm.std(),
                     "z_vs_random": (w - perm.mean()) / perm.std()})

# Same question for sectors (a "natural" grouping), and on market-model RESIDUALS:
# after removing each share's beta exposure, are same-sector shares still correlated?
resid = ret.sub(ret_m.values[:, None] * feat.loc[tickers, "beta"].values[None, :], axis=1)
resid = resid.sub(resid.mean())                          # remove intercepts
Cr = resid.corr().values[iu]
sec = feat.loc[tickers, "sector"].to_numpy(dtype=object)   # plain numpy so [:, None] indexing works
same_sec = (sec[:, None] == sec[None, :])[iu]
dep_rows.append({"solution": "sector (GICS)", "within_cluster_mean_corr": corr_all[same_sec].mean(),
                 "across_cluster_mean_corr": corr_all[~same_sec].mean(), "random_label_within_mean": np.nan,
                 "random_label_within_sd": np.nan, "z_vs_random": np.nan})
dep = pd.DataFrame(dep_rows).round(4)
dep.to_csv(TAB / "t3_return_dependence.csv", index=False)
print(dep.to_string(index=False))
resid_summary = {"all_pairs_mean_corr": float(corr_all.mean()),
                 "residual_same_sector_mean_corr": float(Cr[same_sec].mean()),
                 "residual_diff_sector_mean_corr": float(Cr[~same_sec].mean())}
print("Residual (beta-removed) correlation:", resid_summary)
results["return_dependence"] = dep.set_index("solution").to_dict("index")
results["residual_dependence"] = resid_summary

# Beta similarity vs co-movement: pairs with near-identical beta but low/high correlation
b = feat.loc[tickers, "beta"].values
close_beta = (np.abs(b[:, None] - b[None, :]) < 0.05)[iu]
results["pairs_beta_within_0.05"] = {"n_pairs": int(close_beta.sum()),
                                     "corr_min": float(corr_all[close_beta].min()),
                                     "corr_median": float(np.median(corr_all[close_beta])),
                                     "corr_max": float(corr_all[close_beta].max())}
print("Pairs with beta within 0.05:", results["pairs_beta_within_0.05"])

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter(np.abs(b[:, None] - b[None, :])[iu][::7], corr_all[::7], s=3, alpha=0.3)
ax.set_xlabel("|beta_i - beta_j|"); ax.set_ylabel("weekly return correlation of the pair")
ax.set_title("Similar beta does not imply similar co-movement (1 in 7 pairs plotted)")
save_fig(fig, "f3_3_beta_gap_vs_correlation.png")

# %% 3.2.3b Price-adjustment assumption: Adj Close vs Close betas
# Recompute weekly returns/betas from CLOSE (split-adjusted, not dividend-adjusted) and compare.
prices = pd.read_parquet(RAW / "prices_daily.parquet")
prices["date"] = pd.to_datetime(prices["date"])
bench = pd.read_csv(RAW / "gspc_daily.csv", index_col=0, parse_dates=True)


def weekly_returns_from(daily):
    """Friday-ending weekly returns; same week rules as script 01 (drop incomplete first/last week)."""
    w = daily.resample("W-FRI").last()
    monday = w.index - pd.Timedelta(days=4)
    w = w[(monday >= pd.Timestamp(START)) & (w.index <= pd.Timestamp(END) - pd.Timedelta(days=1))]
    return w.pct_change().iloc[1:]


close = prices[prices.Ticker.isin(feat.index)].pivot(index="date", columns="Ticker", values="Close")[feat.index]
ret_close = weekly_returns_from(close)
rm_close = weekly_returns_from(bench[["Close"]])["Close"]
assert ret_close.index.equals(ret.index) and rm_close.index.equals(ret_m.index)
beta_close = ret_close.apply(lambda s: s.cov(rm_close)) / rm_close.var()
adj_gap = pd.DataFrame({"beta_adj": feat["beta"], "beta_close": beta_close,
                        "mean_ret_adj": feat["mean_weekly_return"], "mean_ret_close": ret_close.mean()})
adj_gap["beta_diff"] = adj_gap.beta_adj - adj_gap.beta_close
adj_gap["mean_ret_diff_pct_wk"] = (adj_gap.mean_ret_adj - adj_gap.mean_ret_close) * 100
adj_gap.round(5).to_csv(TAB / "t3_adjusted_vs_close_betas.csv")
adj_summary = {"corr_beta_adj_vs_close": float(adj_gap.beta_adj.corr(adj_gap.beta_close)),
               "mean_abs_beta_diff": float(adj_gap.beta_diff.abs().mean()),
               "max_abs_beta_diff": float(adj_gap.beta_diff.abs().max()),
               "mean_return_uplift_pct_per_week": float(adj_gap.mean_ret_diff_pct_wk.mean()),
               "mean_return_uplift_pct_annualised_approx": float(adj_gap.mean_ret_diff_pct_wk.mean() * 52),
               "benchmark_adj_equals_close": bool(np.allclose(bench["Adj Close"], bench["Close"]))
               if "Adj Close" in bench else None}
print("Adjusted vs Close:", adj_summary)
results["adjustment_check"] = adj_summary

# %% 3.2.3c Survivorship bias evidence: shares added to the index during/after the study start
cons = pd.read_csv(RAW / "sp500_constituents.csv")
cons["ticker"] = cons["Symbol"].str.replace(".", "-", regex=False)
cons["date_added"] = pd.to_datetime(cons["Date added"], errors="coerce")
feat = feat.join(cons.set_index("ticker")["date_added"])
late = feat["date_added"] > pd.Timestamp(START)          # joined the index after the study began
surv = pd.DataFrame({
    "group": ["added to index after 2021-01-01", "already constituents at 2021-01-01 (or date unknown)"],
    "n_shares": [int(late.sum()), int((~late).sum())],
    "mean_weekly_return_pct": [feat.loc[late, "mean_weekly_return"].mean() * 100,
                               feat.loc[~late, "mean_weekly_return"].mean() * 100],
    "median_weekly_return_pct": [feat.loc[late, "mean_weekly_return"].median() * 100,
                                 feat.loc[~late, "mean_weekly_return"].median() * 100],
    "mean_beta": [feat.loc[late, "beta"].mean(), feat.loc[~late, "beta"].mean()]}).round(4)
surv.to_csv(TAB / "t3_survivorship_evidence.csv", index=False)
stat, p = mannwhitneyu(feat.loc[late, "mean_weekly_return"], feat.loc[~late, "mean_weekly_return"])
print(surv.to_string(index=False), f"\nMann-Whitney p = {p:.4f}")
results["survivorship"] = {"n_added_after_start": int(late.sum()), "n_total": int(len(feat)),
                           "mann_whitney_p": float(p), "table": surv.to_dict("records")}
print("Note: shares REMOVED from the index before the retrieval date are absent from the data entirely.")

# %% Save results for the report
(PROC / "task3_results.json").write_text(json.dumps(results, indent=2, default=float))
print("Task 3 analysis complete. Next: 04_build_datamart.py")
