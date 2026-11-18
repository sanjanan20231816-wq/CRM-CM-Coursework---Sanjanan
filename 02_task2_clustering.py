# %% [markdown]
# # Task 2: Stock segmentation and sensitivity analysis
# Input : data/processed/feature_table.csv (written by 01_task1_data_features.py)
# Output: tables/, figures/, data/processed/cluster_assignments.csv, data/processed/task2_results.json
#
# Four baseline solutions = 2 feature matrices x 2 algorithms:
#   A = risk-return  [mean weekly return, weekly beta]
#   B = RFM activity [Recency, Frequency, log1p(Monetary)]
#   algorithms: Ward agglomerative, K-means
# Sections follow the brief: 2.1 preparation, 2.2 Ward, 2.3 K-means, 2.4 profiles/comparison,
# 2.5 correlated-input sensitivity.

# %% CONFIG: every tunable choice lives here
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42                  # random seed for K-means and anything random
K_RANGE = range(2, 9)      # candidate cluster counts (brief: 2 to 8)
N_INIT = 50                # K-means restarts per fit
N_STABILITY_SEEDS = 10     # seeds used in the K-means stability check

# k-selection rule: highest silhouette among k >= K_MIN_SELECT whose smallest cluster
# holds at least MIN_CLUSTER_SHARE of shares. k = 2 is still reported in every table but
# is not eligible, because a two-way split is a weak segmentation.
K_MIN_SELECT = 3
MIN_CLUSTER_SHARE = 0.05
# After reading the profiles you may overrule the rule, e.g. {"B_ward": 4}. Leave empty
# to use the rule. Any override must be justified in the report.
K_OVERRIDE = {}

LOG_R = False              # False = keep Recency raw (as in the brief); True = log1p(R)

ROOT = Path(".")
PROC, TAB, FIG = ROOT / "data/processed", ROOT / "tables", ROOT / "figures"
for d in (TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
COLORS = plt.get_cmap("tab10").colors
SOLUTIONS = ["A_ward", "A_kmeans", "B_ward", "B_kmeans"]
results = {}   # numbers collected here are saved to task2_results.json for the report


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    if "inline" in plt.get_backend().lower():   # show the figure when running as a notebook
        plt.show()
    plt.close(fig)


# %% 2.1 Load the common share set
feat = pd.read_csv(PROC / "feature_table.csv", index_col="ticker")
common = feat[feat["common_set"]].copy()          # shares valid for BOTH matrices (used everywhere below)
print(f"Common share set: {len(common)} shares (of {len(feat)} in feature table)")
results["n_common"] = int(len(common))

# %% 2.1 Feature matrices, transformations and scaling
FEATURES = {"A": ["mean_weekly_return", "beta"], "B": ["R", "F", "M_usd"]}


def prepare(name):
    """Return (transformed, standardised, scaler) for matrix 'A' or 'B'.
    Only B is transformed: log1p(Monetary) always, log1p(Recency) only if LOG_R."""
    X = common[FEATURES[name]].copy()
    if name == "B":
        X["M_usd"] = np.log1p(X["M_usd"])
        X = X.rename(columns={"M_usd": "log1p_M"})
        if LOG_R:
            X["R"] = np.log1p(X["R"])
    scaler = StandardScaler()                      # fitted separately per matrix
    Xs = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    return X, Xs, scaler


X_raw, X_std = {}, {}
for m in ("A", "B"):
    _, X_std[m], _ = prepare(m)
    X_raw[m] = common[FEATURES[m]]
Xt_B, _, _ = prepare("B")

# --- skew and outlier diagnostics (before vs after transformation)
diag_rows = []
for m in ("A", "B"):
    transformed, standardised, _ = prepare(m)
    for raw_col, new_col in zip(FEATURES[m], transformed.columns):
        z = standardised[new_col]
        diag_rows.append({
            "matrix": m, "feature": raw_col, "skew_raw": common[raw_col].skew(),
            "skew_used": transformed[new_col].skew(), "n_abs_z_gt_3": int((z.abs() > 3).sum()),
            "max_abs_z": z.abs().max(), "transform": "log1p" if new_col != raw_col else "none"})
# extra candidate: what log1p(R) would do (reported, not used unless LOG_R=True)
diag_rows.append({"matrix": "B", "feature": "R (candidate log1p)", "skew_raw": common["R"].skew(),
                  "skew_used": np.log1p(common["R"]).skew(), "n_abs_z_gt_3": np.nan,
                  "max_abs_z": np.nan, "transform": "candidate only"})
skew_table = pd.DataFrame(diag_rows).round(3)
skew_table.to_csv(TAB / "t2_skew_outlier_diagnostics.csv", index=False)
print(skew_table.to_string(index=False))

# list shares that sit more than 3 SD from the mean on any feature (kept, not removed)
outlier_rows = []
for m in ("A", "B"):
    z = X_std[m]
    for t in z.index[(z.abs() > 3).any(axis=1)]:
        outlier_rows.append({"matrix": m, "ticker": t, "company": common.loc[t, "company"],
                             **{c: round(z.loc[t, c], 2) for c in z.columns}})
pd.DataFrame(outlier_rows).to_csv(TAB / "t2_outliers_kept.csv", index=False)
print(f"{len(outlier_rows)} outlier rows saved (outliers are real data and are kept)")

# --- histograms: raw vs used
fig, axes = plt.subplots(2, 5, figsize=(15, 5))
labels_raw = {"mean_weekly_return": "Mean weekly return", "beta": "Beta", "R": "Recency R (days)",
              "F": "Frequency F (days)", "M_usd": "Monetary M (USD)"}
used = {"mean_weekly_return": common["mean_weekly_return"], "beta": common["beta"],
        "R": Xt_B["R"], "F": Xt_B["F"], "M_usd": Xt_B["log1p_M"]}
for j, col in enumerate(labels_raw):
    axes[0, j].hist(common[col], bins=40, color=COLORS[0])
    axes[0, j].set_title(f"{labels_raw[col]}\nskew {common[col].skew():.2f}", fontsize=9)
    axes[1, j].hist(used[col], bins=40, color=COLORS[1])
    tag = "log1p(M)" if col == "M_usd" else ("log1p(R)" if (col == "R" and LOG_R) else "used as is")
    axes[1, j].set_title(f"{tag}, skew {used[col].skew():.2f}", fontsize=9)
axes[0, 0].set_ylabel("raw"); axes[1, 0].set_ylabel("as clustered (before scaling)")
save_fig(fig, "f2_1_feature_distributions.png")

# %% Helper functions shared by Ward and K-means
def fit_labels(algo, X, k):
    """Cluster rows of X into k groups; return (labels, inertia). Inertia = within-cluster sum of squares."""
    if algo == "ward":
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward", metric="euclidean").fit_predict(X)
        return labels, within_ss(X, labels)
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(X)
    return km.labels_, km.inertia_


def within_ss(X, labels):
    """Within-cluster sum of squared distances to each cluster centroid."""
    X = np.asarray(X)
    return float(sum(((X[labels == c] - X[labels == c].mean(axis=0)) ** 2).sum() for c in np.unique(labels)))


def relabel(labels, order_by):
    """Renumber clusters 0..k-1 so cluster 0 has the LOWEST mean of `order_by`.
    Cluster numbers are arbitrary in clustering; this just makes tables/plots comparable.
    (The adjusted Rand index ignores numbering, so it is unaffected.)"""
    means = pd.Series(np.asarray(order_by)).groupby(labels).mean().sort_values()
    mapping = {old: new for new, old in enumerate(means.index)}
    return np.array([mapping[x] for x in labels])


def profile_table(labels):
    """Cluster profile in ORIGINAL units (not standardised)."""
    df = common.assign(cluster=labels)
    g = df.groupby("cluster")
    out = pd.DataFrame({
        "n_shares": g.size(),
        "share_pct": (g.size() / len(df) * 100),
        "return_mean_pct_wk": g["mean_weekly_return"].mean() * 100,
        "return_median_pct_wk": g["mean_weekly_return"].median() * 100,
        "beta_mean": g["beta"].mean(), "beta_median": g["beta"].median(),
        "R_mean_days": g["R"].mean(), "R_median_days": g["R"].median(),
        "F_mean": g["F"].mean(), "F_median": g["F"].median(),
        "M_median_usd_bn": g["M_usd"].median() / 1e9, "M_mean_usd_bn": g["M_usd"].mean() / 1e9,
    })
    return out.round(3)


# %% 2.2 + 2.3 Diagnostics for k = 2..8 (both algorithms, both matrices)
diag = []
for m in ("A", "B"):
    for algo in ("ward", "kmeans"):
        for k in K_RANGE:
            labels, inertia = fit_labels(algo, X_std[m], k)
            sizes = np.bincount(labels)
            diag.append({"matrix": m, "algo": algo, "k": k,
                         "silhouette": silhouette_score(X_std[m], labels),
                         "inertia": inertia, "min_size": sizes.min(),
                         "min_share": sizes.min() / len(labels),
                         "sizes": "/".join(map(str, sorted(sizes, reverse=True)))})
diag = pd.DataFrame(diag)
diag.round(4).to_csv(TAB / "t2_k_diagnostics_all.csv", index=False)
print(diag.round(3).to_string(index=False))


def choose_k(matrix, algo):
    """Apply the k-selection rule described in CONFIG; return (k, how_chosen)."""
    key = f"{matrix}_{algo}"
    if key in K_OVERRIDE:
        return K_OVERRIDE[key], "manual override"
    d = diag[(diag.matrix == matrix) & (diag.algo == algo) & (diag.k >= K_MIN_SELECT)]
    eligible = d[d.min_share >= MIN_CLUSTER_SHARE]
    how = "max silhouette, k>=3, smallest cluster >=5%"
    if eligible.empty:
        eligible, how = d, "max silhouette, k>=3 (no k met the 5% size rule)"
    return int(eligible.loc[eligible["silhouette"].idxmax(), "k"]), how


chosen = {s: choose_k(*s.split("_")) for s in SOLUTIONS}
print("\nSelected k:", {s: v[0] for s, v in chosen.items()})
results["selected_k"] = {s: {"k": v[0], "rule": v[1]} for s, v in chosen.items()}
pd.DataFrame([{"solution": s, "k": v[0], "rule": v[1]} for s, v in chosen.items()]).to_csv(
    TAB / "t2_selected_k.csv", index=False)

# --- silhouette and inertia charts (one panel per matrix)
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for j, m in enumerate(("A", "B")):
    for algo, col in (("ward", COLORS[0]), ("kmeans", COLORS[3])):
        d = diag[(diag.matrix == m) & (diag.algo == algo)]
        axes[0, j].plot(d.k, d.silhouette, "o-", color=col, label=algo)
        axes[1, j].plot(d.k, d.inertia, "o-", color=col, label=algo)
        axes[0, j].axvline(chosen[f"{m}_{algo}"][0], color=col, ls=":", alpha=0.7)
        axes[1, j].axvline(chosen[f"{m}_{algo}"][0], color=col, ls=":", alpha=0.7)
    axes[0, j].set_title(f"Matrix {m}: silhouette (higher is better)"); axes[0, j].set_xlabel("k")
    axes[1, j].set_title(f"Matrix {m}: within-cluster SS (elbow); dotted = selected k"); axes[1, j].set_xlabel("k")
    axes[0, j].legend()
save_fig(fig, "f2_2_k_diagnostics.png")

# %% 2.2 Ward dendrograms (one per matrix) + merge-height evidence
Z = {}
for m in ("A", "B"):
    Z[m] = linkage(X_std[m], method="ward", metric="euclidean")
    k = chosen[f"{m}_ward"][0]
    # sanity: scipy's tree cut must agree with sklearn's Ward at the same k
    ari = adjusted_rand_score(fcluster(Z[m], k, "maxclust"), fit_labels("ward", X_std[m], k)[0])
    assert ari > 0.99, f"scipy and sklearn Ward disagree for matrix {m} (ARI={ari:.3f})"

    heights = Z[m][:, 2]
    cut = (heights[-k] + heights[-(k - 1)]) / 2          # horizontal line giving exactly k clusters
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(Z[m], truncate_mode="lastp", p=30, ax=ax, color_threshold=0,
               above_threshold_color="grey", leaf_font_size=8, show_leaf_counts=True)
    ax.axhline(cut, color="tab:red", ls="--")
    ax.text(0.01, cut, f" cut for selected k={k}", color="tab:red", va="bottom", transform=ax.get_yaxis_transform())
    ax.set_title(f"Ward dendrogram, matrix {m} (last 30 merges; leaf label = shares in branch)")
    ax.set_ylabel("Ward merge distance (standardised features)")
    save_fig(fig, f"f2_3_dendrogram_{m}.png")

    # last 10 merges: a big jump in height means two well-separated groups are being joined
    last = pd.DataFrame({"merges_left": np.arange(10, 0, -1) + 0,
                         "clusters_after_merge": np.arange(10, 0, -1),
                         "height": heights[-10:]})
    last["jump_from_previous"] = last["height"].diff()
    last.round(3).to_csv(TAB / f"t2_ward_last_merges_{m}.csv", index=False)

# %% 2.3 K-means: initialisation sensitivity and seed stability at the selected k
km_rows = []
for m in ("A", "B"):
    k = chosen[f"{m}_kmeans"][0]
    base = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(X_std[m])
    # (a) one random start vs fifty: how much does a single start risk a bad local optimum?
    single = [KMeans(n_clusters=k, n_init=1, random_state=s).fit(X_std[m]).inertia_ for s in range(30)]
    # (b) do different seeds (each with 50 starts) give the same partition?
    aris = [adjusted_rand_score(base.labels_, KMeans(n_clusters=k, n_init=N_INIT, random_state=s)
                                .fit_predict(X_std[m])) for s in range(N_STABILITY_SEEDS)]
    km_rows.append({"matrix": m, "k": k, "inertia_n_init50": base.inertia_,
                    "inertia_single_start_min": min(single), "inertia_single_start_max": max(single),
                    "share_single_starts_worse_than_base": np.mean(np.array(single) > base.inertia_ * 1.0001),
                    "ARI_vs_base_min": min(aris), "ARI_vs_base_mean": float(np.mean(aris))})
km_stab = pd.DataFrame(km_rows).round(4)
km_stab.to_csv(TAB / "t2_kmeans_stability.csv", index=False)
print(km_stab.to_string(index=False))

# %% 2.4 Final labels for the four baseline solutions (ordered for readability)
order_by = {"A": common["beta"], "B": np.log1p(common["M_usd"])}   # cluster 0 = lowest beta / lowest traded value
labels_final = {}
for s in SOLUTIONS:
    m, algo = s.split("_")
    raw_labels, _ = fit_labels(algo, X_std[m], chosen[s][0])
    labels_final[s] = relabel(raw_labels, order_by[m])

profiles = {}
for s in SOLUTIONS:
    profiles[s] = profile_table(labels_final[s])
    profiles[s].to_csv(TAB / f"t2_profile_{s}.csv")
    print(f"\n--- Profile {s} (k={chosen[s][0]}), original units ---")
    print(profiles[s].to_string())

# %% 2.4 Labelled 2D visualisations
def flag_extremes(Xs2d, n=6):
    """Indices of the n points farthest from the centre of the 2D cloud (to annotate)."""
    d = np.sqrt((Xs2d ** 2).sum(axis=1))
    return np.argsort(d)[-n:]


def scatter_pair(coords, xlabel, ylabel, fname, title, extra_note=""):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    for ax, algo in zip(axes, ("ward", "kmeans")):
        s = f"{matrix_now}_{algo}"
        lab = labels_final[s]
        for c in np.unique(lab):
            idx = lab == c
            ax.scatter(coords[idx, 0], coords[idx, 1], s=18, color=COLORS[c % 10], alpha=0.75,
                       label=f"cluster {c} (n={idx.sum()})")
        for i in flag_extremes(coords_std_now):
            ax.annotate(common.index[i], (coords[i, 0], coords[i, 1]), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_title(f"{'Ward' if algo == 'ward' else 'K-means'} (k={chosen[s][0]})"); ax.set_xlabel(xlabel)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title + extra_note, fontsize=11)
    save_fig(fig, fname)


# Matrix A has exactly 2 features, so plot them directly: no dimension reduction needed.
matrix_now = "A"
coords_std_now = X_std["A"].values
scatter_pair(np.c_[common["beta"], common["mean_weekly_return"] * 100], "Weekly beta",
             "Mean weekly return (%)", "f2_4_scatter_A.png", "Risk-return clusters (original units, no projection)")

# Matrix B has 3 features: project to 2D with PCA on the standardised features.
pca = PCA(n_components=2, random_state=SEED).fit(X_std["B"])
pc = pca.transform(X_std["B"])
loadings = pd.DataFrame(pca.components_.T, index=X_std["B"].columns, columns=["PC1", "PC2"]).round(3)
loadings.loc["explained_variance_ratio"] = pca.explained_variance_ratio_.round(3)
loadings.to_csv(TAB / "t2_pca_loadings_B.csv")
print("\nPCA loadings (matrix B):\n", loadings)
results["pca_B_explained_variance"] = [float(v) for v in pca.explained_variance_ratio_]
matrix_now = "B"
coords_std_now = pc
scatter_pair(pc, f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)",
             f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)", "f2_4_scatter_B_pca.png",
             "RFM clusters in PCA space", "\n(2D projection of 3 standardised features: distances are distorted)")

# %% 2.4 Algorithm comparison on the same shares: ARI + crosstab heatmaps
def heatmap(ax, mat, xlabels, ylabels, title, cmap="Blues"):
    ax.imshow(mat, cmap=cmap)
    ax.set_xticks(range(len(xlabels)), xlabels); ax.set_yticks(range(len(ylabels)), ylabels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}" if float(mat[i, j]).is_integer() else f"{mat[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > mat.max() * 0.6 else "black")
    ax.set_title(title, fontsize=10)


comparison = []
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, m in zip(axes, ("A", "B")):
    a, b = labels_final[f"{m}_ward"], labels_final[f"{m}_kmeans"]
    ct = pd.crosstab(pd.Series(a, name="Ward cluster"), pd.Series(b, name="K-means cluster"))
    ct.to_csv(TAB / f"t2_crosstab_ward_vs_kmeans_{m}.csv")
    ari = adjusted_rand_score(a, b)
    comparison.append({"matrix": m, "k_ward": chosen[f"{m}_ward"][0], "k_kmeans": chosen[f"{m}_kmeans"][0],
                       "ARI_ward_vs_kmeans": ari,
                       "silhouette_ward": silhouette_score(X_std[m], a),
                       "silhouette_kmeans": silhouette_score(X_std[m], b)})
    heatmap(ax, ct.values, ct.columns, ct.index, f"Matrix {m}: shares per cell (ARI = {ari:.2f})")
    ax.set_xlabel("K-means cluster"); ax.set_ylabel("Ward cluster")
save_fig(fig, "f2_5_ward_vs_kmeans_crosstab.png")
comparison = pd.DataFrame(comparison).round(4)
comparison.to_csv(TAB / "t2_algorithm_comparison.csv", index=False)
print("\n", comparison.to_string(index=False))
results["ari_ward_vs_kmeans"] = {r.matrix: float(r.ARI_ward_vs_kmeans) for r in comparison.itertuples()}

# %% 2.5.1 Correlation analysis among R, F and log1p(M)
corr_in = pd.DataFrame({"R": common["R"], "F": common["F"], "log1p_M": np.log1p(common["M_usd"])})
pearson = corr_in.corr(method="pearson").round(3)
spearman = corr_in.corr(method="spearman").round(3)
pearson.to_csv(TAB / "t2_corr_pearson.csv"); spearman.to_csv(TAB / "t2_corr_spearman.csv")
print("\nPearson:\n", pearson, "\nSpearman:\n", spearman)

# How much of the total standardised variance lies along ONE shared direction (PC1)?
eig = np.sort(np.linalg.eigvalsh(corr_in.corr().values))[::-1]
pc1_share = float(eig[0] / eig.sum())
print(f"Share of variance on PC1 of (R, F, log1p M): {pc1_share:.2%}; eigenvalues {eig.round(3)}")
results["corr_pearson"] = pearson.to_dict()
results["corr_spearman"] = spearman.to_dict()
results["pc1_share_B"] = pc1_share

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, cm) in zip(axes, (("Pearson", pearson), ("Spearman", spearman))):
    heatmap(ax, cm.values, cm.columns, cm.index, f"{name} correlation", cmap="RdBu_r")
    ax.images[0].set_clim(-1, 1)
save_fig(fig, "f2_6_correlation_heatmap.png")

# %% 2.5.2 Remove Monetary: Ward on R and F only (same shares, same k, scaler refitted)
k_full = chosen["B_ward"][0]
Xrf = pd.DataFrame(StandardScaler().fit_transform(common[["R", "F"]]), index=common.index, columns=["R", "F"])
lab_rf_raw, _ = fit_labels("ward", Xrf, k_full)
lab_full = labels_final["B_ward"]                                # full R+F+log1p(M) Ward solution
lab_rf = relabel(lab_rf_raw, np.log1p(common["M_usd"]))          # ordered by traded value for readability
ari_rf = adjusted_rand_score(lab_full, lab_rf)
sil_rf = silhouette_score(Xrf, lab_rf)

ct_rf = pd.crosstab(pd.Series(lab_full, name="Full R,F,logM"), pd.Series(lab_rf, name="R,F only"))
ct_rf.to_csv(TAB / "t2_crosstab_full_vs_RF.csv")


def eta_squared(values, labels):
    """Share of the variance of `values` explained by cluster membership (0 to 1)."""
    v = np.asarray(values); lab = np.asarray(labels)
    ss_total = ((v - v.mean()) ** 2).sum()
    ss_between = sum((lab == c).sum() * (v[lab == c].mean() - v.mean()) ** 2 for c in np.unique(lab))
    return float(ss_between / ss_total)


logM = np.log1p(common["M_usd"])
m_profile = []
for name, lab in (("full R,F,log1p(M)", lab_full), ("R,F only", lab_rf)):
    df = common.assign(cluster=lab)
    for c, g in df.groupby("cluster"):
        m_profile.append({"solution": name, "cluster": c, "n": len(g),
                          "M_median_usd_bn": g.M_usd.median() / 1e9, "M_mean_usd_bn": g.M_usd.mean() / 1e9,
                          "M_q25_usd_bn": g.M_usd.quantile(.25) / 1e9, "M_q75_usd_bn": g.M_usd.quantile(.75) / 1e9,
                          "R_median": g.R.median(), "F_median": g.F.median()})
    m_profile.append({"solution": name, "cluster": "ALL", "n": len(df), "M_median_usd_bn": df.M_usd.median() / 1e9,
                      "M_mean_usd_bn": df.M_usd.mean() / 1e9, "M_q25_usd_bn": df.M_usd.quantile(.25) / 1e9,
                      "M_q75_usd_bn": df.M_usd.quantile(.75) / 1e9, "R_median": df.R.median(), "F_median": df.F.median()})
m_profile = pd.DataFrame(m_profile).round(3)
m_profile.to_csv(TAB / "t2_monetary_removal_profiles.csv", index=False)
print("\nTraded value (M) profiles by cluster:\n", m_profile.to_string(index=False))

eta_full, eta_rf = eta_squared(logM, lab_full), eta_squared(logM, lab_rf)
print(f"\nARI(full vs R,F only) = {ari_rf:.3f} | eta^2 of log1p(M): full {eta_full:.3f}, R,F only {eta_rf:.3f}")
print(ct_rf)

# robustness extra: does using log1p(R) instead of raw R change the selected Ward RFM solution?
Xalt, _, _ = prepare("B")
Xalt = Xalt.copy(); Xalt["R"] = np.log1p(common["R"]) if not LOG_R else common["R"]
Xalt = pd.DataFrame(StandardScaler().fit_transform(Xalt), index=common.index, columns=Xalt.columns)
ari_logR = adjusted_rand_score(lab_full, relabel(fit_labels("ward", Xalt, k_full)[0], logM))
print(f"Robustness: ARI between current RFM Ward and the version with the other R treatment = {ari_logR:.3f}")

results["monetary_removal"] = {"k": k_full, "ARI_full_vs_RF": ari_rf, "eta2_logM_full": eta_full,
                               "eta2_logM_RF_only": eta_rf, "silhouette_RF_only": float(sil_rf),
                               "silhouette_full": float(silhouette_score(X_std["B"], lab_full)),
                               "ARI_alt_R_treatment": float(ari_logR), "LOG_R_used": LOG_R}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
heatmap(axes[0], ct_rf.values, ct_rf.columns, ct_rf.index, f"Full vs R,F only: shares per cell (ARI {ari_rf:.2f})")
axes[0].set_xlabel("R,F-only cluster"); axes[0].set_ylabel("Full-RFM cluster")
for name, lab, col in (("full R,F,log1p(M)", lab_full, COLORS[0]), ("R,F only", lab_rf, COLORS[3])):
    med = common.assign(c=lab).groupby("c").M_usd.median() / 1e9
    axes[1].plot(med.index, med.values, "o-", color=col, label=name)
axes[1].set_yscale("log"); axes[1].set_xlabel("cluster (0 = lowest traded value)")
axes[1].set_ylabel("median M (USD bn, log scale)"); axes[1].legend(); axes[1].set_title("Traded value by cluster")
save_fig(fig, "f2_7_monetary_removal.png")

# %% Save assignments and results for Task 3 / report
assign = pd.DataFrame({s: labels_final[s] for s in SOLUTIONS}, index=common.index)
assign["B_ward_RF_only"] = lab_rf
assign.index.name = "ticker"
assign.to_csv(PROC / "cluster_assignments.csv")
full = feat.join(assign)                              # feature table + all assignments (submission checklist item)
full.to_csv(PROC / "feature_table_with_clusters.csv")
results["seed"], results["n_init"], results["k_range"] = SEED, N_INIT, [min(K_RANGE), max(K_RANGE)]
results["cluster_sizes"] = {s: np.bincount(labels_final[s]).tolist() for s in SOLUTIONS}
(PROC / "task2_results.json").write_text(json.dumps(results, indent=2, default=float))
print("\nTask 2 complete. Next: 03_task3_review.py")
