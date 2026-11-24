# 📊 S&P 500 Share Segmentation
### Risk-Return vs Adapted RFM

**Module:** 6BUIS017W CRM & CM with Business Intelligence – Coursework 1  
**Author:** Sanjanan Krishnakumar (20231816)

---

## 📌 Project Overview

This project analyses and groups S&P 500 shares using two different approaches:

- 📈 **Risk and Return:** Based on average weekly returns and beta (how a share moves compared with the S&P 500).
- 📊 **Trading Activity (Adapted RFM):** Based on how recently and frequently unusually high trading-volume days occurred, and the approximate value traded.

Both approaches use **Ward Agglomerative Clustering** and **K-means** to group similar shares. The results are then compared, including additional tests to understand how changes to the input variables affect the groups.

> **Note:** RFM measures trading activity in this project. It does not measure company value, customer loyalty or profitability.

---

## 🚀 Getting Started

### ☁️ Google Colab (Recommended)

[**Open the notebook in Google Colab →**](https://colab.research.google.com/github/sanjanan20231816-wq/CRM-CM-Coursework---Sanjanan/blob/main/sanjanan-20231816.ipynb)

1. Open the notebook.
2. Select **Runtime → Run all**.
3. Wait for all cells to finish running.

The notebook uses the saved data included in this repository, so the results should match those presented in the coursework.

### 💻 Running Locally

If you prefer to run the notebook on your computer:

```bash
git clone https://github.com/sanjanan20231816-wq/CRM-CM-Coursework---Sanjanan.git
cd CRM-CM-Coursework---Sanjanan
pip install pandas numpy yfinance scikit-learn scipy matplotlib pyarrow lxml html5lib
```

Then open `sanjanan-20231816.ipynb` in Jupyter Notebook and run all cells.

**💡 Tip:** The saved data is used by default when available. Downloading fresh data may produce slightly different results because S&P 500 membership changes over time.

---

## 📂 Data Overview

| Item | Details |
|---|---|
| Share list | Wikipedia – List of S&P 500 Companies |
| Price data | Yahoo Finance |
| Benchmark | S&P 500 (^GSPC) |
| Study period | 1 January 2021 – 31 December 2024 |
| Daily observations | 1,005 trading sessions |
| Weekly returns | 207 weeks |
| RFM calculation date | 31 December 2024 |
| Shares analysed | 486 out of 503 |

Some shares were excluded because of missing data, late listings or data errors. These exclusions are recorded in `exclusion_log.csv`.

The original data has been saved in `data/raw` so that the reported results can be reproduced without downloading the data again.

---

## ⚙️ Methodology

| Step | What was done |
|---|---|
| 🔍 Data checking | Checked for missing values, duplicate dates, invalid prices and trading volumes. Shares that failed the checks were excluded. |
| 📈 Average return | Calculated the average weekly return for each share using adjusted closing prices. |
| 📉 Beta | Measured how each share's weekly returns moved in relation to the S&P 500. |
| 🔊 High-volume event | Identified days when trading volume was more than twice the average volume of the previous 20 trading days. |
| 🕒 Recency (R) | Number of calendar days since the most recent high-volume event. |
| 🔄 Frequency (F) | Number of high-volume days during the previous 60 trading sessions. |
| 💰 Monetary (M) | Estimated total trading value over the previous 60 trading sessions, using closing price × trading volume. |
| 🛠️ Data preparation | Reduced the effect of unusually large Monetary values and standardised the variables before clustering. |
| 🧩 Clustering | Used Ward agglomerative clustering and K-means, testing between 2 and 8 clusters. |
| 🎯 Cluster selection | Selected the number of clusters using silhouette scores, ensuring each cluster contained at least 5% of the shares. |
| 🔬 Sensitivity analysis | Examined relationships between RFM variables and repeated Ward clustering without Monetary to see how the results changed. |

### 🎯 Final Clustering Solutions

| Approach | Ward | K-means |
|---|---:|---:|
| Risk and Return | 5 clusters | 5 clusters |
| Adapted RFM | 3 clusters | 4 clusters |

---

## 🗂️ Repository Structure

```text
📦 CRM-CM-Coursework
│
├── 📓 sanjanan-20231816.ipynb
│   └── Main notebook containing the analysis
│
├── 📁 data/
│   ├── 📁 raw/                 Original saved data
│   └── 📁 processed/           Calculated variables and results
│
├── 📁 tables/                  Tables used in the coursework report
│
└── 📁 figures/                 Charts and visualisations
```

### 📑 Main Outputs

| Output | File |
|---|---|
| Excluded shares | `tables/exclusion_summary.csv` |
| Feature summaries | `tables/feature_summary.csv` |
| AAPL worked example | `tables/worked_example_AAPL_events.csv` |
| Feature distributions | `figures/f2_1_feature_distributions.png` |
| Cluster selection | `figures/f2_2_k_diagnostics.png` |
| Dendrograms | `figures/f2_3_dendrogram_A.png`, `figures/f2_3_dendrogram_B.png` |
| Cluster profiles | `tables/t2_profile_*.csv` |
| Clustering comparison | `tables/t2_algorithm_comparison.csv` |
| RFM correlations | `tables/t2_corr_pearson.csv`, `tables/t2_corr_spearman.csv` |
| Monetary sensitivity analysis | `tables/t2_monetary_removal_profiles.csv` |
| Sector analysis | `tables/t3_sector_share_*.csv` |
| Limitations analysis | `tables/t3_return_dependence.csv`, `tables/t3_adjusted_vs_close_betas.csv`, `tables/t3_survivorship_evidence.csv` |

---

## 🧰 Tools & Libraries

The analysis was completed using Python and the following libraries:

| Library | Purpose |
|---|---|
| Python | Main programming language |
| pandas | Organising and analysing data |
| NumPy | Mathematical calculations |
| yfinance | Downloading share price data |
| scikit-learn | Clustering and standardisation |
| SciPy | Statistical calculations and hierarchical clustering |
| Matplotlib | Creating charts and visualisations |

Exact software versions are recorded in `tables/package_versions.txt`.

---

## 📝 Data Note

Share returns were calculated using adjusted closing prices, which account for dividends and stock splits. However, the S&P 500 benchmark (^GSPC) is a price index and does not include dividends.

---

**Thank you for exploring this project!** 🎓
