-- Star schema for the S&P 500 share-segmentation data mart (SQLite dialect).
-- Facts are append-only: a re-run creates a NEW dim_run row, old rows are never overwritten.

CREATE TABLE dim_run (            -- one row per calculation run = provenance + versions
  run_key INTEGER PRIMARY KEY, created_utc TEXT, calc_version TEXT, git_commit TEXT,
  source_name TEXT, constituents_retrieved_utc TEXT, study_start TEXT, study_end_exclusive TEXT,
  observation_date TEXT, baseline_sessions INTEGER, window_sessions INTEGER, volume_multiplier REAL,
  random_seed INTEGER, python_version TEXT, pandas_version TEXT, yfinance_version TEXT, sklearn_version TEXT);

CREATE TABLE dim_date (           -- one row per calendar date in the study window
  date_key INTEGER PRIMARY KEY,   -- YYYYMMDD
  full_date TEXT, year INTEGER, month INTEGER, day_of_week TEXT,
  week_ending_friday TEXT, is_trading_day INTEGER, is_complete_week INTEGER);

CREATE TABLE dim_sector (         -- GICS classification (kept separate so history can be versioned)
  sector_key INTEGER PRIMARY KEY, gics_sector TEXT, gics_sub_industry TEXT);

CREATE TABLE dim_share (          -- Type 2 slowly changing: new row if ticker/sector/membership changes
  share_key INTEGER PRIMARY KEY, ticker TEXT, company TEXT, cik TEXT, sector_key INTEGER REFERENCES dim_sector,
  date_added_to_index TEXT, valid_from TEXT, valid_to TEXT, is_current INTEGER);

CREATE TABLE dim_benchmark (
  benchmark_key INTEGER PRIMARY KEY, symbol TEXT, name TEXT, price_basis TEXT);

CREATE TABLE fact_daily_price (   -- GRAIN: one share x one trading day (per run)
  run_key INTEGER REFERENCES dim_run, date_key INTEGER REFERENCES dim_date, share_key INTEGER REFERENCES dim_share,
  open REAL, high REAL, low REAL, close REAL, adj_close REAL, volume REAL,
  traded_value_usd REAL,          -- close * volume (derived fact)
  baseline_mean_vol_20 REAL, event_threshold REAL, is_high_volume_event INTEGER,
  PRIMARY KEY (run_key, date_key, share_key));

CREATE TABLE fact_weekly_return ( -- GRAIN: one share x one Friday-ending week
  run_key INTEGER REFERENCES dim_run, week_end_date_key INTEGER REFERENCES dim_date,
  share_key INTEGER REFERENCES dim_share, benchmark_key INTEGER REFERENCES dim_benchmark,
  weekly_return REAL, benchmark_weekly_return REAL,
  PRIMARY KEY (run_key, week_end_date_key, share_key));

CREATE TABLE fact_share_feature ( -- GRAIN: one share x observation date x run
  run_key INTEGER REFERENCES dim_run, observation_date_key INTEGER REFERENCES dim_date,
  share_key INTEGER REFERENCES dim_share, benchmark_key INTEGER REFERENCES dim_benchmark,
  mean_weekly_return REAL, weekly_beta REAL, beta_std_error REAL, n_weeks INTEGER,
  recency_days INTEGER, frequency_events INTEGER, monetary_usd REAL, last_event_date TEXT,
  PRIMARY KEY (run_key, observation_date_key, share_key));

CREATE TABLE fact_cluster_assignment ( -- GRAIN: one share x run x feature matrix x algorithm
  run_key INTEGER REFERENCES dim_run, share_key INTEGER REFERENCES dim_share,
  feature_matrix TEXT,            -- 'A_risk_return' | 'B_rfm' | 'B_rfm_RF_only'
  algorithm TEXT, k INTEGER, cluster_id INTEGER,
  PRIMARY KEY (run_key, share_key, feature_matrix, algorithm));
