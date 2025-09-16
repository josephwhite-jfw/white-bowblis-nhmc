#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ─────────────────────────────────────────────────────────────────────────────
# PBJ Nurse Staffing — Build Monthly Panel
#   * Reads all quarterly PBJ nurse CSVs (pbj_nurse_yyyy_Q*.csv)
#   * Cleans CCNs, dates, hours, and census
#   * Daily → monthly totals with coverage + IQR outlier filtering
#   * Saves one combined monthly panel CSV in data/interim
# ─────────────────────────────────────────────────────────────────────────────

import os, warnings
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================== Paths / Config ================================
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").is_dir() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

RAW_DIR  = Path(os.getenv("NH_DATA_DIR", PROJECT_ROOT / "data" / "raw")).resolve()
PBJ_DIR  = RAW_DIR / "pbj-nurse"
PBJ_GLOB = "pbj_nurse_????_Q[1-4].csv"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
OUT_FP = INTERIM_DIR / "pbj_monthly_panel.csv"

print(f"[paths] RAW_DIR={RAW_DIR}")
print(f"[paths] PBJ_DIR={PBJ_DIR}")
print(f"[paths] OUT_FP={OUT_FP}")

# ============================== CSV Reader ====================================
def read_csv_robust(fp: Path) -> pd.DataFrame:
    """Try multiple encodings. Fall back to replace/skip if needed."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(fp, low_memory=False, encoding=enc, encoding_errors="strict")
        except Exception as e:
            last_err = e
    for enc in ["cp1252", "latin1"]:
        try:
            return pd.read_csv(fp, low_memory=False, encoding=enc,
                               encoding_errors="replace", on_bad_lines="skip")
        except Exception as e:
            last_err = e
    raise last_err

# ============================== Helpers =======================================
def zero_pad_ccn(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.zfill(6)

def to_date_from_int_yyyymmdd(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("Int64"), format="%Y%m%d", errors="coerce")

def normalize_needed_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean one PBJ quarterly file down to the columns we need."""
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # rename basics
    if "provnum" in df.columns and "cms_certification_number" not in df.columns:
        df.rename(columns={"provnum": "cms_certification_number"}, inplace=True)
    if "mdscensus" in df.columns and "mds_census" not in df.columns:
        df.rename(columns={"mdscensus": "mds_census"}, inplace=True)

    # add missing hour cols if absent
    for col in ["hrs_rn", "hrs_lpn", "hrs_cna"]:
        if col not in df.columns:
            df[col] = 0.0

    # CCN
    if "cms_certification_number" in df.columns:
        df["cms_certification_number"] = zero_pad_ccn(df["cms_certification_number"])
    else:
        warnings.warn("Missing cms_certification_number/provnum")

    # workdate
    if "workdate" in df.columns:
        if pd.api.types.is_integer_dtype(df["workdate"]) or pd.api.types.is_string_dtype(df["workdate"]):
            df["workdate"] = to_date_from_int_yyyymmdd(df["workdate"])
        else:
            df["workdate"] = pd.to_datetime(df["workdate"], errors="coerce")
    else:
        raise ValueError("Missing workdate column")

    # hours numeric, float32
    for c in ["hrs_rn", "hrs_lpn", "hrs_cna"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32").fillna(0.0)

    # census optional
    if "mds_census" not in df.columns:
        df["mds_census"] = np.nan
    df["mds_census"] = pd.to_numeric(df["mds_census"], errors="coerce").astype("float32")

    keep = ["cms_certification_number","workdate","hrs_rn","hrs_lpn","hrs_cna","mds_census"]
    return df[keep]

# ====================== File → Monthly Aggregation ============================
def process_file_monthly(fp: Path,
                         coverage_threshold: float = 0.5,
                         iqr_mult: float = 1.5) -> pd.DataFrame:
    """
    Build monthly totals per CCN from one file.
    Coverage: drop months with <50% of days reported (default).
    Outliers: drop daily values outside IQR × multiplier (default 1.5).
    """
    df = normalize_needed_columns(read_csv_robust(fp))

    # daily totals
    daily = (df.groupby(["cms_certification_number","workdate"], as_index=False)
               .agg(hrs_rn=("hrs_rn","sum"),
                    hrs_lpn=("hrs_lpn","sum"),
                    hrs_cna=("hrs_cna","sum"),
                    mds_census=("mds_census","mean")))
    daily["total_hours"] = daily[["hrs_rn","hrs_lpn","hrs_cna"]].sum(axis=1).astype("float32")
    daily["year_month"]  = daily["workdate"].dt.to_period("M")
    daily["days_in_month"] = daily["workdate"].dt.days_in_month

    # coverage filter
    cov = (daily.groupby(["cms_certification_number","year_month"], as_index=False)
                 .agg(days_reported=("workdate","nunique"),
                      days_in_month=("days_in_month","max")))
    cov["coverage_ratio"] = cov["days_reported"] / cov["days_in_month"]
    cov_ok = cov.loc[cov["coverage_ratio"] >= coverage_threshold,
                     ["cms_certification_number","year_month","days_reported"]]
    if cov_ok.empty:
        return pd.DataFrame(columns=["cms_certification_number","month",
                                     "hrs_rn","hrs_lpn","hrs_cna","total_hours","mds_census",
                                     "hrs_rn_per_patient","hrs_lpn_per_patient",
                                     "hrs_cna_per_patient","total_hours_per_patient","n"])

    good = daily.merge(cov_ok, on=["cms_certification_number","year_month"], how="inner")

    # outlier bounds (IQR)
    KEYS = ["cms_certification_number","year_month"]
    stats = (good.groupby(KEYS)
                 .agg(rn_q1=('hrs_rn', lambda s: s.quantile(0.25)),
                      rn_q3=('hrs_rn', lambda s: s.quantile(0.75)),
                      lpn_q1=('hrs_lpn', lambda s: s.quantile(0.25)),
                      lpn_q3=('hrs_lpn', lambda s: s.quantile(0.75)),
                      cna_q1=('hrs_cna', lambda s: s.quantile(0.25)),
                      cna_q3=('hrs_cna', lambda s: s.quantile(0.75)),
                      tot_q1=('total_hours', lambda s: s.quantile(0.25)),
                      tot_q3=('total_hours', lambda s: s.quantile(0.75)))
                 .reset_index())
    for pref in ["rn","lpn","cna","tot"]:
        q1, q3 = f"{pref}_q1", f"{pref}_q3"
        stats[f"{pref}_iqr"] = stats[q3] - stats[q1]
        stats[f"{pref}_lo"]  = stats[q1] - iqr_mult * stats[f"{pref}_iqr"]
        stats[f"{pref}_hi"]  = stats[q3] + iqr_mult * stats[f"{pref}_iqr"]
        z = stats[f"{pref}_iqr"] == 0
        stats.loc[z, f"{pref}_lo"] = stats.loc[z, q1]
        stats.loc[z, f"{pref}_hi"] = stats.loc[z, q3]

    bounds = stats[KEYS + [f"{p}_{b}" for p in ["rn","lpn","cna","tot"] for b in ["lo","hi"]]]
    good = good.merge(bounds, on=KEYS, how="left")

    is_outlier = (
        (good["hrs_rn"]  < good["rn_lo"])  | (good["hrs_rn"]  > good["rn_hi"])  |
        (good["hrs_lpn"] < good["lpn_lo"]) | (good["hrs_lpn"] > good["lpn_hi"]) |
        (good["hrs_cna"] < good["cna_lo"]) | (good["hrs_cna"] > good["cna_hi"]) |
        (good["total_hours"] < good["tot_lo"]) | (good["total_hours"] > good["tot_hi"])
    )
    kept = good.loc[~is_outlier,
                    ["cms_certification_number","year_month","hrs_rn","hrs_lpn","hrs_cna",
                     "total_hours","mds_census","days_reported"]]

    # monthly totals
    monthly = (kept.groupby(["cms_certification_number","year_month"], as_index=False)
                    .agg(hrs_rn=("hrs_rn","sum"),
                         hrs_lpn=("hrs_lpn","sum"),
                         hrs_cna=("hrs_cna","sum"),
                         total_hours=("total_hours","sum"),
                         mds_census=("mds_census","mean"),
                         n=("days_reported","max")))

    # per-patient metrics
    denom = monthly["mds_census"].replace({0: np.nan})
    monthly["hrs_rn_per_patient"]      = monthly["hrs_rn"]      / denom
    monthly["hrs_lpn_per_patient"]     = monthly["hrs_lpn"]     / denom
    monthly["hrs_cna_per_patient"]     = monthly["hrs_cna"]     / denom
    monthly["total_hours_per_patient"] = monthly["total_hours"] / denom

    # month label (MM/YYYY)
    monthly["month"] = monthly["year_month"].dt.strftime("%m/%Y")

    # final order + dtypes
    monthly = monthly[["cms_certification_number","month",
                       "hrs_rn","hrs_lpn","hrs_cna","total_hours","mds_census",
                       "hrs_rn_per_patient","hrs_lpn_per_patient","hrs_cna_per_patient",
                       "total_hours_per_patient","n"]]
    for c in ["hrs_rn","hrs_lpn","hrs_cna","total_hours","mds_census",
              "hrs_rn_per_patient","hrs_lpn_per_patient","hrs_cna_per_patient","total_hours_per_patient"]:
        monthly[c] = monthly[c].astype("float32")
    monthly["n"] = monthly["n"].astype("Int16")
    return monthly

# ============================== Main Runner ===================================
def main(coverage_threshold: float = 0.5, iqr_mult: float = 1.5):
    files = sorted(PBJ_DIR.glob(PBJ_GLOB))
    print(f"[scan] {len(files)} files found")

    frames = []
    for fp in files:
        try:
            m = process_file_monthly(fp, coverage_threshold=coverage_threshold, iqr_mult=iqr_mult)
            print(f"[ok] {fp.name}: {len(m):,} rows")
            if not m.empty:
                frames.append(m)
        except Exception as e:
            print(f"[fail] {fp.name}: {e}")

    monthly_panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"[done] monthly_panel rows = {len(monthly_panel):,}")

    monthly_panel.to_csv(OUT_FP, index=False)
    print(f"[saved] {OUT_FP}")

# ============================== Script Entry ==================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PBJ monthly staffing panel")
    parser.add_argument("--coverage_threshold", type=float, default=0.5,
                        help="Min days reported / days in month (default 0.5)")
    parser.add_argument("--iqr_mult", type=float, default=1.5,
                        help="Multiplier for IQR outlier filtering (default 1.5)")
    args = parser.parse_args()

    main(coverage_threshold=args.coverage_threshold, iqr_mult=args.iqr_mult)

