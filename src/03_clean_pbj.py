#!/usr/bin/env python
# coding: utf-8
# ─────────────────────────────────────────────────────────────────────────────
# PBJ Nurse Staffing — Monthly Panel (HPPD for RN/LPN/CNA/Total)
#   * Reads pbj_nurse_YYYY_Q*.csv from RAW_DIR/pbj-nurse
#   * CCN normalization (keeps alphanumeric; zero-pad numeric to 6)
#   * Daily → Monthly aggregation
#       - resident_days = Σ daily MDS census within month
#       - *_hours_month = Σ daily hours within month
#       - *_hppd = *_hours_month / resident_days
#   * Coverage + gap tracking
#   * Saves:
#       - data/interim/pbj_monthly_panel.csv
#       - data/interim/pbj_monthly_coverage.csv
#       - data/interim/pbj_monthly_panel_analysis_ready.csv (optional filtered)
#   * Run with: %run 03_clean_pbj.py
# ─────────────────────────────────────────────────────────────────────────────

import os, warnings
from pathlib import Path
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

OUT_FP      = INTERIM_DIR / "pbj_monthly_panel.csv"
COV_FP      = INTERIM_DIR / "pbj_monthly_coverage.csv"
OUT_READY   = INTERIM_DIR / "pbj_monthly_panel_analysis_ready.csv"

# Keep raw monthly hours columns alongside HPPD? (set False to drop)
KEEP_HOUR_TOTALS = True
# Optional “analysis-ready” filter thresholds
READY_MIN_COVERAGE = 0.80
READY_MIN_RES_DAYS = 100

print(f"[paths] RAW_DIR={RAW_DIR}")
print(f"[paths] PBJ_DIR={PBJ_DIR}")
print(f"[paths] OUT_FP={OUT_FP}")
print(f"[paths] COV_FP={COV_FP}")
print(f"[paths] OUT_READY={OUT_READY}")

# ============================== Robust CSV Reader =============================
def read_csv_robust(fp: Path) -> pd.DataFrame:
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

# ============================== Helpers ======================================
def normalize_ccn_any(series: pd.Series) -> pd.Series:
    """
    Preserve alphanumeric CCNs; strip separators; pad only if numeric.
      - uppercase, remove spaces / - / / / .
      - if ALL digits → zfill(6)
      - if has letters → keep as-is
    """
    s = series.astype("string").fillna("").str.strip().str.upper()
    s = s.str.replace(r"[ \-\/\.]", "", regex=True)
    is_digits = s.str.fullmatch(r"\d+")
    s = s.mask(is_digits, s.str.zfill(6))
    s = s.replace({"": pd.NA})
    return s

def to_date_from_int_yyyymmdd(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("Int64"), format="%Y%m%d", errors="coerce")

# ============================== Normalization =================================
def normalize_needed_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # rename basics
    if "provnum" in df.columns and "cms_certification_number" not in df.columns:
        df.rename(columns={"provnum": "cms_certification_number"}, inplace=True)
    if "mdscensus" in df.columns and "mds_census" not in df.columns:
        df.rename(columns={"mdscensus": "mds_census"}, inplace=True)

    # ensure hour cols exist (fill missing with 0)
    for col in ["hrs_rn", "hrs_lpn", "hrs_cna"]:
        if col not in df.columns:
            df[col] = 0.0

    # CCN
    if "cms_certification_number" in df.columns:
        df["cms_certification_number"] = normalize_ccn_any(df["cms_certification_number"])
    else:
        raise ValueError("Missing cms_certification_number/provnum")

    # workdate
    if "workdate" in df.columns:
        if pd.api.types.is_integer_dtype(df["workdate"]) or pd.api.types.is_string_dtype(df["workdate"]):
            df["workdate"] = to_date_from_int_yyyymmdd(df["workdate"])
        else:
            df["workdate"] = pd.to_datetime(df["workdate"], errors="coerce")
    else:
        raise ValueError("Missing workdate column")

    # hours numeric
    for c in ["hrs_rn", "hrs_lpn", "hrs_cna"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32").fillna(0.0)

    # daily census numeric (optional column → we still need it for HPPD)
    if "mds_census" not in df.columns:
        df["mds_census"] = np.nan
    df["mds_census"] = pd.to_numeric(df["mds_census"], errors="coerce").astype("float32")

    keep = ["cms_certification_number","workdate","hrs_rn","hrs_lpn","hrs_cna","mds_census"]
    return df[keep]

# ====================== File → Monthly Aggregation ============================
def process_file_monthly(fp: Path) -> pd.DataFrame:
    """
    Build monthly totals per CCN from one PBJ file.
    HPPD = monthly sum of hours / monthly sum of daily MDS census (resident-days).
    """
    df = normalize_needed_columns(read_csv_robust(fp))

    # 1) Daily totals per CCN×date (sum hours, mean census for that day if multiple rows)
    daily = (df.groupby(["cms_certification_number","workdate"], as_index=False)
               .agg(hrs_rn=("hrs_rn","sum"),
                    hrs_lpn=("hrs_lpn","sum"),
                    hrs_cna=("hrs_cna","sum"),
                    mds_census=("mds_census","mean")))
    # Derived totals
    daily["total_hours"] = daily[["hrs_rn","hrs_lpn","hrs_cna"]].sum(axis=1).astype("float32")
    daily["year_month"]  = daily["workdate"].dt.to_period("M")
    daily["days_in_month"] = daily["workdate"].dt.days_in_month

    # 2) Monthly aggregation:
    #    - *_hours_month: Σ hours across all days in that month
    #    - resident_days: Σ daily census (denominator)
    #    - coverage_ratio: (# distinct workdates reported) / (days_in_month)
    monthly = (daily.groupby(["cms_certification_number","year_month"], as_index=False)
                    .agg(rn_hours_month=("hrs_rn","sum"),
                         lpn_hours_month=("hrs_lpn","sum"),
                         cna_hours_month=("hrs_cna","sum"),
                         total_hours=("total_hours","sum"),
                         resident_days=("mds_census","sum"),
                         avg_daily_census=("mds_census","mean"),
                         days_reported=("workdate","nunique"),
                         days_in_month=("days_in_month","max")))

    monthly["coverage_ratio"] = monthly["days_reported"] / monthly["days_in_month"]

    # 3) HPPD (guard against 0 / NaN resident_days)
    denom = monthly["resident_days"].replace({0: np.nan})
    monthly["rn_hppd"]    = monthly["rn_hours_month"]  / denom
    monthly["lpn_hppd"]   = monthly["lpn_hours_month"] / denom
    monthly["cna_hppd"]   = monthly["cna_hours_month"] / denom
    monthly["total_hppd"] = monthly["total_hours"]     / denom

    # 4) Labels + types
    monthly["month"] = monthly["year_month"].dt.strftime("%m/%Y")

    # Dtypes
    float_cols = [
        "rn_hours_month","lpn_hours_month","cna_hours_month","total_hours",
        "resident_days","avg_daily_census",
        "rn_hppd","lpn_hppd","cna_hppd","total_hppd",
        "coverage_ratio",
    ]
    for c in float_cols:
        monthly[c] = pd.to_numeric(monthly[c], errors="coerce").astype("float32")

    monthly["days_reported"] = monthly["days_reported"].astype("Int16")
    monthly["days_in_month"] = monthly["days_in_month"].astype("Int16")

    # Optionally drop raw hour totals if you don't want them in final file
    if not KEEP_HOUR_TOTALS:
        monthly.drop(columns=["rn_hours_month","lpn_hours_month","cna_hours_month","total_hours"], inplace=True)

    return monthly

# ============================== Coverage / Gaps ===============================
def add_gap_tracking(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add per-row gap_from_prev_months and provider-level coverage summary."""
    if df.empty:
        cov = pd.DataFrame(columns=[
            "cms_certification_number","first_month","last_month",
            "months_observed","months_expected","months_missing","has_gaps"
        ])
        return df, cov

    df = df.copy()
    ym = df["year_month"].astype("period[M]")
    df["month_index"] = (ym.dt.year.astype(int) * 12 + ym.dt.month.astype(int)).astype("Int32")

    df = df.sort_values(["cms_certification_number","month_index"])
    df["gap_from_prev_months"] = df.groupby("cms_certification_number")["month_index"].diff().fillna(1).astype("Int16") - 1
    df["gap_from_prev_months"] = df["gap_from_prev_months"].clip(lower=0)

    g = df.groupby("cms_certification_number", as_index=False)
    cov = g.agg(
        first_idx=("month_index","min"),
        last_idx =("month_index","max"),
        months_observed=("month_index","nunique"),
        any_gaps=("gap_from_prev_months", lambda s: bool((s > 0).any()))
    )
    cov["months_expected"] = (cov["last_idx"] - cov["first_idx"] + 1).astype("Int32")
    cov["months_missing"]  = (cov["months_expected"] - cov["months_observed"]).astype("Int32")
    cov["has_gaps"] = cov["any_gaps"].astype(bool)
    cov.drop(columns=["any_gaps"], inplace=True)

    # pretty month labels from actual periods
    first_last = df.groupby("cms_certification_number").agg(
        first_month=("year_month","min"),
        last_month =("year_month","max")
    ).reset_index()
    cov = cov.merge(first_last, on="cms_certification_number", how="left")
    cov["first_month"] = cov["first_month"].dt.strftime("%Y-%m")
    cov["last_month"]  = cov["last_month"].dt.strftime("%Y-%m")
    cov = cov[["cms_certification_number","first_month","last_month",
               "months_observed","months_expected","months_missing","has_gaps"]]
    return df, cov

# ============================== Quick Summary =================================
def print_quick_summary(monthly: pd.DataFrame, cov: pd.DataFrame):
    try:
        n_rows = len(monthly)
        n_ccn  = monthly["cms_certification_number"].nunique() if not monthly.empty else 0
        gaps   = monthly.get("gap_from_prev_months")
        covr   = monthly.get("coverage_ratio")

        print("\n[summary]")
        print(f"rows={n_rows:,}  |  CCNs={n_ccn:,}")
        if gaps is not None and len(gaps):
            g = gaps.fillna(0).astype("int32")
            print(f"avg gap={g.mean():.2f}  |  median gap={g.median():.0f}  |  share rows gap>0={(g>0).mean():.2%}")
        if covr is not None and len(covr):
            print(f"avg coverage={covr.mean():.3f}  |  median coverage={covr.median():.3f}  |  share coverage<0.5={(covr<0.5).mean():.2%}")

        for k in ["rn_hppd","lpn_hppd","cna_hppd","total_hppd"]:
            if k in monthly.columns:
                desc = monthly[k].describe(percentiles=[.01,.05,.5,.95,.99])
                print(f"\n[{k}]")
                print(desc)
    except Exception as e:
        print(f"[warn] quick summary failed: {e}")

# ============================== Main ==========================================
def main():
    files = sorted(PBJ_DIR.glob(PBJ_GLOB))
    print(f"[scan] {len(files)} files found")

    frames = []
    for fp in files:
        try:
            m = process_file_monthly(fp)
            print(f"[ok] {fp.name}: {len(m):,} rows")
            if not m.empty:
                frames.append(m)
        except Exception as e:
            print(f"[fail] {fp.name}: {e}")

    monthly = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"[concat] monthly rows = {len(monthly):,}")

    # Gaps/coverage tracking
    monthly, cov = add_gap_tracking(monthly)

    # Final order & CSV-friendly period
    if not monthly.empty:
        # Keep consistent field order
        base_cols = [
            "cms_certification_number","month","year_month","month_index","gap_from_prev_months",
            "resident_days","avg_daily_census",
            "rn_hppd","lpn_hppd","cna_hppd","total_hppd",
            "days_reported","days_in_month","coverage_ratio",
        ]
        if KEEP_HOUR_TOTALS:
            base_cols = ([
                "cms_certification_number","month","year_month","month_index","gap_from_prev_months",
                "rn_hours_month","lpn_hours_month","cna_hours_month","total_hours",
                "resident_days","avg_daily_census",
                "rn_hppd","lpn_hppd","cna_hppd","total_hppd",
                "days_reported","days_in_month","coverage_ratio",
            ])
        # Coerce year_month to str for CSV stability
        monthly["year_month"] = monthly["year_month"].astype("period[M]").astype(str)
        monthly = monthly[base_cols]

    # Save
    monthly.to_csv(OUT_FP, index=False)
    cov.to_csv(COV_FP, index=False)
    print(f"[saved] panel    → {OUT_FP}")
    print(f"[saved] coverage → {COV_FP}")

    # Optional analysis-ready export (coverage + resident-days screens)
    if not monthly.empty:
        m2 = monthly.copy()
        m2 = m2[(m2["coverage_ratio"] >= READY_MIN_COVERAGE) &
                (pd.to_numeric(m2["resident_days"], errors="coerce") >= READY_MIN_RES_DAYS)]
        # Require finite total_hppd
        m2 = m2[np.isfinite(m2["total_hppd"].astype(float))]
        m2.to_csv(OUT_READY, index=False)
        print(f"[saved] analysis-ready → {OUT_READY}  (rows={len(m2):,})")

    # Summary
    print_quick_summary(monthly, cov)

# ============================== Script Entry ==================================
if __name__ == "__main__":
    main()