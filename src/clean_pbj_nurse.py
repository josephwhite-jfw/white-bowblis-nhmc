#!/usr/bin/env python
# coding: utf-8

import os, re, warnings
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

OUT_FP = INTERIM_DIR / "pbj_nurse.csv"
KEEP_HOUR_TOTALS = True

print(f"[paths] RAW_DIR={RAW_DIR}")
print(f"[paths] PBJ_DIR={PBJ_DIR}")
print(f"[paths] OUT_FP={OUT_FP}")

# ============================== IO ===========================================
def read_csv_robust(fp: Path) -> pd.DataFrame:
    for enc in ("utf-8","utf-8-sig","cp1252","latin1"):
        try:
            return pd.read_csv(fp, low_memory=False, encoding=enc, encoding_errors="strict")
        except Exception:
            pass
    return pd.read_csv(fp, low_memory=False, encoding="latin1",
                       encoding_errors="replace", on_bad_lines="skip")

# ============================== Helpers ======================================
def normalize_ccn_any(s: pd.Series) -> pd.Series:
    s = s.astype("string").fillna("").str.strip().str.upper()
    s = s.str.replace(r"[ \-\/\.]", "", regex=True)
    is_digits = s.str.fullmatch(r"\d+")
    s = s.mask(is_digits, s.str.zfill(6)).replace({"": pd.NA})
    return s

def to_date_from_int_yyyymmdd(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("Int64"), format="%Y%m%d", errors="coerce")

# -------- fast, vectorized CY_QTR parsing ----------
_QRX = re.compile(r"(?i)(?:CY)?\s*(20\d{2})?\s*[- ]?Q(?:TR)?\s*([1-4])|^\s*([1-4])\s*$")
def normalize_cy_qtr(cy_qtr: pd.Series, workdate: pd.Series) -> pd.Series:
    s = cy_qtr.astype("string")
    m = s.str.extract(_QRX)
    y = pd.to_numeric(m[0], errors="coerce").astype("Int64")
    q = pd.to_numeric(m[1].fillna(m[2]), errors="coerce").astype("Int64")
    y = y.fillna(workdate.dt.year.astype("Int64"))
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    mask = y.notna() & q.notna()
    out.loc[mask] = y[mask].astype(str) + "Q" + q[mask].astype(str)
    still = out.isna()
    if still.any():
        qn = ((workdate.dt.month - 1)//3 + 1).astype("Int64")
        out.loc[still] = workdate.dt.year.astype("Int64").astype(str) + "Q" + qn.astype(str)
    return out

# ============================== Normalization =================================
def normalize_needed_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "provnum" in df.columns and "cms_certification_number" not in df.columns:
        df.rename(columns={"provnum": "cms_certification_number"}, inplace=True)
    if "mdscensus" in df.columns and "mds_census" not in df.columns:
        df.rename(columns={"mdscensus": "mds_census"}, inplace=True)

    for col in ["hrs_rn", "hrs_lpn", "hrs_cna"]:
        if col not in df.columns:
            df[col] = 0.0

    if "cms_certification_number" not in df.columns:
        raise ValueError("Missing cms_certification_number/provnum")
    df["cms_certification_number"] = normalize_ccn_any(df["cms_certification_number"])

    if "workdate" not in df.columns:
        raise ValueError("Missing workdate column")
    if pd.api.types.is_integer_dtype(df["workdate"]) or pd.api.types.is_string_dtype(df["workdate"]):
        df["workdate"] = to_date_from_int_yyyymmdd(df["workdate"])
    else:
        df["workdate"] = pd.to_datetime(df["workdate"], errors="coerce")

    for c in ["hrs_rn", "hrs_lpn", "hrs_cna"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32").fillna(0.0)

    if "mds_census" not in df.columns:
        df["mds_census"] = np.nan
    df["mds_census"] = pd.to_numeric(df["mds_census"], errors="coerce").astype("float32")

    if "cy_qtr" not in df.columns:
        df["cy_qtr"] = pd.NA

    return df[["cms_certification_number","workdate","hrs_rn","hrs_lpn","hrs_cna","mds_census","cy_qtr"]]

# ====================== File → Monthly Aggregation ============================
def process_file_monthly(fp: Path) -> pd.DataFrame:
    df = normalize_needed_columns(read_csv_robust(fp))
    df["quarter_row"] = normalize_cy_qtr(df["cy_qtr"], df["workdate"])

    # Daily
    daily = (df.groupby(["cms_certification_number","workdate"], as_index=False)
               .agg(hrs_rn=("hrs_rn","sum"),
                    hrs_lpn=("hrs_lpn","sum"),
                    hrs_cna=("hrs_cna","sum"),
                    mds_census=("mds_census","mean"),
                    quarter=("quarter_row","first")))

    daily["total_hours"]  = daily[["hrs_rn","hrs_lpn","hrs_cna"]].sum(axis=1).astype("float32")
    daily["year_month_p"] = daily["workdate"].dt.to_period("M")
    daily["days_in_mo"]   = daily["workdate"].dt.days_in_month

    # Monthly
    monthly = (daily.groupby(["cms_certification_number","year_month_p"], as_index=False)
                    .agg(rn_hours_month=("hrs_rn","sum"),
                         lpn_hours_month=("hrs_lpn","sum"),
                         cna_hours_month=("hrs_cna","sum"),
                         total_hours=("total_hours","sum"),
                         resident_days=("mds_census","sum"),
                         avg_daily_census=("mds_census","mean"),
                         days_reported=("workdate","nunique"),
                         days_in_month=("days_in_mo","max"),
                         quarter=("quarter","first")))

    monthly["coverage_ratio"] = monthly["days_reported"] / monthly["days_in_month"]

    denom = monthly["resident_days"].replace({0: np.nan})
    monthly["rn_hppd"]    = monthly["rn_hours_month"]  / denom
    monthly["lpn_hppd"]   = monthly["lpn_hours_month"] / denom
    monthly["cna_hppd"]   = monthly["cna_hours_month"] / denom
    monthly["total_hppd"] = monthly["total_hours"]     / denom

    # year_month as 'YYYY/MM'
    ym = monthly["year_month_p"].astype("period[M]")
    monthly["year_month"] = ym.dt.year.astype(int).astype(str) + "/" + ym.dt.month.astype(int).astype(str).str.zfill(2)

    # ---- gap_from_prev_months (vectorized) ----
    month_index = (ym.dt.year.astype(int) * 12 + ym.dt.month.astype(int)).astype("Int32")
    monthly = monthly.assign(_month_index=month_index)
    monthly = monthly.sort_values(["cms_certification_number","_month_index"], kind="mergesort")
    monthly["gap_from_prev_months"] = (
        monthly.groupby("cms_certification_number")["_month_index"]
               .diff()
               .fillna(1)
               .astype("Int16") - 1
    ).clip(lower=0)

    # Casts
    for c in ["rn_hours_month","lpn_hours_month","cna_hours_month","total_hours",
              "resident_days","avg_daily_census","rn_hppd","lpn_hppd","cna_hppd","total_hppd","coverage_ratio"]:
        monthly[c] = pd.to_numeric(monthly[c], errors="coerce").astype("float32")
    monthly["days_reported"] = monthly["days_reported"].astype("Int16")
    monthly["days_in_month"] = monthly["days_in_month"].astype("Int16")

    # Final ordering (sorted by CCN, then month)
    monthly = monthly.sort_values(["cms_certification_number","_month_index"], kind="mergesort")

    # Drop temp
    monthly = monthly.drop(columns=["year_month_p","_month_index"])

    return monthly

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

    if monthly.empty:
        monthly.to_csv(OUT_FP, index=False)
        print(f"[saved] pbj nurse panel → {OUT_FP} (rows=0)")
        return

    # Keep only the columns we want to save
    cols = [
        "cms_certification_number", "quarter", "year_month",
        *(["rn_hours_month","lpn_hours_month","cna_hours_month","total_hours"] if KEEP_HOUR_TOTALS else []),
        "resident_days","avg_daily_census",
        "rn_hppd","lpn_hppd","cna_hppd","total_hppd",
        "days_reported","days_in_month","coverage_ratio",
    ]
    monthly = monthly[cols]

    # ---------- GLOBAL sort + gap computation (across all files) ----------
    # Parse YYYY/MM to a real month key
    ord_dt = pd.to_datetime(monthly["year_month"] + "/01", format="%Y/%m/%d", errors="coerce")
    monthly = monthly.assign(_ord=ord_dt,
                             _mi=(ord_dt.dt.year*12 + ord_dt.dt.month).astype("Int32"))

    # Sort globally by CCN, then month
    monthly = monthly.sort_values(["cms_certification_number","_ord"], kind="mergesort")

    # gap_from_prev_months: 0 if consecutive month; otherwise size of the gap
    monthly["gap_from_prev_months"] = (
        monthly.groupby("cms_certification_number")["_mi"]
               .diff()               # difference in month index
               .fillna(1)
               .astype("Int16") - 1   # convert consecutive (diff=1) → 0
    ).clip(lower=0)

    # Drop helpers and ensure final sorted order
    monthly = monthly.drop(columns=["_ord","_mi"]) \
                     .sort_values(["cms_certification_number","year_month"], kind="mergesort") \
                     .reset_index(drop=True)
    # ----------------------------------------------------------------------

    monthly.to_csv(OUT_FP, index=False)
    print(f"[saved] pbj nurse panel → {OUT_FP} (rows={len(monthly):,})")

if __name__ == "__main__":
    main()