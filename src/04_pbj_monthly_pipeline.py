# 04_pbj_monthly_pipeline.py
"""
Build a monthly PBJ panel from the 'pbj-nurse' DAILY files.

Expected raw (one CSV per quarter) in PBJ_RAW_DIR with columns like:
  provnum, workdate, mdscensus,
  hrs_rn*, hrs_rndon*, hrs_rnadmin*,
  hrs_lpn*, hrs_lpnadmin*,
  hrs_cna*

We:
  • Read each quarterly daily CSV (robust encoding fallback)
  • Normalize CCN, parse dates, coerce numerics
  • Sum RN/LPN/CNA across employee/contractor/admin/DON buckets
  • Aggregate to CCN×month
  • Compute resident_days, coverage_ratio, HPPD metrics, gap_from_prev_months
  • Save one monthly panel CSV at PBJ_MONTHLY_CSV
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from config import PBJ_RAW_DIR, PBJ_MONTHLY_CSV


# ----------------------------- Utils -----------------------------------------

def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Try several encodings, last resort replace errors."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, dtype=str, encoding="utf-8",
                       encoding_errors="replace", low_memory=False)


def _normalize_ccn(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.upper().str.replace(r"[^\dA-Z]", "", regex=True)
    is_digits = s.str.fullmatch(r"\d+")
    return s.mask(is_digits, s.str.zfill(6))


def _sum_found(df: pd.DataFrame, candidates: set[str]) -> pd.Series:
    """Sum any columns from `candidates` that exist in df (coerced numeric)."""
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return pd.Series(0.0, index=df.index)
    vals = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in cols})
    return vals.sum(axis=1, skipna=True)


# Buckets to add up per discipline (lowercase colnames)
RN_SETS = [
    {"hrs_rn", "hrs_rn_emp", "hrs_rn_ctr"},
    {"hrs_rnadmin", "hrs_rnadmin_emp", "hrs_rnadmin_ctr"},
    {"hrs_rndon", "hrs_rndon_emp", "hrs_rndon_ctr"},
]
LPN_SETS = [
    {"hrs_lpn", "hrs_lpn_emp", "hrs_lpn_ctr", "hrs_lvn", "hrs_lvn_emp", "hrs_lvn_ctr"},
    {"hrs_lpnadmin", "hrs_lpnadmin_emp", "hrs_lpnadmin_ctr"},
]
CNA_SETS = [
    {"hrs_cna", "hrs_cna_emp", "hrs_cna_ctr"},
]


# ------------------------ Core parser/aggregator ------------------------------

def parse_pbj_nurse_daily_to_monthly(path: Path) -> pd.DataFrame | None:
    """
    Handle pbj_nurse_YYYY_Q*.csv daily schema → monthly CCN aggregates.
    Returns a DataFrame with columns:
      ['cms_certification_number','month','rn_hours_month','lpn_hours_month',
       'cna_hours_month','resident_days','days_reported','days_in_month',
       'coverage_ratio','total_hours','avg_daily_census',
       'rn_hppd','lpn_hppd','cna_hppd','total_hppd','gap_from_prev_months']
    """
    df = _safe_read_csv(path)
    if df is None or df.empty:
        return None

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # minimal daily schema check
    required = {"provnum", "workdate", "mdscensus"}
    if not required.issubset(df.columns):
        # not a pbj-nurse daily file
        return None

    # coerce basics
    df["cms_certification_number"] = _normalize_ccn(df["provnum"])
    df["date"] = pd.to_datetime(df["workdate"], errors="coerce")
    df["resident_days"] = pd.to_numeric(df["mdscensus"], errors="coerce")

    # sum hour buckets that exist in this file
    rn = sum((_sum_found(df, s) for s in RN_SETS), start=pd.Series(0.0, index=df.index))
    lpn = sum((_sum_found(df, s) for s in LPN_SETS), start=pd.Series(0.0, index=df.index))
    cna = sum((_sum_found(df, s) for s in CNA_SETS), start=pd.Series(0.0, index=df.index))

    df["_rn_hours"] = rn
    df["_lpn_hours"] = lpn
    df["_cna_hours"] = cna

    # drop rows lacking core fields
    df = df.dropna(subset=["cms_certification_number", "date"])

    # roll up to month
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp("s")
    g = df.groupby(["cms_certification_number", "month"], as_index=False).agg(
        rn_hours_month=("._rn_hours".lstrip("."), "sum") if "._rn_hours" in df.columns else ("_rn_hours", "sum"),
        lpn_hours_month=("_lpn_hours", "sum"),
        cna_hours_month=("_cna_hours", "sum"),
        resident_days=("resident_days", "sum"),
        days_reported=("date", "nunique"),
    )

    # monthly derived metrics
    g["days_in_month"] = g["month"].dt.daysinmonth
    g["coverage_ratio"] = np.where(g["days_in_month"] > 0,
                                   g["days_reported"] / g["days_in_month"], np.nan)
    g["total_hours"] = g["rn_hours_month"] + g["lpn_hours_month"] + g["cna_hours_month"]
    g["avg_daily_census"] = g["resident_days"] / g["days_in_month"]

    den = g["resident_days"].replace({0: np.nan})
    g["rn_hppd"] = g["rn_hours_month"] / den
    g["lpn_hppd"] = g["lpn_hours_month"] / den
    g["cna_hppd"] = g["cna_hours_month"] / den
    g["total_hppd"] = g["total_hours"] / den

    # continuity metric (gap from previous month in ~months)
    g = g.sort_values(["cms_certification_number", "month"], kind="mergesort")
    g["gap_from_prev_months"] = (
        g.groupby("cms_certification_number")["month"]
         .diff().dt.days.fillna(0).astype(int).div(30).round().astype(int)
    )

    return g


# ----------------------------- Orchestrator -----------------------------------

def build_monthly_from_raw():
    print(f"[PBJ] Scanning raw under: {PBJ_RAW_DIR}")
    if not PBJ_RAW_DIR.exists():
        raise FileNotFoundError(f"PBJ_RAW_DIR does not exist: {PBJ_RAW_DIR}. Update config.py.")

    csv_paths = sorted([p for p in PBJ_RAW_DIR.glob("*.csv") if p.is_file()])
    print(f"[PBJ] Found {len(csv_paths)} CSV(s)")
    if csv_paths[:6]:
        print("       First few CSV names:")
        for p in csv_paths[:6]:
            print(f"        - {p.name}")

    frames = []
    good, skipped = 0, 0
    for p in csv_paths:
        try:
            out = parse_pbj_nurse_daily_to_monthly(p)
            if out is not None and not out.empty:
                frames.append(out)
                good += 1
                print(f"[ok] {p.name}: {len(out):,} monthly rows")
            else:
                skipped += 1
                print(f"[skip] {p.name}: no matching daily schema")
        except Exception as e:
            skipped += 1
            print(f"[warn] {p.name}: failed with {e}")

    if not frames:
        raise RuntimeError("No parsable PBJ tables found in pbj-nurse daily schema.")

    monthly = pd.concat(frames, ignore_index=True)
    # ensure one row per CCN×month if duplicates slipped through
    monthly = (monthly
               .drop_duplicates(["cms_certification_number", "month"])
               .sort_values(["cms_certification_number", "month"], kind="mergesort")
               .reset_index(drop=True))

    monthly.to_csv(PBJ_MONTHLY_CSV, index=False)
    print(f"[PBJ] monthly built rows={len(monthly):,}, "
          f"CCNs={monthly['cms_certification_number'].nunique():,} "
          f"→ {PBJ_MONTHLY_CSV}")


# --------------------------------- CLI ---------------------------------------

if __name__ == "__main__":
    build_monthly_from_raw()