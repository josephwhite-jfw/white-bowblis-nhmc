# nhmc/pbj.py
# PBJ nurse staffing: daily → monthly HPPD panel (+coverage)

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .paths import RAW_DIR, INTERIM, PBJ_MONTHLY, PBJ_COVERAGE
from .utils import normalize_ccn_any

PBJ_DIR  = RAW_DIR / "pbj-nurse"
PBJ_GLOB = "pbj_nurse_????_Q[1-4].csv"
INTERIM.mkdir(parents=True, exist_ok=True)

def _read_robust(fp: Path) -> pd.DataFrame:
    for enc in ("utf-8","utf-8-sig","cp1252","latin1"):
        try:
            return pd.read_csv(fp, low_memory=False, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(fp, low_memory=False, encoding="latin1", encoding_errors="replace", on_bad_lines="skip")

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    if "provnum" in d.columns and "cms_certification_number" not in d.columns:
        d = d.rename(columns={"provnum":"cms_certification_number"})
    if "mdscensus" in d.columns and "mds_census" not in d.columns:
        d = d.rename(columns={"mdscensus":"mds_census"})
    req = ["cms_certification_number","workdate"]
    for r in req:
        if r not in d.columns: raise ValueError(f"PBJ missing col: {r}")

    d["cms_certification_number"] = normalize_ccn_any(d["cms_certification_number"])
    if str(d["workdate"].dtype).startswith(("int","Int")):
        d["workdate"] = pd.to_datetime(d["workdate"].astype("Int64"), format="%Y%m%d", errors="coerce")
    else:
        d["workdate"] = pd.to_datetime(d["workdate"], errors="coerce")
    for c in ["hrs_rn","hrs_lpn","hrs_cna","mds_census"]:
        if c not in d.columns: d[c] = 0.0
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32").fillna(0.0 if c!="mds_census" else np.nan)
    return d[["cms_certification_number","workdate","hrs_rn","hrs_lpn","hrs_cna","mds_census"]]

def _coverage_and_hppd(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = (daily.groupby(["cms_certification_number","workdate"], as_index=False)
                  .agg(hrs_rn=("hrs_rn","sum"),
                       hrs_lpn=("hrs_lpn","sum"),
                       hrs_cna=("hrs_cna","sum"),
                       mds_census=("mds_census","mean")))
    daily["total_hours"] = daily[["hrs_rn","hrs_lpn","hrs_cna"]].sum(axis=1).astype("float32")
    daily["year_month"] = daily["workdate"].dt.to_period("M")
    daily["dim"] = daily["workdate"].dt.days_in_month

    monthly = (daily.groupby(["cms_certification_number","year_month"], as_index=False)
                    .agg(rn_hours_month=("hrs_rn","sum"),
                         lpn_hours_month=("hrs_lpn","sum"),
                         cna_hours_month=("hrs_cna","sum"),
                         total_hours=("total_hours","sum"),
                         resident_days=("mds_census","sum"),
                         avg_daily_census=("mds_census","mean"),
                         days_reported=("workdate","nunique"),
                         days_in_month=("dim","max")))
    monthly["coverage_ratio"] = monthly["days_reported"] / monthly["days_in_month"]
    denom = monthly["resident_days"].replace({0: np.nan})
    monthly["rn_hppd"]    = monthly["rn_hours_month"]/denom
    monthly["lpn_hppd"]   = monthly["lpn_hours_month"]/denom
    monthly["cna_hppd"]   = monthly["cna_hours_month"]/denom
    monthly["total_hppd"] = monthly["total_hours"]/denom
    monthly["month"]      = monthly["year_month"].dt.strftime("%m/%Y")

    # gap tracking
    tmp = monthly.copy()
    tmp["month_index"] = (tmp["year_month"].dt.year*12 + tmp["year_month"].dt.month).astype("Int32")
    tmp = tmp.sort_values(["cms_certification_number","month_index"])
    tmp["gap_from_prev_months"] = tmp.groupby("cms_certification_number")["month_index"].diff().fillna(1).astype("Int16")-1
    tmp["gap_from_prev_months"] = tmp["gap_from_prev_months"].clip(lower=0)
    cov = (tmp.groupby("cms_certification_number", as_index=False)
               .agg(first_idx=("month_index","min"),
                    last_idx=("month_index","max"),
                    months_observed=("month_index","nunique"),
                    any_gaps=("gap_from_prev_months", lambda s: bool((s>0).any()))))
    cov["months_expected"] = (cov["last_idx"]-cov["first_idx"]+1).astype("Int32")
    cov["months_missing"]  = (cov["months_expected"]-cov["months_observed"]).astype("Int32")
    cov["has_gaps"] = cov["any_gaps"]
    cov = cov.drop(columns=["any_gaps"])
    return monthly, cov

def build_pbj_monthly() -> None:
    files = sorted(PBJ_DIR.glob(PBJ_GLOB))
    frames = []
    for fp in files:
        try:
            d = _normalize(_read_robust(fp))
            frames.append(d)
            print(f"[pbj] ok: {fp.name} rows={len(d):,}")
        except Exception as e:
            print(f"[pbj] fail: {fp.name} -> {e}")
    if not frames:
        raise RuntimeError("No PBJ files ingested.")

    daily_all = pd.concat(frames, ignore_index=True)
    monthly, cov = _coverage_and_hppd(daily_all)

    # order & save
    base_cols = [
        "cms_certification_number","month","year_month","resident_days","avg_daily_census",
        "rn_hours_month","lpn_hours_month","cna_hours_month","total_hours",
        "rn_hppd","lpn_hppd","cna_hppd","total_hppd",
        "days_reported","days_in_month","coverage_ratio"
    ]
    monthly["year_month"] = monthly["year_month"].astype(str)
    monthly = monthly[base_cols]
    monthly.to_csv(PBJ_MONTHLY, index=False)
    cov.to_csv(PBJ_COVERAGE, index=False)
    print(f"[pbj] saved panel → {PBJ_MONTHLY} (rows={len(monthly):,})")
    print(f"[pbj] saved coverage → {PBJ_COVERAGE} (rows={len(cov):,})")
