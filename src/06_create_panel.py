#!/usr/bin/env python
# coding: utf-8
# ─────────────────────────────────────────────────────────────────────────────
# Final PBJ Panel with CHOW Dummies (fixed)
#   * Inputs (data/interim):
#       - ccn_chow_lite.csv                     (ownership CHOW-lite)
#       - mcr_chow_provider_events_all.csv      (MCR CHOW wide)
#       - pbj_monthly_panel.csv                 (PBJ provider-month panel)
#   * Optional input (raw/provider-info-files):
#       - provider_resides_in_hospital_by_ccn.csv (CCN-level in-hospital flag)
#   * CCN normalization: alphanumeric-safe (pad only if purely digits)
#   * Hospital filter: apply to PBJ & MCR; skip LITE (already filtered upstream)
#   * Agreement rule: keep units that are either match_0 OR (exactly one CHOW in
#     each source with months within ±1 month); anchor change_month to LITE month.
#   * Output: data/clean/pbj_panel_with_chow_dummies.csv
# Run with: %run 06_build_final_pbj_panel.py
# ─────────────────────────────────────────────────────────────────────────────

import os, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================== Paths =========================================
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").is_dir() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

REPO       = PROJECT_ROOT
INTERIM    = REPO / "data" / "interim"
CLEAN_DIR  = REPO / "data" / "clean"; CLEAN_DIR.mkdir(parents=True, exist_ok=True)

LITE_FP    = INTERIM / "ccn_chow_lite.csv"
MCR_FP     = INTERIM / "mcr_chow_provider_events_all.csv"
PBJ_FP     = INTERIM / "pbj_monthly_panel.csv"

RAW_DIR    = Path(os.getenv("NH_DATA_DIR", PROJECT_ROOT / "data" / "raw")).resolve()
PROV_DIR   = RAW_DIR / "provider-info-files"
HOSP_FP    = PROV_DIR / "provider_resides_in_hospital_by_ccn.csv"  # optional

OUT_FP     = CLEAN_DIR / "pbj_panel_with_chow_dummies.csv"

# Hospital filter switches
DO_HOSPITAL_FILTER_LITE = False   # LITE already filtered upstream
DO_HOSPITAL_FILTER_MCR  = True
DO_HOSPITAL_FILTER_PBJ  = True

print(f"[paths] INTERIM={INTERIM}")
print(f"[paths] PBJ_FP ={PBJ_FP}")
print(f"[paths] LITE_FP={LITE_FP}")
print(f"[paths] MCR_FP ={MCR_FP}")
print(f"[paths] HOSP_FP={HOSP_FP} (exists={HOSP_FP.exists()})")
print(f"[out]   {OUT_FP}")

# ============================== Helpers =======================================
def normalize_ccn_any(series: pd.Series) -> pd.Series:
    """
    Preserve alphanumeric CCNs; strip separators; pad only if numeric.
    """
    s = series.astype("string").fillna("").str.strip().str.upper()
    s = s.str.replace(r"[ \-\/\.]", "", regex=True)
    is_digits = s.str.fullmatch(r"\d+")
    s = s.mask(is_digits, s.str.zfill(6))
    s = s.replace({"": pd.NA})
    return s

def to_monthstart(x) -> pd.Series:
    s = pd.to_datetime(x, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp("s")

def first_month_from_date_cols(df: pd.DataFrame, pat_list) -> pd.Series:
    """
    Find first (min) month across any columns matching regexes in pat_list.
    Returns a Series of month-start Timestamps.
    """
    cols = []
    for pat in pat_list:
        cols += [c for c in df.columns if re.search(pat, c, re.I)]
    cols = sorted(set(cols))
    if not cols:
        return pd.Series(pd.NaT, index=df.index)
    tmp = df[cols].apply(pd.to_datetime, errors="coerce")
    return to_monthstart(tmp.min(axis=1))

def bool_from_any(x: pd.Series) -> pd.Series:
    s = x.astype("string").str.strip().str.lower()
    return s.map({"1":True,"y":True,"yes":True,"true":True,"t":True,
                  "0":False,"n":False,"no":False,"false":False,"f":False}).astype("boolean")

def same_month_or_within_one(a, b) -> bool:
    if pd.isna(a) or pd.isna(b): return False
    pa, pb = pd.Period(a, "M"), pd.Period(b, "M")
    return abs((pa - pb).n) <= 1

def find_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

# ============================== Load inputs ===================================
lite = pd.read_csv(LITE_FP, dtype={"cms_certification_number":"string"}, low_memory=False)
mcr  = pd.read_csv(MCR_FP,  dtype={"cms_certification_number":"string"}, low_memory=False)
pbj  = pd.read_csv(PBJ_FP,  dtype={"cms_certification_number":"string"}, low_memory=False)

# CCN normalization
lite["cms_certification_number"] = normalize_ccn_any(lite["cms_certification_number"])
mcr["cms_certification_number"]  = normalize_ccn_any(mcr["cms_certification_number"])
pbj["cms_certification_number"]  = normalize_ccn_any(pbj["cms_certification_number"])

# ============================== Hospital filter ===============================
if (DO_HOSPITAL_FILTER_LITE or DO_HOSPITAL_FILTER_MCR or DO_HOSPITAL_FILTER_PBJ) and HOSP_FP.exists():
    hosp = pd.read_csv(HOSP_FP, dtype=str, low_memory=False)
    hosp.columns = [c.strip().lower() for c in hosp.columns]
    ccn_col = next((c for c in ["cms_certification_number","ccn","provnum","prvdr_num"] if c in hosp.columns), None)
    if ccn_col is None or "provider_resides_in_hospital" not in hosp.columns:
        raise ValueError("[hospital] expected CCN + provider_resides_in_hospital columns.")
    hosp["cms_certification_number"] = normalize_ccn_any(hosp[ccn_col])
    hosp["provider_resides_in_hospital"] = bool_from_any(hosp["provider_resides_in_hospital"])
    drop = set(hosp.loc[hosp["provider_resides_in_hospital"]==True,"cms_certification_number"].dropna().unique())

    def _drop_in_hosp(df, name, do_filter):
        if not do_filter:
            print(f"[hospital filter] {name}: skipped")
            return df
        before = df["cms_certification_number"].nunique()
        df2 = df[~df["cms_certification_number"].isin(drop)].copy()
        after = df2["cms_certification_number"].nunique()
        print(f"[hospital filter] {name}: CCNs {before:,} -> {after:,} (removed {before-after:,})")
        return df2

    lite = _drop_in_hosp(lite, "LITE", DO_HOSPITAL_FILTER_LITE)
    mcr  = _drop_in_hosp(mcr,  "MCR",  DO_HOSPITAL_FILTER_MCR)
    pbj  = _drop_in_hosp(pbj,  "PBJ",  DO_HOSPITAL_FILTER_PBJ)
else:
    print("[hospital filter] skipped or file missing")

# ============================== First-event months ============================
# Ownership (lite)
lite_counts = (
    lite[["cms_certification_number","num_chows"]].copy()
    if "num_chows" in lite.columns else
    lite[["cms_certification_number"]].assign(num_chows=0)
)
lite_counts["num_chows"] = pd.to_numeric(lite_counts["num_chows"], errors="coerce").fillna(0).astype(int)
lite_counts = lite_counts.drop_duplicates("cms_certification_number").reset_index(drop=True)
lite_counts["first_event_month_lite"] = first_month_from_date_cols(
    lite, pat_list=[r"^chow_date_\d+$", r"^chow_\d+_date$"]
)

# MCR (wide)
mcr_counts = (
    mcr[["cms_certification_number","n_chow"]].copy()
    if "n_chow" in mcr.columns else
    mcr[["cms_certification_number"]].assign(n_chow=0)
)
mcr_counts["n_chow"] = pd.to_numeric(mcr_counts["n_chow"], errors="coerce").fillna(0).astype(int)
mcr_counts = mcr_counts.drop_duplicates("cms_certification_number").reset_index(drop=True)
mcr_counts["first_event_month_mcr"] = first_month_from_date_cols(
    mcr, pat_list=[r"^chow_\d+_date$"]
)

print("[diag] lite first-month non-null:", int(lite_counts["first_event_month_lite"].notna().sum()))
print("[diag] mcr  first-month non-null:", int(mcr_counts["first_event_month_mcr"].notna().sum()))

# ============================== Agreement & change month ======================
merged = (lite_counts
          .merge(mcr_counts, on="cms_certification_number", how="outer")
          .fillna({"num_chows":0, "n_chow":0}))

merged["is_chow_lite"] = merged["num_chows"] > 0
merged["is_chow_mcr"]  = merged["n_chow"]    > 0

def agreement_picker(r):
    label = "mismatch"
    change_month = pd.NaT
    # match_0: neither source has a CHOW
    if (r["num_chows"]==0) and (r["n_chow"]==0):
        label = "match_0"
    # match_1_same_month (±1 month tolerance): exactly one CHOW in each
    elif (r["num_chows"]==1) and (r["n_chow"]==1):
        a, b = r["first_event_month_lite"], r["first_event_month_mcr"]
        if same_month_or_within_one(a, b):
            label = "match_1_same_month"
            # Anchor to LITE month if present, else MCR
            change_month = pd.Period(a, "M").to_timestamp("s") if pd.notna(a) else (
                           pd.Period(b, "M").to_timestamp("s") if pd.notna(b) else pd.NaT)
        elif pd.notna(a) and pd.notna(b):
            label = "match_1_diff_month"
        else:
            label = "match_1_unknown_month"
    return pd.Series({"agreement": label, "change_month": change_month})

ag = merged.apply(agreement_picker, axis=1)
merged = pd.concat([merged, ag], axis=1)
agree = merged.loc[merged["agreement"].isin(["match_0","match_1_same_month"])].copy()

print(f"[agree] match_0={int((agree['agreement']=='match_0').sum()):,} | "
      f"match_1_same_month={int((agree['agreement']=='match_1_same_month').sum()):,} | "
      f"total_agree={len(agree):,}")

# ============================== Prepare PBJ panel =============================
# Detect the month column robustly
pbj_month_col = find_col(pbj.columns, ["year_month","month","pbj_month","date","period_month"])
if pbj_month_col is None:
    raise ValueError(f"[PBJ] Could not find month column in PBJ panel. Columns={list(pbj.columns)}")

# Build month_start
if pbj_month_col.lower() == "year_month":
    pbj["month"] = pd.PeriodIndex(pbj["year_month"].astype(str), freq="M").to_timestamp("s")
else:
    pbj["month"] = to_monthstart(pbj[pbj_month_col])

# Canonical CCN
pbj["cms_certification_number"] = normalize_ccn_any(pbj["cms_certification_number"])

# ============================== Merge & dummies ===============================
# Keep only CCNs in the agreed universe (match_0 + match_1_same_month)
panel = pbj.merge(
    agree[["cms_certification_number","agreement","change_month"]],
    on="cms_certification_number",
    how="inner"
)

# treat_post: 1 starting IN the change month and onward for match_1_same_month
panel["treat_post"] = 0
is_match1 = panel["agreement"].eq("match_1_same_month") & panel["change_month"].notna()
panel.loc[is_match1, "treat_post"] = (panel.loc[is_match1, "month"] >= panel.loc[is_match1, "change_month"]).astype(int)

# event_time: months relative to change month (NaN for match_0)
panel["event_time"] = np.nan
panel.loc[is_match1, "event_time"] = (
    (panel.loc[is_match1, "month"].values.astype("datetime64[M]") -
     panel.loc[is_match1, "change_month"].values.astype("datetime64[M]")).astype(int)
)

# ever treated (diagnostic)
ever = (panel.groupby("cms_certification_number", as_index=False)["treat_post"].max()
             .rename(columns={"treat_post":"ever_treated"}))
panel = panel.merge(ever, on="cms_certification_number", how="left")

# ============================== Diagnostics ==================================
print("[diag] treat_post==1 rows:", int((panel["treat_post"]==1).sum()))
print("[diag] change_month non-null:", int(panel["change_month"].notna().sum()))
print("[diag] unique CCNs in panel:", panel["cms_certification_number"].nunique())

# ============================== Save =========================================
panel = panel.sort_values(["cms_certification_number","month"]).reset_index(drop=True)
panel.to_csv(OUT_FP, index=False)
print(f"[save] {OUT_FP} rows={len(panel):,} cols={panel.shape[1]}")
print(panel[["cms_certification_number","month","agreement","change_month","treat_post","event_time"]].head(12))