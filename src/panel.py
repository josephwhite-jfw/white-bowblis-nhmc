#!/usr/bin/env python
# coding: utf-8
# -----------------------------------------------------------------------------
# Build Final Panel (outer join) + CHOW filter + treatment/post/event time + case-mix
# + date window + within-quarter fills (AFTER final panel) + gap indicator
# Also writes an analytical panel with HPPD + hospital filters.
# -----------------------------------------------------------------------------

import os, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================== Paths =========================================
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").is_dir() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

REPO        = PROJECT_ROOT
INTERIM     = REPO / "data" / "interim"
CLEAN_DIR   = REPO / "data" / "clean"; CLEAN_DIR.mkdir(parents=True, exist_ok=True)

PROVIDER_FP = INTERIM / "provider.csv"
PBJ_FP      = INTERIM / "pbj_nurse.csv"
MCR_FP      = INTERIM / "mcr.csv"
CHOW_FP     = INTERIM / "chow.csv"
OUT_PBJ_FP  = CLEAN_DIR / "pbj_panel.csv"
OUT_ANL_FP  = CLEAN_DIR / "analytical_panel.csv"

print(f"[paths] provider={PROVIDER_FP.exists()}  pbj={PBJ_FP.exists()}  mcr={MCR_FP.exists()}  chow={CHOW_FP.exists()}")
print(f"[out]   pbj={OUT_PBJ_FP}")
print(f"[out]   analytical={OUT_ANL_FP}")

# ============================== Config ========================================
START_YM = "2017/01"
END_YM   = "2024/06"
START_Q  = "2017Q1"
END_Q    = "2024Q2"

# ============================== Helpers =======================================
def normalize_ccn_any(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("").str.strip().str.upper()
    s = s.str.replace(r"[ \-\/\.]", "", regex=True)
    is_digits = s.str.fullmatch(r"\d+")
    s = s.mask(is_digits, s.str.zfill(6))
    s = s.replace({"": pd.NA})
    return s

def to_monthstart(x) -> pd.Series:
    s = pd.to_datetime(x, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp("s")

def first_chow_month(df: pd.DataFrame, patt: str) -> pd.Series:
    cols = [c for c in df.columns if re.search(patt, c, flags=re.I)]
    if not cols:
        return pd.Series(pd.NaT, index=df.index)
    tmp = df[cols].apply(pd.to_datetime, errors="coerce")
    return to_monthstart(tmp.min(axis=1))

def months_diff(a, b) -> float:
    if pd.isna(a) or pd.isna(b): return np.inf
    pa, pb = pd.Period(a, "M"), pd.Period(b, "M")
    return float((pa - pb).n)

def within_k_months(a, b, k=6) -> bool:
    d = months_diff(a, b)
    return (d != np.inf) and (abs(d) <= k)

def rank_bins_pct(s: pd.Series, n_bins: int) -> pd.Series:
    pct = s.rank(method="average", pct=True)
    bins = np.ceil(pct * n_bins)
    bins = pd.to_numeric(bins, errors="coerce").clip(1, n_bins)
    bins = bins.where(s.notna())
    return bins.astype("Int16")

def make_case_mix_bins_and_dummies(panel: pd.DataFrame, cm_col: str, state_col: str = "state"):
    out = panel.copy()
    out[cm_col] = pd.to_numeric(out[cm_col], errors="coerce")

    # National bins per month
    out["cm_quart_nat"] = out.groupby("year_month", observed=True)[cm_col].transform(lambda s: rank_bins_pct(s, 4))
    out["cm_decil_nat"] = out.groupby("year_month", observed=True)[cm_col].transform(lambda s: rank_bins_pct(s, 10))

    # State×month
    if state_col in out.columns:
        mask = out[state_col].notna()
        out.loc[mask, "cm_quart_state"] = (
            out[mask].groupby(["year_month", state_col], observed=True)[cm_col]
                     .transform(lambda s: rank_bins_pct(s, 4))
        ).astype("Int16")
        out.loc[mask, "cm_decil_state"] = (
            out[mask].groupby(["year_month", state_col], observed=True)[cm_col]
                     .transform(lambda s: rank_bins_pct(s, 10))
        ).astype("Int16")
    else:
        out["cm_quart_state"] = pd.Series([pd.NA]*len(out), dtype="Int16")
        out["cm_decil_state"] = pd.Series([pd.NA]*len(out), dtype="Int16")

    # Dummies (drop bin=1 as reference; add missing)
    def dums(df, col, prefix):
        miss = df[col].isna().astype("Int8").rename(f"{prefix}_missing")
        d = pd.get_dummies(df[col], prefix=prefix, dtype="Int8")
        ref = f"{prefix}_1"
        if ref in d.columns:
            d = d.drop(columns=[ref])
        return pd.concat([d, miss], axis=1)

    parts = []
    for col, pre in [("cm_quart_nat","cm_q_nat"), ("cm_decil_nat","cm_d_nat"),
                     ("cm_quart_state","cm_q_state"), ("cm_decil_state","cm_d_state")]:
        parts.append(dums(out, col, pre))
    out = pd.concat([out, pd.concat(parts, axis=1)], axis=1)
    return out

def filter_to_window(df: pd.DataFrame) -> pd.DataFrame:
    if "year_month" in df.columns:
        ym = pd.PeriodIndex(df["year_month"].astype(str), freq="M")
        mask_ym = (ym >= pd.Period(START_YM, "M")) & (ym <= pd.Period(END_YM, "M"))
    else:
        mask_ym = pd.Series(True, index=df.index)
    if "quarter" in df.columns:
        q = pd.PeriodIndex(df["quarter"].astype(str), freq="Q")
        mask_q = (q >= pd.Period(START_Q, "Q")) & (q <= pd.Period(END_Q, "Q"))
    else:
        mask_q = pd.Series(True, index=df.index)
    return df[mask_ym & mask_q].copy()

def coalesce_suffix_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = list(out.columns)
    suffixed = [c for c in cols if re.search(r"\.\d+$", c)]
    by_base = {}
    for c in suffixed:
        base = re.sub(r"\.\d+$", "", c)
        by_base.setdefault(base, []).append(c)
    for base, dups in by_base.items():
        if base in out.columns:
            for dup in dups:
                out[base] = out[base].where(out[base].notna(), out[dup])
            out = out.drop(columns=dups)
        else:
            out[base] = pd.NA
            for dup in dups:
                out[base] = out[base].where(out[base].notna(), out[dup])
            out = out.drop(columns=dups)
    return out

# ============================== Load ==========================================
provider = pd.read_csv(PROVIDER_FP, low_memory=False)
pbj      = pd.read_csv(PBJ_FP,      low_memory=False)
mcr      = pd.read_csv(MCR_FP,      low_memory=False)
chow     = pd.read_csv(CHOW_FP,     low_memory=False)

for df in (provider, pbj, mcr, chow):
    if "cms_certification_number" in df.columns:
        df["cms_certification_number"] = normalize_ccn_any(df["cms_certification_number"])

# Restrict window BEFORE merge
provider = filter_to_window(provider)
pbj      = filter_to_window(pbj)
mcr      = filter_to_window(mcr)

# ============================== CHOW agreement filter =========================
chow["n_chow_nh_compare"] = pd.to_numeric(chow.get("n_chow_nh_compare"), errors="coerce").fillna(0).astype(int)
chow["n_chow_mcr"]        = pd.to_numeric(chow.get("n_chow_mcr"),        errors="coerce").fillna(0).astype(int)
chow["first_nh_month"]  = first_chow_month(chow, r"^nh_compare_chow_\d+_date$")
chow["first_mcr_month"] = first_chow_month(chow, r"^mcr_chow_\d+_date$")

def _agree_row(r):
    if r["n_chow_nh_compare"] in (0,1) and r["n_chow_mcr"] in (0,1):
        if (r["n_chow_nh_compare"] == 0) and (r["n_chow_mcr"] == 0):
            return True
        if (r["n_chow_nh_compare"] == 1) and (r["n_chow_mcr"] == 1):
            return within_k_months(r["first_nh_month"], r["first_mcr_month"], k=6)
    return False

agree_mask = chow.apply(_agree_row, axis=1)
agree_ccns = set(chow.loc[agree_mask, "cms_certification_number"].dropna().unique())
print(f"[chow] CCNs passing (0/0 or 1/1 within 6m): {len(agree_ccns):,}")

nh_timing = chow.loc[chow["cms_certification_number"].isin(agree_ccns),
                     ["cms_certification_number","n_chow_nh_compare","first_nh_month"]].drop_duplicates("cms_certification_number")

# ============================== Outer join base ===============================
keys = ["cms_certification_number","quarter","year_month"]
for name, df in [("provider",provider),("pbj",pbj),("mcr",mcr)]:
    miss = [k for k in keys if k not in df.columns]
    if miss:
        raise KeyError(f"[{name}] missing key columns: {miss}")

base = provider.merge(pbj, on=keys, how="outer") \
               .merge(mcr, on=keys, how="outer")

# Keep only CHOW-agree CCNs
base["cms_certification_number"] = normalize_ccn_any(base["cms_certification_number"])
base = base[base["cms_certification_number"].isin(agree_ccns)].copy()

# Attach NH timing
base = base.merge(nh_timing, on="cms_certification_number", how="left")

# ============================== Treatment / Post / Event-time =================
ym_periods = pd.PeriodIndex(base["year_month"].astype(str), freq="M")

# First CHOW month as Period[M]
first_p = pd.to_datetime(base["first_nh_month"], errors="coerce").dt.to_period("M")

# treatment = ever treated at the CCN level (1 if n_chow_nh_compare==1 anywhere for that CCN)
base["treatment"] = (
    base["n_chow_nh_compare"].eq(1)
        .groupby(base["cms_certification_number"])
        .transform("max")
        .astype(int)
)

# Robust month-difference calculation
base["event_time"] = np.nan
mask = first_p.notna()
ym_y = pd.Series(ym_periods.year,  index=base.index)
ym_m = pd.Series(ym_periods.month, index=base.index)
fp_y = first_p.dt.year
fp_m = first_p.dt.month
et_vals = (ym_y[mask] - fp_y[mask]) * 12 + (ym_m[mask] - fp_m[mask])
base.loc[mask, "event_time"] = et_vals.astype(int)

# post = 1 for months strictly AFTER the CHOW month (uses event_time > 0)
base["post"] = 0
has_one = base["n_chow_nh_compare"].eq(1) & mask
base.loc[has_one, "post"] = (base.loc[has_one, "event_time"] > 0).astype(int)

# anticipation dummy = 1 for event_time in {-3,-2,-1,0,1,2}
base["anticipation"] = 0
base.loc[base["event_time"].isin([-3, -2, -1, 0, 1, 2]), "anticipation"] = 1

# ============================== Case-mix dummies ==============================
if "case_mix_total" not in base.columns:
    base["case_mix_total"] = pd.NA
base = make_case_mix_bins_and_dummies(base, cm_col="case_mix_total", state_col="state")

# ============================== Build FINAL panel =============================
want_cols = [
    "cms_certification_number",
    "quarter",
    "year_month",
    "treatment",
    "post",
    "event_time",
    "anticipation",
    "provider_resides_in_hospital",
    "gap_from_prev_months",
    "coverage_ratio",
    "rn_hppd","lpn_hppd","cna_hppd","total_hppd",
    "non_profit","government",
    "chain",
    "num_beds",
    "beds_prov",                 # <-- NEW: provider-file beds
    "ccrc_facility","sff_facility",
    "occupancy_rate",
    "pct_medicare","pct_medicaid",
    "urban",
]

cm_dummy_cols_all = [c for c in base.columns if c.startswith(("cm_q_nat_","cm_d_nat_","cm_q_state_","cm_d_state_")) or
                     (c.endswith("_missing") and c.startswith(("cm_q_","cm_d_")))]
want_cols += [c for c in cm_dummy_cols_all if c not in want_cols]

want_cols = [c for c in want_cols if c in base.columns]
panel = base[want_cols].copy()
panel = coalesce_suffix_duplicates(panel)

# ============================== Within-quarter fill (AFTER final panel) =======
binary_quarter_fill = [
    "provider_resides_in_hospital",
    "non_profit", "government",
    "chain",
    "ccrc_facility", "sff_facility",
    "urban",
] + [c for c in panel.columns if c.startswith(("cm_q_nat_","cm_d_nat_","cm_q_state_","cm_d_state_")) or
      (c.endswith("_missing") and c.startswith(("cm_q_","cm_d_")))]

numeric_quarter_fill = [
    "num_beds",
    "beds_prov",
    "occupancy_rate",
    "pct_medicare","pct_medicaid",
]

_fill_cols = [c for c in (binary_quarter_fill + numeric_quarter_fill) if c in panel.columns]
_key_cols  = ["cms_certification_number", "quarter"]

if _fill_cols:
    _orig_dtypes = panel[_fill_cols].dtypes.to_dict()
    filled_block = (
        panel.sort_values(_key_cols + ["year_month"])
             .groupby(_key_cols, observed=True, sort=False)[_fill_cols]
             .transform(lambda df: df.ffill().bfill())
    )
    panel[_fill_cols] = panel[_fill_cols].where(panel[_fill_cols].notna(), filled_block)
    # restore original dtypes where possible; ensure beds_prov numeric/nullable int
    for c, dt in _orig_dtypes.items():
        try:
            if c == "beds_prov":
                panel[c] = pd.to_numeric(panel[c], errors="coerce").astype("Int64")
            else:
                panel[c] = panel[c].astype(dt)
        except Exception:
            pass

# ============================== Unified 'beds' variable ======================
if {"num_beds", "beds_prov"}.issubset(panel.columns):
    panel["beds"] = panel["num_beds"].where(panel["num_beds"].notna(), panel["beds_prov"])
    panel["beds"] = pd.to_numeric(panel["beds"], errors="coerce").astype("Int64")
else:
    # if one source missing, just carry whichever exists
    if "num_beds" in panel.columns:
        panel["beds"] = pd.to_numeric(panel["num_beds"], errors="coerce").astype("Int64")
    elif "beds_prov" in panel.columns:
        panel["beds"] = pd.to_numeric(panel["beds_prov"], errors="coerce").astype("Int64")
    else:
        panel["beds"] = pd.Series([pd.NA]*len(panel), dtype="Int64")

# ============================== GAP indicator =================================
if "gap_from_prev_months" in panel.columns:
    panel["gap"] = (panel["gap_from_prev_months"] > 0).groupby(panel["cms_certification_number"]).transform("max").astype("Int8")
else:
    panel["gap"] = pd.Series([pd.NA]*len(panel), dtype="Int8")

# ============================== Final types, save PBJ panel ===================
for col in ["non_profit","government","chain","urban","ccrc_facility","sff_facility",
            "provider_resides_in_hospital","gap","post","treatment","anticipation"]:
    if col in panel.columns:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").astype("Int8")

# ensure beds_prov is numeric/nullable int even if not filled above
if "beds_prov" in panel.columns:
    panel["beds_prov"] = pd.to_numeric(panel["beds_prov"], errors="coerce").astype("Int64")

panel = panel.sort_values(["cms_certification_number","year_month"]).reset_index(drop=True)
panel.to_csv(OUT_PBJ_FP, index=False)
print(f"[save] PBJ panel → {OUT_PBJ_FP} rows={len(panel):,} cols={panel.shape[1]}")

# ============================== Analytical panel ==============================
analytical = panel.copy()
analytical = analytical.replace(r"^\s*$", np.nan, regex=True)

# Drop rows with any NaN in HPPDs
hppd_cols = [c for c in ["rn_hppd","lpn_hppd","cna_hppd","total_hppd"] if c in analytical.columns]
if hppd_cols:
    before = len(analytical)
    analytical = analytical.dropna(subset=hppd_cols)
    print(f"[filter] drop rows with NaN in any HPPD: {before:,} -> {len(analytical):,}")

# Drop hospital-resident rows
if "provider_resides_in_hospital" in analytical.columns:
    before = len(analytical)
    analytical = analytical[analytical["provider_resides_in_hospital"] != 1]
    print(f"[filter] drop 'provider_resides_in_hospital'==1: {before:,} -> {len(analytical):,}")

analytical.to_csv(OUT_ANL_FP, index=False)
print(f"[done] saved analytical panel → {OUT_ANL_FP}")