#!/usr/bin/env python
# coding: utf-8
# ─────────────────────────────────────────────────────────────────────────────
# Final PBJ Panel with CHOW Dummies + MCR Controls (single script)
#   * Inputs (data/interim):
#       - ccn_chow_lite.csv                     (ownership CHOW-lite)
#       - mcr_chow_provider_events_all.csv      (MCR CHOW wide; POST-2017 aligned)
#       - pbj_monthly_panel.csv                 (PBJ provider-month panel)
#   * Optional input (raw/provider-info-files):
#       - provider_resides_in_hospital_by_ccn.csv (CCN-level in-hospital flag)
#   * Also reads MCR raw files (SAS preferred; CSV fallback) to BUILD controls:
#       - ownership_type (For-profit / Nonprofit / Government)
#       - pct_medicare, pct_medicaid (patient-day shares)
#       - num_beds (avg beds), occupancy_rate (%), state, urban_rural, is_chain
#   * Agreement rule for panel inclusion:
#       - keep (match_0) OR (exactly one CHOW in each and LITE vs MCR first-event
#         months within ±6 months); anchor change_month to the LITE month.
#   * Output: data/clean/pbj_panel_with_chow_dummies.csv
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

print(f"[paths] INTERIM={INTERIM}")
print(f"[paths] PBJ_FP ={PBJ_FP}")
print(f"[paths] LITE_FP={LITE_FP}")
print(f"[paths] MCR_FP ={MCR_FP}")
print(f"[paths] HOSP_FP={HOSP_FP} (exists={HOSP_FP.exists()})")
print(f"[paths] RAW_DIR={RAW_DIR}")
print(f"[out]   {OUT_FP}")

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

def first_month_from_date_cols(df: pd.DataFrame, pat_list) -> pd.Series:
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

def months_diff(a, b) -> float:
    if pd.isna(a) or pd.isna(b): return np.inf
    pa, pb = pd.Period(a, "M"), pd.Period(b, "M")
    return float((pa - pb).n)

def within_k_months(a, b, k=6) -> bool:
    d = months_diff(a,b)
    return (d != np.inf) and (abs(d) <= k)

def find_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand and cand.lower() in lower:
            return lower[cand.lower()]
    return None

# ============================== Hospital filter loader ========================
def load_hospital_dropset():
    if not HOSP_FP.exists():
        print("[hospital filter] skipped or file missing")
        return set()
    hosp = pd.read_csv(HOSP_FP, dtype=str, low_memory=False)
    hosp.columns = [c.strip().lower() for c in hosp.columns]
    ccn_col = next((c for c in ["cms_certification_number","ccn","provnum","prvdr_num"] if c in hosp.columns), None)
    if ccn_col is None or "provider_resides_in_hospital" not in hosp.columns:
        raise ValueError("[hospital] expected CCN + provider_resides_in_hospital columns.")
    hosp["cms_certification_number"] = normalize_ccn_any(hosp[ccn_col])
    hosp["provider_resides_in_hospital"] = bool_from_any(hosp["provider_resides_in_hospital"])
    drop = set(hosp.loc[hosp["provider_resides_in_hospital"]==True,"cms_certification_number"].dropna().unique())
    print(f"[hospital] drop-set size: {len(drop):,}")
    return drop

# ============================== Build MCR Controls (inline) ===================
def build_mcr_controls_monthly():
    """
    Reads SAS first (mcr_flatfile_20??.sas7bdat), else CSVs (mcr_flatfile_20??.csv).
    Returns monthly controls with columns:
      cms_certification_number, month, ownership_type, pct_medicare, pct_medicaid,
      num_beds, occupancy_rate, state, urban_rural, is_chain
    """
    MCR_DIR = RAW_DIR / "medicare-cost-reports"
    sas_files = sorted(MCR_DIR.glob("mcr_flatfile_20??.sas7bdat"))
    csv_files = sorted(MCR_DIR.glob("mcr_flatfile_20??.csv"))

    use_sas = len(sas_files) > 0
    if use_sas:
        try:
            import pyreadstat  # noqa: F401
        except Exception:
            print("[controls] pyreadstat not available; falling back to CSVs")
            use_sas = False

    print(f"[controls] source={'SAS' if use_sas else 'CSV'}  dir={MCR_DIR}")

    # Column candidates — tuned to your actual 2023 columns
    CAND = dict(
        PRVDR_NUM       = ["PRVDR_NUM","provnum","prvdr_num","Provider Number"],
        FY_BGN_DT       = ["FY_BGN_DT","fy_bgn_dt","Cost Report Fiscal Year beginning date"],
        FY_END_DT       = ["FY_END_DT","fy_end_dt","Cost Report Fiscal Year ending date"],
        MRC_OWNERSHIP   = ["MRC_OWNERSHIP","MRC_ownership","mrc_ownership","MRC_ownership_code"],

        # Patient days (numerator for occupancy)
        PAT_DAYS_TOT    = ["S3_1_PATDAYS_TOTAL","PATDAYS_TOTAL","PATIENT_DAYS_TOTAL"],
        PAT_DAYS_MCR    = ["S3_1_PATDAYS_MEDICARE","PATDAYS_MEDICARE","PATIENT_DAYS_MEDICARE"],
        PAT_DAYS_MCD    = ["S3_1_PATDAYS_MEDICAID","PATDAYS_MEDICAID","PATIENT_DAYS_MEDICAID"],

        # Bed-days available (preferred denominator for occupancy)
        BEDDAYS_AVAIL   = ["S3_1_BEDDAYS_AVAL","BEDDAYS_AVAL","S3_1_BED_DAYS_AVAIL","BED_DAYS_AVAIL"],

        # Beds (preferred → fallbacks)
        TOT_BEDS        = ["S3_1_TOTALBEDS","TOTAL_BEDS","TOT_BEDS","BEDS","S3_1_BEDS"],  # S3_1_BEDS as fallback too
        AVG_BEDS        = ["AVG_BEDS","AVERAGE_BEDS","AVG_INPT_BEDS","S3_1_AVG_BEDS"],    # rarely present but keep

        STATE           = ["MCR_STATE","STATE","PROV_STATE","STATE_CD","PROV_STATE_CD"],
        URBAN           = ["MCR_URBAN","URBAN_RURAL","URBAN_RURAL_INDICATOR","URBAN_IND","URBAN","URBRUR"],
        HOME_OFFICE     = ["MCR_HOME_OFFICE","HOME_OFFICE","HOME_OFFICE_IND","HOME_OFFICE_INDICATOR","HOME_OFFICE_FLAG"]
    )

    def _pick(cols, names): return find_col(cols, names)

    frames = []
    if use_sas:
        import pyreadstat
        for fp in sas_files:
            df, _ = pyreadstat.read_sas7bdat(str(fp), disable_datetime_conversion=0)
            df.columns = [c.upper().strip() for c in df.columns]
            cols = list(df.columns)

            keep = dict(
                PRVDR_NUM     = _pick(cols, CAND["PRVDR_NUM"]),
                FY_BGN_DT     = _pick(cols, CAND["FY_BGN_DT"]),
                FY_END_DT     = _pick(cols, CAND["FY_END_DT"]),
                MRC_OWNERSHIP = _pick(cols, CAND["MRC_OWNERSHIP"]),
                PAT_DAYS_TOT  = _pick(cols, CAND["PAT_DAYS_TOT"]),
                PAT_DAYS_MCR  = _pick(cols, CAND["PAT_DAYS_MCR"]),
                PAT_DAYS_MCD  = _pick(cols, CAND["PAT_DAYS_MCD"]),
                BEDDAYS_AVAIL = _pick(cols, CAND["BEDDAYS_AVAIL"]),
                TOT_BEDS      = _pick(cols, CAND["TOT_BEDS"]),
                AVG_BEDS      = _pick(cols, CAND["AVG_BEDS"]),
                STATE         = _pick(cols, CAND["STATE"]),
                URBAN         = _pick(cols, CAND["URBAN"]),
                HOME_OFFICE   = _pick(cols, CAND["HOME_OFFICE"]),
            )

            # Fallback: any column that contains both 'HOME' and 'OFFICE'
            if keep["HOME_OFFICE"] is None:
                ho_like = [c for c in cols if ("HOME" in c and "OFFICE" in c)]
                if len(ho_like) == 1:
                    keep["HOME_OFFICE"] = ho_like[0]
                elif len(ho_like) > 1:
                    nunique_sorted = sorted([(c, df[c].dropna().nunique()) for c in ho_like], key=lambda x: x[1])
                    keep["HOME_OFFICE"] = nunique_sorted[0][0] if nunique_sorted else None

            for k, v in keep.items():
                if v is None:
                    df[k] = pd.NA
                    keep[k] = k

            sub = df[[keep[k] for k in keep]].copy()
            sub.columns = list(keep.keys())
            sub["file_year"] = int(re.search(r"(\d{4})", fp.name).group(1))
            print(f"[read] {fp.name} (SAS) rows={len(sub):,} cols={sub.shape[1]}")
            frames.append(sub)
    else:
        for fp in csv_files:
            df = pd.read_csv(fp, low_memory=False)
            df.columns = [c.upper().strip() for c in df.columns]
            cols = list(df.columns)

            keep = dict(
                PRVDR_NUM     = _pick(cols, CAND["PRVDR_NUM"]),
                FY_BGN_DT     = _pick(cols, CAND["FY_BGN_DT"]),
                FY_END_DT     = _pick(cols, CAND["FY_END_DT"]),
                MRC_OWNERSHIP = _pick(cols, CAND["MRC_OWNERSHIP"]),
                PAT_DAYS_TOT  = _pick(cols, CAND["PAT_DAYS_TOT"]),
                PAT_DAYS_MCR  = _pick(cols, CAND["PAT_DAYS_MCR"]),
                PAT_DAYS_MCD  = _pick(cols, CAND["PAT_DAYS_MCD"]),
                BEDDAYS_AVAIL = _pick(cols, CAND["BEDDAYS_AVAIL"]),
                TOT_BEDS      = _pick(cols, CAND["TOT_BEDS"]),
                AVG_BEDS      = _pick(cols, CAND["AVG_BEDS"]),
                STATE         = _pick(cols, CAND["STATE"]),
                URBAN         = _pick(cols, CAND["URBAN"]),
                HOME_OFFICE   = _pick(cols, CAND["HOME_OFFICE"]),
            )

            if keep["HOME_OFFICE"] is None:
                ho_like = [c for c in cols if ("HOME" in c and "OFFICE" in c)]
                if len(ho_like) == 1:
                    keep["HOME_OFFICE"] = ho_like[0]
                elif len(ho_like) > 1:
                    nunique_sorted = sorted([(c, df[c].dropna().nunique()) for c in ho_like], key=lambda x: x[1])
                    keep["HOME_OFFICE"] = nunique_sorted[0][0] if nunique_sorted else None

            for k, v in keep.items():
                if v is None:
                    df[k] = pd.NA
                    keep[k] = k

            sub = df[[keep[k] for k in keep]].copy()
            sub.columns = list(keep.keys())
            sub["file_year"] = int(re.search(r"(\d{4})", fp.name).group(1))
            print(f"[read] {fp.name} (CSV) rows={len(sub):,} cols={sub.shape[1]}")
            frames.append(sub)

    if not frames:
        print("[controls] no MCR files found; returning empty controls")
        return pd.DataFrame(columns=[
            "cms_certification_number","month","ownership_type","pct_medicare","pct_medicaid",
            "num_beds","occupancy_rate","state","urban_rural","is_chain"
        ])

    raw = pd.concat(frames, ignore_index=True)

    # Types / cleaning
    raw["cms_certification_number"] = normalize_ccn_any(raw["PRVDR_NUM"])
    raw["FY_BGN_DT"] = pd.to_datetime(raw["FY_BGN_DT"], errors="coerce")
    raw["FY_END_DT"] = pd.to_datetime(raw["FY_END_DT"], errors="coerce")
    for c in ["PAT_DAYS_TOT","PAT_DAYS_MCR","PAT_DAYS_MCD","BEDDAYS_AVAIL","AVG_BEDS","TOT_BEDS"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # Ownership mapping
    CODE_TO_OWN = {
        "3":"Government","4":"Government","8":"Government","10":"Government",
        "11":"Government","12":"Government","13":"Government",
        "5":"For-profit","6":"For-profit","7":"For-profit","9":"For-profit",
        "1":"Nonprofit","2":"Nonprofit",
        "0": None
    }
    raw["_own_code"] = raw["MRC_OWNERSHIP"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    raw["ownership_type"] = raw["_own_code"].map(CODE_TO_OWN)

    # State
    raw["state"] = raw["STATE"].astype("string").str.strip().str.upper()
    raw.loc[raw["state"].isin(["", "NA", "NAN", "NONE"]), "state"] = pd.NA

    # Urban/Rural normalization ⇒ "Urban"/"Rural"
    def _norm_urban(x):
        if x is None or (isinstance(x, float) and pd.isna(x)): return pd.NA
        s = str(x).strip().upper()
        if s in {"U","URBAN","1","YES","Y","TRUE","T"}:  return "Urban"
        if s in {"R","RURAL","0","NO","N","FALSE","F","2"}:  return "Rural"
        return pd.NA
    raw["urban_rural"] = raw["URBAN"].apply(_norm_urban)

    # Chain (home office) ⇒ 1/0
    def _to_chain_flag(val):
        if val is None or (isinstance(val, float) and pd.isna(val)): return 0
        try:
            f = float(str(val).strip())
            return int(f != 0.0)
        except Exception:
            s = str(val).strip().upper()
            if s in {"", "0", "N", "NO", "NONE", "FALSE", "F"}: return 0
            return 1
    raw["is_chain"] = raw["HOME_OFFICE"].apply(_to_chain_flag).astype("Int8")

    # Fiscal period days (inclusive)
    period_days = (raw["FY_END_DT"] - raw["FY_BGN_DT"]).dt.days.add(1).where(lambda s: s > 0)

    # ---------- KEY FIXES ----------
    # num_beds: prefer TOTALBEDS (or BEDS), else AVG_BEDS, else derive from BEDDAYS_AVAIL / period_days
    raw["num_beds"] = np.select(
        [
            raw["TOT_BEDS"].notna(),
            raw["AVG_BEDS"].notna(),
            raw["BEDDAYS_AVAIL"].notna() & period_days.notna() & (period_days > 0),
        ],
        [
            raw["TOT_BEDS"],
            raw["AVG_BEDS"],
            raw["BEDDAYS_AVAIL"] / period_days
        ],
        default=np.nan
    )

    # occupancy_rate (%) = PAT_DAYS_TOT / BEDDAYS_AVAIL * 100  (primary);
    # fallback: PAT_DAYS_TOT / (num_beds * period_days) * 100
    den_primary = raw["BEDDAYS_AVAIL"]
    den_fallback = raw["num_beds"] * period_days
    occ = np.where(
        raw["PAT_DAYS_TOT"].notna() & den_primary.notna() & (den_primary > 0),
        (raw["PAT_DAYS_TOT"] / den_primary) * 100.0,
        np.where(
            raw["PAT_DAYS_TOT"].notna() & den_fallback.notna() & (den_fallback > 0),
            (raw["PAT_DAYS_TOT"] / den_fallback) * 100.0,
            np.nan
        )
    )
    raw["occupancy_rate"] = pd.to_numeric(occ, errors="coerce").clip(0, 100)

    # payer mix (clip 0..100)
    def _share(n, d): return pd.to_numeric(100.0 * (n / d), errors="coerce")
    raw["pct_medicare"] = _share(raw["PAT_DAYS_MCR"], raw["PAT_DAYS_TOT"]).clip(0, 100)
    raw["pct_medicaid"] = _share(raw["PAT_DAYS_MCD"], raw["PAT_DAYS_TOT"]).clip(0, 100)

    # Expand to monthly rows across fiscal span
    def month_range_df(start, end):
        if pd.isna(start) or pd.isna(end):
            return pd.DataFrame({"month":[]})
        s = pd.Period(start, "M").to_timestamp("s")
        e = pd.Period(end,   "M").to_timestamp("s")
        if e < s:
            s, e = e, s
        months = pd.period_range(s, e, freq="M").to_timestamp("s")
        return pd.DataFrame({"month": months})

    rows = []
    it = raw.dropna(subset=["cms_certification_number","FY_BGN_DT","FY_END_DT"]).itertuples(index=False)
    for r in it:
        months = month_range_df(r.FY_BGN_DT, r.FY_END_DT)
        if months.empty:
            continue
        block = months.copy()
        block["cms_certification_number"] = r.cms_certification_number
        block["ownership_type"] = getattr(r, "ownership_type", pd.NA)
        block["pct_medicare"]  = getattr(r, "pct_medicare", np.nan)
        block["pct_medicaid"]  = getattr(r, "pct_medicaid", np.nan)
        block["num_beds"]      = getattr(r, "num_beds", np.nan)
        block["occupancy_rate"]= getattr(r, "occupancy_rate", np.nan)
        block["state"]         = getattr(r, "state", pd.NA)
        block["urban_rural"]   = getattr(r, "urban_rural", pd.NA)
        block["is_chain"]      = getattr(r, "is_chain", pd.NA)
        rows.append(block)

    monthly = (pd.concat(rows, ignore_index=True)
               if rows else
               pd.DataFrame(columns=[
                   "cms_certification_number","month","ownership_type","pct_medicare","pct_medicaid",
                   "num_beds","occupancy_rate","state","urban_rural","is_chain"
               ]))

    # Deduplicate overlaps within CCN×month
    monthly = (monthly.sort_values(["cms_certification_number","month"])
                     .groupby(["cms_certification_number","month"], as_index=False)
                     .agg({
                         "ownership_type": lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
                         "state":         lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
                         "urban_rural":   lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
                         "is_chain":      "max",
                         "pct_medicare":  "mean",
                         "pct_medicaid":  "mean",
                         "num_beds":      "mean",
                         "occupancy_rate":"mean",
                     }))

    # Final hygiene & types
    for c in ["pct_medicare","pct_medicaid","occupancy_rate"]:
        if c in monthly.columns:
            monthly[c] = pd.to_numeric(monthly[c], errors="coerce").clip(0,100)
    if "num_beds" in monthly.columns:
        monthly["num_beds"] = pd.to_numeric(monthly["num_beds"], errors="coerce")
    if "is_chain" in monthly.columns:
        monthly["is_chain"] = monthly["is_chain"].fillna(0).astype("Int8")

    monthly["month"] = pd.to_datetime(monthly["month"], errors="coerce").dt.to_period("M").dt.to_timestamp("s")
    monthly["cms_certification_number"] = normalize_ccn_any(monthly["cms_certification_number"])

    print(f"[controls] built monthly controls: rows={len(monthly):,} CCNs={monthly['cms_certification_number'].nunique():,}")
    print("[controls] non-null coverage:",
          {c:int(monthly[c].notna().sum()) for c in ["num_beds","occupancy_rate","is_chain"] if c in monthly.columns})
    return monthly

# ============================== Load main inputs ==============================
lite = pd.read_csv(LITE_FP, dtype={"cms_certification_number":"string"}, low_memory=False)
mcr  = pd.read_csv(MCR_FP,  dtype={"cms_certification_number":"string"}, low_memory=False)
pbj  = pd.read_csv(PBJ_FP,  dtype={"cms_certification_number":"string"}, low_memory=False)

# CCN normalization
lite["cms_certification_number"] = normalize_ccn_any(lite["cms_certification_number"])
mcr["cms_certification_number"]  = normalize_ccn_any(mcr["cms_certification_number"])
pbj["cms_certification_number"]  = normalize_ccn_any(pbj["cms_certification_number"])

# ============================== Hospital filter ===============================
drop_hosp = load_hospital_dropset()
if len(drop_hosp):
    mcr_before = mcr["cms_certification_number"].nunique()
    mcr = mcr[~mcr["cms_certification_number"].isin(drop_hosp)].copy()
    print(f"[hospital filter] MCR: CCNs {mcr_before:,} -> {mcr['cms_certification_number'].nunique():,}")

    pbj_before = pbj["cms_certification_number"].nunique()
    pbj = pbj[~pbj["cms_certification_number"].isin(drop_hosp)].copy()
    print(f"[hospital filter] PBJ: CCNs {pbj_before:,} -> {pbj['cms_certification_number'].nunique():,}")

# ============================== First-event months (LITE vs MCR) =============
lite_counts = lite[["cms_certification_number","num_chows"]].copy() \
                if "num_chows" in lite.columns else \
              lite[["cms_certification_number"]].assign(num_chows=0)
lite_counts["num_chows"] = pd.to_numeric(lite_counts["num_chows"], errors="coerce").fillna(0).astype(int)
lite_counts = lite_counts.drop_duplicates("cms_certification_number").reset_index(drop=True)
lite_counts["first_event_month_lite"] = first_month_from_date_cols(lite, [r"^chow_date_\d+$", r"^chow_\d+_date$"])

mcr_counts = mcr[["cms_certification_number","n_chow"]].copy() \
               if "n_chow" in mcr.columns else \
             mcr[["cms_certification_number"]].assign(n_chow=0)
mcr_counts["n_chow"] = pd.to_numeric(mcr_counts["n_chow"], errors="coerce").fillna(0).astype(int)
mcr_counts = mcr_counts.drop_duplicates("cms_certification_number").reset_index(drop=True)
mcr_counts["first_event_month_mcr"] = first_month_from_date_cols(mcr, [r"^chow_\d+_date$"])

print("[diag] lite first-month non-null:", int(lite_counts["first_event_month_lite"].notna().sum()))
print("[diag] mcr  first-month non-null:", int(mcr_counts["first_event_month_mcr"].notna().sum()))

merged = (lite_counts
          .merge(mcr_counts, on="cms_certification_number", how="outer")
          .fillna({"num_chows":0, "n_chow":0}))

# ============================== Agreement logic (±6 months) ===================
def agreement_picker(r):
    label = "mismatch"
    change_month = pd.NaT
    a, b = r["first_event_month_lite"], r["first_event_month_mcr"]

    if (r["num_chows"]==0) and (r["n_chow"]==0):
        label = "match_0"
    elif (r["num_chows"]==1) and (r["n_chow"]==1):
        if within_k_months(a, b, k=6):
            label = "match_1_within_6m"
            change_month = pd.Period(a, "M").to_timestamp("s") if pd.notna(a) else (
                           pd.Period(b, "M").to_timestamp("s") if pd.notna(b) else pd.NaT)
        else:
            label = "match_1_diff_month"

    return pd.Series({"agreement": label, "change_month": change_month})

ag = merged.apply(agreement_picker, axis=1)
merged = pd.concat([merged, ag], axis=1)
agree = merged.loc[merged["agreement"].isin(["match_0","match_1_within_6m"])].copy()

print(f"[agree] match_0={int((agree['agreement']=='match_0').sum()):,} | "
      f"match_1_within_6m={int((agree['agreement']=='match_1_within_6m').sum()):,} | "
      f"total_agree={len(agree):,}")

# ============================== Prepare PBJ panel =============================
pbj_month_col = find_col(pbj.columns, ["year_month","month","pbj_month","date","period_month"])
if pbj_month_col is None:
    raise ValueError(f"[PBJ] Could not find month column in PBJ panel. Columns={list(pbj.columns)}")

if pbj_month_col.lower() == "year_month":
    pbj["month"] = pd.PeriodIndex(pbj["year_month"].astype(str), freq="M").to_timestamp("s")
else:
    pbj["month"] = to_monthstart(pbj[pbj_month_col])

pbj["cms_certification_number"] = normalize_ccn_any(pbj["cms_certification_number"])

# ============================== Merge agreement & dummies =====================
panel = pbj.merge(
    agree[["cms_certification_number","agreement","change_month"]],
    on="cms_certification_number",
    how="inner"
)

panel["treat_post"] = 0
is_treat = panel["agreement"].eq("match_1_within_6m") & panel["change_month"].notna()
panel.loc[is_treat, "treat_post"] = (panel.loc[is_treat, "month"] >= panel.loc[is_treat, "change_month"]).astype(int)

panel["event_time"] = np.nan
panel.loc[is_treat, "event_time"] = (
    (panel.loc[is_treat, "month"].values.astype("datetime64[M]") -
     panel.loc[is_treat, "change_month"].values.astype("datetime64[M]")).astype(int)
)

ever = (panel.groupby("cms_certification_number", as_index=False)["treat_post"].max()
             .rename(columns={"treat_post":"ever_treated"}))
panel = panel.merge(ever, on="cms_certification_number", how="left")

# ============================== BUILD + MERGE CONTROLS =======================
controls_monthly = build_mcr_controls_monthly()

if not controls_monthly.empty:
    # Merge controls by CCN×month
    panel = panel.merge(
        controls_monthly,
        on=["cms_certification_number","month"],
        how="left"
    )

    # --- Join audit BEFORE fills ---
    audit_cols = ["ownership_type","pct_medicare","pct_medicaid",
                  "num_beds","occupancy_rate","state","urban_rural","is_chain"]
    matched_any = panel[audit_cols].notna().any(axis=1)
    print(f"[controls] join matches on CCN×month: {matched_any.sum():,} rows ({matched_any.mean()*100:.1f}%)")
    pre_counts = {c: int(panel[c].notna().sum()) for c in audit_cols if c in panel.columns}
    print("[controls] non-null counts (pre-fill):", pre_counts)

    # Sort for fills
    panel = panel.sort_values(["cms_certification_number","month"]).reset_index(drop=True)

    # Categorical fills (as before)
    for cat_col in ["ownership_type","state","urban_rural","is_chain"]:
        if cat_col in panel.columns:
            panel[cat_col] = panel.groupby("cms_certification_number")[cat_col].transform(lambda s: s.ffill().bfill())
    if "is_chain" in panel.columns:
        panel["is_chain"] = panel["is_chain"].fillna(0).astype("Int8")

    # NEW: Numeric fills — carry forward/back within CCN to cover small gaps
    for num_col in ["num_beds","occupancy_rate","pct_medicare","pct_medicaid"]:
        if num_col in panel.columns:
            panel[num_col] = panel.groupby("cms_certification_number")[num_col].transform(lambda s: s.ffill().bfill())

    # Clip numeric ranges
    for c in ["pct_medicare","pct_medicaid","occupancy_rate"]:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce").clip(0, 100)
    if "num_beds" in panel.columns:
        panel["num_beds"] = pd.to_numeric(panel["num_beds"], errors="coerce")

    # --- Join audit AFTER fills ---
    post_counts = {c: int(panel[c].notna().sum()) for c in audit_cols if c in panel.columns}
    print("[controls] non-null counts (post-fill):", post_counts)

    cov = ((panel["pct_medicare"].notna()) | (panel["pct_medicaid"].notna())).mean()
    print(f"[controls] merged into panel — payer% coverage: {cov*100:.1f}%")
else:
    print("[controls] empty — no controls merged")

# ============================== Save =========================================
panel = panel.sort_values(["cms_certification_number","month"]).reset_index(drop=True)
panel.to_csv(OUT_FP, index=False)
cols_show = [c for c in ["cms_certification_number","month","agreement","change_month",
                         "treat_post","event_time","ownership_type","pct_medicare",
                         "pct_medicaid","num_beds","occupancy_rate","state","urban_rural","is_chain"]
             if c in panel.columns]
print(f"[save] {OUT_FP} rows={len(panel):,} cols={panel.shape[1]}")
print(panel[cols_show].head(12))
