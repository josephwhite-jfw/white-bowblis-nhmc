#!/usr/bin/env python
# coding: utf-8
# ─────────────────────────────────────────────────────────────────────────────
# Final PBJ Panel with CHOW Dummies + MCR Controls + Provider-Info Vars
# Adds:
#   • Merge provider-info monthly variables:
#       case_mix_total_num (already 2-quarter LEADed upstream),
#       ccrc_facility (0/1), sff_facility (0/1), sff_class
#   • Urban dummy (0/1) from urban_rural
#   • Case-mix bins (4 cols): national quartile/decile, state×month quartile/decile
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

PROV_COMBINED_TESTVARS = PROV_DIR / "provider_info_testvars_combined.csv"
PROV_COMBINED_DEFAULT  = PROV_DIR / "provider_info_combined.csv"

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

def rank_bins_pct(s: pd.Series, n_bins: int) -> pd.Series:
    pct = s.rank(method="average", pct=True)
    bins = np.ceil(pct * n_bins)
    bins = pd.to_numeric(bins, errors="coerce").clip(1, n_bins)
    bins = bins.where(s.notna())
    return bins.astype("Int16")

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
    Returns monthly controls with:
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

    CAND = dict(
        PRVDR_NUM       = ["PRVDR_NUM","provnum","prvdr_num","Provider Number"],
        FY_BGN_DT       = ["FY_BGN_DT","fy_bgn_dt","Cost Report Fiscal Year beginning date"],
        FY_END_DT       = ["FY_END_DT","fy_end_dt","Cost Report Fiscal Year ending date"],
        MRC_OWNERSHIP   = ["MRC_OWNERSHIP","MRC_ownership","mrc_ownership","MRC_ownership_code"],
        PAT_DAYS_TOT    = ["S3_1_PATDAYS_TOTAL","PATDAYS_TOTAL","PATIENT_DAYS_TOTAL"],
        PAT_DAYS_MCR    = ["S3_1_PATDAYS_MEDICARE","PATDAYS_MEDICARE","PATIENT_DAYS_MEDICARE"],
        PAT_DAYS_MCD    = ["S3_1_PATDAYS_MEDICAID","PATDAYS_MEDICAID","PATIENT_DAYS_MEDICAID"],
        BEDDAYS_AVAIL   = ["S3_1_BEDDAYS_AVAL","BEDDAYS_AVAL","S3_1_BED_DAYS_AVAIL","BED_DAYS_AVAIL"],
        TOT_BEDS        = ["S3_1_TOTALBEDS","TOTAL_BEDS","TOT_BEDS","BEDS","S3_1_BEDS"],
        AVG_BEDS        = ["AVG_BEDS","AVERAGE_BEDS","AVG_INPT_BEDS","S3_1_AVG_BEDS"],
        STATE           = ["MCR_STATE","STATE","PROV_STATE","STATE_CD","PROV_STATE_CD"],
        URBAN           = ["MCR_URBAN","URBAN_RURAL","URBAN_RURAL_INDICATOR","URBAN_IND","URBAN","URBRUR"],
        HOME_OFFICE     = ["MCR_HOME_OFFICE","HOME_OFFICE","HOME_OFFICE_IND","HOME_OFFICE_INDICATOR","HOME_OFFICE_FLAG"]
    )
    def _pick(cols, names): return find_col(cols, CAND[names] if isinstance(names,str) else names)

    frames = []
    if use_sas:
        import pyreadstat
        for fp in sas_files:
            df, _ = pyreadstat.read_sas7bdat(str(fp), disable_datetime_conversion=0)
            df.columns = [c.upper().strip() for c in df.columns]
            cols = list(df.columns)

            keep = dict(
                PRVDR_NUM     = _pick(cols, "PRVDR_NUM"),
                FY_BGN_DT     = _pick(cols, "FY_BGN_DT"),
                FY_END_DT     = _pick(cols, "FY_END_DT"),
                MRC_OWNERSHIP = _pick(cols, "MRC_OWNERSHIP"),
                PAT_DAYS_TOT  = _pick(cols, "PAT_DAYS_TOT"),
                PAT_DAYS_MCR  = _pick(cols, "PAT_DAYS_MCR"),
                PAT_DAYS_MCD  = _pick(cols, "PAT_DAYS_MCD"),
                BEDDAYS_AVAIL = _pick(cols, "BEDDAYS_AVAIL"),
                TOT_BEDS      = _pick(cols, "TOT_BEDS"),
                AVG_BEDS      = _pick(cols, "AVG_BEDS"),
                STATE         = _pick(cols, "STATE"),
                URBAN         = _pick(cols, "URBAN"),
                HOME_OFFICE   = _pick(cols, "HOME_OFFICE"),
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
            print(f"[read] {fp.name} (SAS) rows={len(sub):,} cols={sub.shape[1]}")
            frames.append(sub)
    else:
        for fp in csv_files:
            df = pd.read_csv(fp, low_memory=False)
            df.columns = [c.upper().strip() for c in df.columns]
            cols = list(df.columns)

            keep = dict(
                PRVDR_NUM     = _pick(cols, "PRVDR_NUM"),
                FY_BGN_DT     = _pick(cols, "FY_BGN_DT"),
                FY_END_DT     = _pick(cols, "FY_END_DT"),
                MRC_OWNERSHIP = _pick(cols, "MRC_OWNERSHIP"),
                PAT_DAYS_TOT  = _pick(cols, "PAT_DAYS_TOT"),
                PAT_DAYS_MCR  = _pick(cols, "PAT_DAYS_MCR"),
                PAT_DAYS_MCD  = _pick(cols, "PAT_DAYS_MCD"),
                BEDDAYS_AVAIL = _pick(cols, "BEDDAYS_AVAIL"),
                TOT_BEDS      = _pick(cols, "TOT_BEDS"),
                AVG_BEDS      = _pick(cols, "AVG_BEDS"),
                STATE         = _pick(cols, "STATE"),
                URBAN         = _pick(cols, "URBAN"),
                HOME_OFFICE   = _pick(cols, "HOME_OFFICE"),
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

    # Ownership bucketing
    def map_ownership_bucket(code_str: str):
        if code_str is None or (isinstance(code_str, float) and pd.isna(code_str)):
            return None
        s = str(code_str).strip().upper().replace(".0", "")
        if s in {"1","2"}:                  return "Nonprofit"
        if s in {"3","4","5","6"}:          return "For-profit"
        if s in {"7","8","9","10","11","12","13"}: return "Government"
        return None

    raw["_own_code"] = raw["MRC_OWNERSHIP"]
    raw["ownership_type"] = raw["_own_code"].map(map_ownership_bucket)

    # State / urban-rural / chain
    raw["state"] = raw["STATE"].astype("string").str.strip().str.upper()
    raw.loc[raw["state"].isin(["", "NA", "NAN", "NONE"]), "state"] = pd.NA

    def _norm_urban(x):
        if x is None or (isinstance(x, float) and pd.isna(x)): return pd.NA
        s = str(x).strip().upper()
        if s in {"U","URBAN","1","YES","Y","TRUE","T"}:  return "Urban"
        if s in {"R","RURAL","0","NO","N","FALSE","F","2"}:  return "Rural"
        return pd.NA
    raw["urban_rural"] = raw["URBAN"].apply(_norm_urban)

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

    # Period derived stuff
    period_days = (raw["FY_END_DT"] - raw["FY_BGN_DT"]).dt.days.add(1).where(lambda s: s > 0)

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

# ============================== Provider-info loader ==========================
def load_provider_info_monthly():
    src = PROV_COMBINED_TESTVARS if PROV_COMBINED_TESTVARS.exists() else PROV_COMBINED_DEFAULT
    if not src.exists():
        print(f"[provider-info] WARNING: {src} not found — skipping provider-info merge")
        return pd.DataFrame(columns=[
            "cms_certification_number","month",
            "case_mix_total_num","ccrc_facility","sff_facility","sff_class"
        ])
    df = pd.read_csv(src, dtype=str, low_memory=False)
    # Standardize
    df.columns = [c.strip().lower() for c in df.columns]
    ccn_col  = find_col(df.columns, ["cms_certification_number","ccn","provnum","provider_number","federal_provider_number"])
    date_col = find_col(df.columns, ["date","month","period","as_of_date"])
    cm_col   = find_col(df.columns, ["case_mix_total_num","case_mix_total","cm_total","exp_total"])
    ccrc_col = find_col(df.columns, ["ccrc_facility","continuing_care_retirement_community"])
    sff_fac  = find_col(df.columns, ["sff_facility"])
    sff_cls  = find_col(df.columns, ["sff_class","sff_status"])
    if ccn_col is None or date_col is None:
        print("[provider-info] columns not found — skipping")
        return pd.DataFrame(columns=[
            "cms_certification_number","month",
            "case_mix_total_num","ccrc_facility","sff_facility","sff_class"
        ])

    out = pd.DataFrame({
        "cms_certification_number": normalize_ccn_any(df[ccn_col]),
        "month": to_monthstart(df[date_col]),
    })

    # case mix numeric (should already be 2-quarter LEADed in the combined file)
    out["case_mix_total_num"] = pd.to_numeric(df[cm_col], errors="coerce") if cm_col else np.nan

    # ccrc_facility -> 0/1 Int8
    if ccrc_col:
        s = df[ccrc_col].astype("string").str.strip().str.lower()
        out["ccrc_facility"] = s.map({
            "1":1,"true":1,"t":1,"yes":1,"y":1,
            "0":0,"false":0,"f":0,"no":0,"n":0
        }).astype("Int8")
    else:
        out["ccrc_facility"] = pd.Series([pd.NA]*len(out), dtype="Int8")

    # sff_facility -> 0/1 Int8 (current or candidate = 1 in provider combine)
    if sff_fac:
        out["sff_facility"] = pd.to_numeric(df[sff_fac], errors="coerce").astype("Int8")
    else:
        out["sff_facility"] = pd.Series([pd.NA]*len(out), dtype="Int8")

    # sff_class (keep as reference)
    out["sff_class"] = df[sff_cls].astype("string") if sff_cls else pd.Series([pd.NA]*len(out), dtype="string")

    # Drop exact dupes and return
    out = (out.dropna(subset=["cms_certification_number","month"])
              .drop_duplicates(["cms_certification_number","month"])
              .reset_index(drop=True))
    print(f"[provider-info] loaded: rows={len(out):,} CCNs={out['cms_certification_number'].nunique():,}")
    return out

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

# === Use OVERLAP (inner join) as the agreement base ===
merged_overlap = lite_counts.merge(mcr_counts, on="cms_certification_number", how="inner")

def _cat(n):
    try: n = int(n)
    except Exception: return "0"
    if n <= 0: return "0"
    if n == 1: return "1"
    return "2+"

_overlap = merged_overlap.copy()
_overlap["own_cat"] = _overlap["num_chows"].map(_cat)
_overlap["mcr_cat"] = _overlap["n_chow"].map(_cat)
_overlap_ctab = pd.crosstab(_overlap["own_cat"], _overlap["mcr_cat"]).reindex(
    index=["0","1","2+"], columns=["0","1","2+"], fill_value=0
)
_overlap_ctab["Total"] = _overlap_ctab[["0","1","2+"]].sum(axis=1)
print("\n=== Crosstab 0/1/2+ (OVERLAP base) ===")
print(_overlap_ctab.to_string())

# ============================== Agreement logic (±6 months) ===================
def agreement_picker(r):
    a, b = r["first_event_month_lite"], r["first_event_month_mcr"]
    label, change_month = "mismatch", pd.NaT
    if (r["num_chows"] == 0) and (r["n_chow"] == 0):
        label = "match_0"
    elif (r["num_chows"] == 1) and (r["n_chow"] == 1):
        if within_k_months(a, b, k=6):
            label = "match_1_within_6m"
            change_month = pd.Period(a if pd.notna(a) else b, "M").to_timestamp("s")
        else:
            label = "match_1_diff_month"
    return pd.Series({"agreement": label, "change_month": change_month})

ag = merged_overlap.apply(agreement_picker, axis=1)
agree = pd.concat([merged_overlap[["cms_certification_number","num_chows","n_chow",
                                   "first_event_month_lite","first_event_month_mcr"]],
                   ag], axis=1)

print(f"[agree] match_0={int((agree['agreement']=='match_0').sum()):,} | "
      f"match_1_within_6m={int((agree['agreement']=='match_1_within_6m').sum()):,} | "
      f"total_agree={int(agree['agreement'].isin(['match_0','match_1_within_6m']).sum()):,}")

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

# ============================== ANALYTIC FILTER (match on 0 or 1) ============
MATCH_SET = {"match_0", "match_1_within_6m"}
if "agreement" not in panel.columns:
    raise KeyError("Expected 'agreement' column not found in panel after merge.")
panel = panel[panel["agreement"].isin(MATCH_SET)].copy()
n_fac = panel["cms_certification_number"].nunique()
print(f"[analytic filter] providers kept = {n_fac:,} (agreement in {sorted(MATCH_SET)})")

# ============================== BUILD + MERGE CONTROLS =======================
controls_monthly = build_mcr_controls_monthly()

if not controls_monthly.empty:
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

    # Categorical fills
    for cat_col in ["ownership_type","state","urban_rural","is_chain"]:
        if cat_col in panel.columns:
            panel[cat_col] = panel.groupby("cms_certification_number")[cat_col].transform(lambda s: s.ffill().bfill())
    if "is_chain" in panel.columns:
        panel["is_chain"] = panel["is_chain"].fillna(0).astype("Int8")

    # Numeric fills
    for num_col in ["num_beds","occupancy_rate","pct_medicare","pct_medicaid"]:
        if num_col in panel.columns:
            panel[num_col] = panel.groupby("cms_certification_number")[num_col].transform(lambda s: s.ffill().bfill())

    # Clip numeric ranges
    for c in ["pct_medicare","pct_medicaid","occupancy_rate"]:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce").clip(0, 100)
    if "num_beds" in panel.columns:
        panel["num_beds"] = pd.to_numeric(panel["num_beds"], errors="coerce")

    # ✅ CHANGED: Ownership dummies with FOR-PROFIT as the REFERENCE
    if "ownership_type" in panel.columns:
        ot = panel["ownership_type"].astype("string").str.strip().str.lower()
        # normalize minor variants
        ot = (ot.str.replace(r"[\s_]+", "-", regex=True)
                .str.replace(r"^non[- ]?profit$", "nonprofit", regex=True)
                .str.replace(r"^for[- ]?profit$", "for-profit", regex=True))

        panel["non_profit"] = ot.eq("nonprofit").astype("Int8")
        panel["government"] = ot.eq("government").astype("Int8")

        # drop for_profit to enforce reference category (optional but recommended)
        if "for_profit" in panel.columns:
            panel = panel.drop(columns=["for_profit"])

    post_counts = {c: int(panel[c].notna().sum()) for c in audit_cols if c in panel.columns}
    print("[controls] non-null counts (post-fill):", post_counts)
else:
    print("[controls] empty — no controls merged")

# ============================== MERGE PROVIDER-INFO VARS ======================
prov_info = load_provider_info_monthly()
if not prov_info.empty:
    keep_cols = ["cms_certification_number","month","case_mix_total_num","ccrc_facility","sff_facility","sff_class"]
    panel = panel.merge(prov_info[keep_cols], on=["cms_certification_number","month"], how="left")

    # Ensure 0/1 Int8 dummies
    if "ccrc_facility" in panel.columns:
        panel["ccrc_facility"] = pd.to_numeric(panel["ccrc_facility"], errors="coerce").astype("Int8")
    if "sff_facility" in panel.columns:
        panel["sff_facility"] = pd.to_numeric(panel["sff_facility"], errors="coerce").astype("Int8")

    print("[provider-info] matches case-mix:", int(panel["case_mix_total_num"].notna().sum()) if "case_mix_total_num" in panel.columns else 0)
    print("[provider-info] ccrc matched:", int(panel["ccrc_facility"].notna().sum()) if "ccrc_facility" in panel.columns else 0)
    print("[provider-info] sff matched:", int(panel["sff_facility"].notna().sum()) if "sff_facility" in panel.columns else 0)
else:
    print("[provider-info] empty — no provider-info variables merged")

# ============================== URBAN 0/1 DUMMY ===============================
def make_urban_dummy(df):
    if "urban_rural" in df.columns:
        s = df["urban_rural"].astype("string").str.strip().str.title()
        return s.map({"Urban":1, "Rural":0}).astype("Int8")
    return pd.Series([pd.NA]*len(df), dtype="Int8")

panel["urban"] = make_urban_dummy(panel)

# ============================== CASE-MIX QUANTILES ============================
if "case_mix_total_num" in panel.columns:
    panel["case_mix_quartile_nat"] = (
        panel.groupby("month", observed=True)["case_mix_total_num"]
             .transform(lambda s: rank_bins_pct(s, 4)).astype("Int16")
    )
    panel["case_mix_decile_nat"] = (
        panel.groupby("month", observed=True)["case_mix_total_num"]
             .transform(lambda s: rank_bins_pct(s, 10)).astype("Int16")
    )
    if "state" in panel.columns:
        mask_state = panel["state"].notna()
        panel.loc[mask_state, "case_mix_quartile_state"] = (
            panel[mask_state].groupby(["month","state"], observed=True)["case_mix_total_num"]
                            .transform(lambda s: rank_bins_pct(s, 4)).astype("Int16")
        )
        panel.loc[mask_state, "case_mix_decile_state"] = (
            panel[mask_state].groupby(["month","state"], observed=True)["case_mix_total_num"]
                            .transform(lambda s: rank_bins_pct(s, 10)).astype("Int16")
        )
    print(f"[case-mix] non-null rows={int(panel['case_mix_total_num'].notna().sum()):,}")
else:
    print("[case-mix] WARNING: 'case_mix_total_num' not present — skipping quantile columns")

# ============================== Save =========================================
panel = panel.sort_values(["cms_certification_number","month"]).reset_index(drop=True)
panel.to_csv(OUT_FP, index=False)

# ✅ CHANGED: show non_profit + government (for_profit removed)
cols_show = [c for c in [
    "cms_certification_number","month","agreement","change_month",
    "treat_post","event_time",
    "ownership_type","non_profit","government",
    "pct_medicare","pct_medicaid","num_beds","occupancy_rate","state","urban_rural","urban","is_chain",
    "ccrc_facility","sff_facility","sff_class","case_mix_total_num",
    "case_mix_quartile_nat","case_mix_decile_nat","case_mix_quartile_state","case_mix_decile_state"
] if c in panel.columns]

print(f"[save] {OUT_FP} rows={len(panel):,} cols={panel.shape[1]}")
print(panel[cols_show].head(12))