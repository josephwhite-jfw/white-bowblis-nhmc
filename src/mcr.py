#!/usr/bin/env python
# coding: utf-8
# -----------------------------------------------------------------------------
# Medicare Cost Reports → Monthly Controls (no panel build)
#
# Output: data/interim/mcr.csv  with columns:
#   cms_certification_number, quarter, year_month,
#   pct_medicare, pct_medicaid, num_beds, occupancy_rate,
#   urban, chain, state, non_profit, government
#
# Notes:
#   • CCN normalized (alphanumeric OK; pad if all digits)
#   • Monthly expansion across fiscal year [FY_BGN_DT .. FY_END_DT]
#   • occupancy_rate = patient_days_total / beddays_available (primary),
#                      fallback = patient_days_total / (num_beds * FY_days)
#   • urban = 1 if explicit URBAN, 0 if explicit RURAL (else NA left through)
#   • chain = derived ONLY from MCR_homeoffice (Y/Yes/1 → 1; N/No/0/blank → 0)
#   • ownership_type buckets → non_profit (1/0), government (1/0); for-profit is ref
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

RAW_DIR     = Path(os.getenv("NH_DATA_DIR", PROJECT_ROOT / "data" / "raw")).resolve()
MCR_DIR     = RAW_DIR / "medicare-cost-reports"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"; INTERIM_DIR.mkdir(parents=True, exist_ok=True)
OUT_FP      = INTERIM_DIR / "mcr.csv"

print(f"[paths] RAW_DIR={RAW_DIR}")
print(f"[paths] MCR_DIR={MCR_DIR}")
print(f"[out]   {OUT_FP}")

# ============================== Helpers =======================================
def normalize_ccn_any(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("").str.strip().str.upper()
    s = s.str.replace(r"[ \-\/\.]", "", regex=True)
    is_digits = s.str.fullmatch(r"\d+")
    s = s.mask(is_digits, s.str.zfill(6))
    s = s.replace({"": pd.NA})
    return s

def find_col_case_insensitive(cols, candidates):
    targets = [candidates] if isinstance(candidates, str) else list(candidates)
    lower_map = {c.lower().strip(): c for c in cols}
    for cand in targets:
        if cand and cand.lower().strip() in lower_map:
            return lower_map[cand.lower().strip()]
    return None

def map_ownership_bucket(code_str: str):
    if code_str is None or (isinstance(code_str, float) and pd.isna(code_str)):
        return None
    s = str(code_str).strip().upper().replace(".0", "")
    # CMS crosswalk commonly: 1-2 nonprofit; 3-6 for-profit; 7-13 government
    if s in {"1","2"}: return "Nonprofit"
    if s in {"3","4","5","6"}: return "For-profit"
    if s in {"7","8","9","10","11","12","13"}: return "Government"
    return None

def _share(n, d):
    return pd.to_numeric(100.0 * (n / d), errors="coerce")

def month_range_df(start, end):
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame({"month":[]})
    s = pd.Period(start, "M").to_timestamp("s")
    e = pd.Period(end,   "M").to_timestamp("s")
    if e < s:
        s, e = e, s
    months = pd.period_range(s, e, freq="M").to_timestamp("s")
    return pd.DataFrame({"month": months})

def norm_urban(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return pd.NA
    s = str(x).strip().upper()
    if s in {"U","URBAN","1","YES","Y","TRUE","T"}:  return 1
    if s in {"R","RURAL","0","NO","N","FALSE","F","2"}:  return 0
    return pd.NA

def chain_from_homeoffice(val):
    """
    EXACT logic based on your data audit:
      'Y','YES','1','T','TRUE'  -> 1
      'N','NO','0','F','FALSE','' (blank) -> 0
      numeric nonzero -> 1, numeric zero/NA -> 0
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0
    s = str(val).strip().upper()
    if s in {"Y","YES","1","T","TRUE"}:
        return 1
    if s in {"N","NO","0","F","FALSE",""}:
        return 0
    try:
        f = float(s)
        return int(f != 0.0)
    except Exception:
        return 0

# ============================== Load & Select =================================
# Prefer SAS; fallback to CSV
sas_files = sorted(MCR_DIR.glob("mcr_flatfile_20??.sas7bdat"))
xpt_files = sorted(MCR_DIR.glob("mcr_flatfile_20??.xpt"))
csv_files = sorted(MCR_DIR.glob("mcr_flatfile_20??.csv"))

use_sas = len(sas_files + xpt_files) > 0
try:
    import pyreadstat  # noqa: F401
except Exception:
    if use_sas:
        print("[read] pyreadstat not available; falling back to CSV-only")
    use_sas = False

# Canonical targets (STRICT to provided names)
TARGET_SETS = dict(
    PRVDR_NUM      = ["PRVDR_NUM","provnum","prvdr_num","Provider Number"],
    FY_BGN_DT      = ["FY_BGN_DT","fy_bgn_dt","Cost Report Fiscal Year beginning date"],
    FY_END_DT      = ["FY_END_DT","fy_end_dt","Cost Report Fiscal Year ending date"],
    MRC_OWNERSHIP  = ["MRC_OWNERSHIP"],
    PAT_DAYS_TOT   = ["S3_1_patdays_total","S3_1_PATDAYS_TOTAL"],
    PAT_DAYS_MCR   = ["S3_1_patdays_medicare","S3_1_PATDAYS_MEDICARE"],
    PAT_DAYS_MCD   = ["S3_1_patdays_medicaid","S3_1_PATDAYS_MEDICAID"],
    BEDDAYS_AVAIL  = ["S3_1_beddays_aval","S3_1_BEDDAYS_AVAL"],
    TOT_BEDS       = ["S3_1_beds","S3_1_BEDS"],
    STATE          = ["MCR_STATE"],
    URBAN          = ["MCR_URBAN"],
    MCR_homeoffice = ["MCR_homeoffice"]
)

def read_one_file(fp: Path) -> pd.DataFrame:
    if fp.suffix.lower() in {".sas7bdat", ".xpt"} and use_sas:
        if fp.suffix.lower() == ".sas7bdat":
            _, meta = pyreadstat.read_sas7bdat(fp, metadataonly=True)
        else:
            _, meta = pyreadstat.read_xport(fp, metadataonly=True)
        cols = list(meta.column_names)
    else:
        if fp.suffix.lower() == ".csv":
            head = pd.read_csv(fp, nrows=0, low_memory=False)
            cols = list(head.columns)
        else:
            fmt = "sas7bdat" if fp.suffix.lower() == ".sas7bdat" else "xport"
            head = pd.read_sas(fp, format=fmt, encoding="latin1", nrows=1)
            cols = list(head.columns)

    actual = {}
    for key, cand in TARGET_SETS.items():
        nm = find_col_case_insensitive(cols, cand)
        if nm is not None:
            actual[key] = nm

    essentials = ["PRVDR_NUM","FY_BGN_DT","FY_END_DT"]
    for k in essentials:
        if k not in actual:
            actual[k] = None

    select_cols_actual = [c for c in actual.values() if c is not None]
    if fp.suffix.lower() in {".sas7bdat", ".xpt"} and use_sas:
        if fp.suffix.lower() == ".sas7bdat":
            df, _ = pyreadstat.read_sas7bdat(fp, usecols=select_cols_actual, disable_datetime_conversion=0)
        else:
            df, _ = pyreadstat.read_xport(fp, usecols=select_cols_actual, disable_datetime_conversion=0)
    else:
        if fp.suffix.lower() == ".csv":
            df = pd.read_csv(fp, usecols=select_cols_actual if select_cols_actual else None, low_memory=False)
        else:
            fmt = "sas7bdat" if fp.suffix.lower() == ".sas7bdat" else "xport"
            df = pd.read_sas(fp, format=fmt, encoding="latin1")
            if select_cols_actual:
                df = df[select_cols_actual].copy()

    rename_map = {actual[k]: k for k in actual if actual[k] is not None}
    df = df.rename(columns=rename_map)

    for k in TARGET_SETS.keys():
        if k not in df.columns:
            df[k] = pd.NA

    print(f"[read] {fp.name} rows={len(df):,} cols={len(df.columns)}")
    return df

files = sorted(sas_files + xpt_files + csv_files)
if not files:
    raise FileNotFoundError(f"No MCR flatfiles found in {MCR_DIR}")

frames = [read_one_file(fp) for fp in files]
raw = pd.concat(frames, ignore_index=True, sort=False).copy()

# ============================== Normalize & Types =============================
raw["cms_certification_number"] = normalize_ccn_any(raw["PRVDR_NUM"])
raw["FY_BGN_DT"] = pd.to_datetime(raw["FY_BGN_DT"], errors="coerce")
raw["FY_END_DT"] = pd.to_datetime(raw["FY_END_DT"], errors="coerce")

for c in ["PAT_DAYS_TOT","PAT_DAYS_MCR","PAT_DAYS_MCD","BEDDAYS_AVAIL","TOT_BEDS"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")

raw["ownership_type"] = raw["MRC_OWNERSHIP"].map(map_ownership_bucket)

# State / urban
raw["state"] = raw["STATE"].astype("string").str.strip().str.upper()
raw.loc[raw["state"].isin(["", "NA", "NAN", "NONE"]), "state"] = pd.NA
raw["urban"] = raw["URBAN"].apply(norm_urban).astype("Int8")

# Chain ONLY from MCR_homeoffice
raw["chain"] = raw["MCR_homeoffice"].apply(chain_from_homeoffice).astype("Int8")

# Beds (from S3_1_BEDS only)
raw["num_beds"] = raw["TOT_BEDS"]

# ============================== OUTLIER / MANUAL CORRECTIONS ==================
# Helpers for period overlap with FY rows
def _ts_month_start(ym: str) -> pd.Timestamp:
    return pd.Period(ym, "M").to_timestamp("s")

def _ts_month_end(ym: str) -> pd.Timestamp:
    return pd.Period(ym, "M").to_timestamp("s") + pd.offsets.MonthEnd(0)

def apply_value_corrections(df: pd.DataFrame, corrections: list[dict]) -> int:
    """
    Each correction is a dict:
      {"ccn":"015417", "start":"2018/04", "end":"2019/04", "col":"num_beds", "value":75}
    Applied at the FY-row level if [FY_BGN_DT .. FY_END_DT] overlaps [start..end].
    Returns number of rows updated.
    """
    total = 0
    for item in corrections:
        ccn = str(item["ccn"])
        col = str(item["col"])
        start, end = item["start"], item["end"]
        val = item["value"]
        ccn_norm = normalize_ccn_any(pd.Series([ccn])).iloc[0]
        s_ts, e_ts = _ts_month_start(start), _ts_month_end(end)
        m = (
            df["cms_certification_number"].eq(ccn_norm) &
            df["FY_BGN_DT"].notna() & df["FY_END_DT"].notna() &
            (df["FY_BGN_DT"] <= e_ts) & (df["FY_END_DT"] >= s_ts)
        )
        if m.any():
            df.loc[m, col] = val
            total += int(m.sum())
    return total

# (A) Current hard-coded corrections for beds
CORR_BEDS = [
    {"ccn":"015417", "start":"2018/04", "end":"2019/04", "col":"num_beds", "value": 75},
    {"ccn":"056337", "start":"2017/01", "end":"2017/12", "col":"num_beds", "value":142},
    {"ccn":"105782", "start":"2017/01", "end":"2017/12", "col":"num_beds", "value":120},
    {"ccn":"135080", "start":"2018/01", "end":"2019/12", "col":"num_beds", "value": 60},
    {"ccn":"145816", "start":"2024/01", "end":"2024/06", "col":"num_beds", "value":203},
    {"ccn":"235022", "start":"2019/01", "end":"2019/12", "col":"num_beds", "value": 92},
    {"ccn":"235157", "start":"2021/01", "end":"2021/12", "col":"num_beds", "value":104},
    {"ccn":"235638", "start":"2019/01", "end":"2019/12", "col":"num_beds", "value": 77},
    {"ccn":"425129", "start":"2020/04", "end":"2020/12", "col":"num_beds", "value":108},
]

updated_beds = apply_value_corrections(raw, CORR_BEDS)
print(f"[beds corrections] updated FY rows: {updated_beds}")

# (B) PLACEHOLDER: corrections to raw input fields (e.g., PAT_DAYS_TOT, BEDDAYS_AVAIL, etc.)
CORR_INPUTS = [
    # Example:
    # {"ccn":"123456", "start":"2020/01", "end":"2020/03", "col":"PAT_DAYS_TOT", "value": 9999},
]
updated_inputs = apply_value_corrections(raw, CORR_INPUTS)
if updated_inputs:
    print(f"[input corrections] updated FY rows: {updated_inputs}")

# Occupancy rate (primary + fallback) — runs AFTER beds corrections
fy_days = (raw["FY_END_DT"] - raw["FY_BGN_DT"]).dt.days.add(1).where(lambda s: s > 0)
primary = np.where(
    raw["PAT_DAYS_TOT"].notna() & raw["BEDDAYS_AVAIL"].notna() & (raw["BEDDAYS_AVAIL"] > 0),
    (raw["PAT_DAYS_TOT"] / raw["BEDDAYS_AVAIL"]) * 100.0,
    np.nan
)
fallback = np.where(
    raw["PAT_DAYS_TOT"].notna() & raw["num_beds"].notna() & fy_days.notna() & (raw["num_beds"] * fy_days > 0),
    (raw["PAT_DAYS_TOT"] / (raw["num_beds"] * fy_days)) * 100.0,
    np.nan
)
raw["occupancy_rate"] = pd.to_numeric(np.where(np.isnan(primary), fallback, primary), errors="coerce").clip(0, 100)

# Shares
raw["pct_medicare"] = _share(raw["PAT_DAYS_MCR"], raw["PAT_DAYS_TOT"]).clip(0, 100)
raw["pct_medicaid"] = _share(raw["PAT_DAYS_MCD"], raw["PAT_DAYS_TOT"]).clip(0, 100)

# (C) PLACEHOLDER: corrections to derived outputs (e.g., num_beds, occupancy_rate, pct_* , urban/chain/state)
CORR_DERIVED = [
    # Example:
    # {"ccn":"123456", "start":"2021/05", "end":"2021/07", "col":"occupancy_rate", "value": 88.0},
    # {"ccn":"123456", "start":"2021/05", "end":"2021/07", "col":"pct_medicare",   "value": 55.0},
]
updated_derived = apply_value_corrections(raw, CORR_DERIVED)
if updated_derived:
    print(f"[derived corrections] updated FY rows: {updated_derived}")

# ============================== Expand to Monthly =============================
eligible = raw.dropna(subset=["cms_certification_number","FY_BGN_DT","FY_END_DT"]).copy()

rows = []
for r in eligible.itertuples(index=False):
    months = month_range_df(r.FY_BGN_DT, r.FY_END_DT)
    if months.empty:
        continue
    block = months.copy()
    block["cms_certification_number"] = getattr(r, "cms_certification_number")
    block["state"]         = getattr(r, "state", pd.NA)
    block["urban"]         = getattr(r, "urban", pd.NA)
    block["chain"]         = getattr(r, "chain", pd.NA)
    block["num_beds"]      = getattr(r, "num_beds", np.nan)
    block["occupancy_rate"]= getattr(r, "occupancy_rate", np.nan)
    block["pct_medicare"]  = getattr(r, "pct_medicare", np.nan)
    block["pct_medicaid"]  = getattr(r, "pct_medicaid", np.nan)
    block["ownership_type"]= getattr(r, "ownership_type", None)
    rows.append(block)

monthly = (pd.concat(rows, ignore_index=True)
           if rows else
           pd.DataFrame(columns=[
               "cms_certification_number","month","state","urban","chain",
               "num_beds","occupancy_rate","pct_medicare","pct_medicaid","ownership_type"
           ]))

# Deduplicate overlaps within CCN×month (same behavior as before)
monthly = (monthly.sort_values(["cms_certification_number","month"])
                 .groupby(["cms_certification_number","month"], as_index=False)
                 .agg({
                     "state":          lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
                     "urban":          "max",
                     "chain":          "max",
                     "num_beds":       "mean",
                     "occupancy_rate": "mean",
                     "pct_medicare":   "mean",
                     "pct_medicaid":   "mean",
                     "ownership_type": lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
                 }))

# Ownership dummies (For-profit as reference)
ot = monthly["ownership_type"].astype("string").str.strip().str.lower()
ot = (ot.str.replace(r"[\s_]+", "-", regex=True)
         .str.replace(r"^non[- ]?profit$", "nonprofit", regex=True)
         .str.replace(r"^for[- ]?profit$", "for-profit", regex=True))

monthly["non_profit"] = ot.eq("nonprofit").astype("Int8")
monthly["government"] = ot.eq("government").astype("Int8")

# Period labels
monthly["year_month"] = monthly["month"].dt.strftime("%Y/%m")
monthly["quarter"]    = monthly["month"].dt.to_period("Q").astype(str).str.replace("Q", "Q", regex=False)

# Coerce numeric ranges/types
for c in ["pct_medicare","pct_medicaid","occupancy_rate"]:
    monthly[c] = pd.to_numeric(monthly[c], errors="coerce").clip(0, 100)
monthly["num_beds"] = pd.to_numeric(monthly["num_beds"], errors="coerce")

# Reorder / keep only requested columns
keep = [
    "cms_certification_number", "quarter", "year_month",
    "pct_medicare", "pct_medicaid", "num_beds", "occupancy_rate",
    "urban", "chain", "state", "non_profit", "government"
]
monthly = monthly[keep].sort_values(["cms_certification_number","year_month"]).reset_index(drop=True)

# Save
monthly.to_csv(OUT_FP, index=False)
print(f"[save] controls → {OUT_FP}  rows={len(monthly):,}  CCNs={monthly['cms_certification_number'].nunique():,}")
print(monthly.head(10).to_string(index=False))