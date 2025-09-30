#!/usr/bin/env python
# coding: utf-8
# ─────────────────────────────────────────────────────────────────────────────
# PBJ Controls from MCR (monthly, SAS-first; CSV fallback)
#   * Scans RAW_DIR/medicare-cost-reports for mcr_flatfile_20??.(sas7bdat|csv)
#   * Prefers SAS for accurate codes; falls back to CSV if SAS missing
#   * Keeps alphanumeric CCNs (pad only if purely numeric)
#   * Expands each cost-report period to monthly rows
#   * Controls:
#       - ownership_type (For-profit / Nonprofit / Government) from MRC_ownership code
#       - pct_medicare, pct_medicaid (patient-day shares)
#       - num_beds (avg), occupancy_rate (%)
#       - state (2-letter), urban_rural (Urban/Rural)
#       - is_chain (1 if has Home Office / chain indicator, else 0)
#   * Output: data/interim/pbj_controls_from_mcr_monthly.csv
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------- Paths ----------------
RAW_DIR = Path(r"C:\Users\Owner\OneDrive\NursingHomeData")
MCR_DIR = RAW_DIR / "medicare-cost-reports"

REPO = Path.cwd()
while not (REPO / "data").is_dir() and REPO != REPO.parent:
    REPO = REPO.parent
INTERIM = REPO / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

OUT_FP = INTERIM / "pbj_controls_from_mcr_monthly.csv"

print(f"[paths] RAW_DIR={RAW_DIR}")
print(f"[paths] MCR_DIR={MCR_DIR}")
print(f"[out]   {OUT_FP}")

# ---------------- Helpers ----------------
def normalize_ccn_any(s: pd.Series) -> pd.Series:
    """Alphanumeric-safe CCN normalization: strip separators, pad only if all digits."""
    s = s.astype("string").fillna("").str.strip().str.upper()
    s = s.str.replace(r"[ \-\/\.]", "", regex=True)
    is_digits = s.str.fullmatch(r"\d+")
    return s.mask(is_digits, s.str.zfill(6)).replace({"": pd.NA})

def month_range_df(start, end) -> pd.DataFrame:
    """Closed monthly span from start..end (month-start Timestamps)."""
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame({"month_start":[]})
    s = pd.Period(start, "M").to_timestamp()
    e = pd.Period(end,   "M").to_timestamp()
    if e < s:
        s, e = e, s
    months = pd.period_range(s, e, freq="M").to_timestamp()  # month-starts
    return pd.DataFrame({"month_start": months})

# Collapse ownership codes → 3 buckets, based on MCR/HCRIS documentation:
# For-profit:    {5,6,7,9}
# Nonprofit:     {1,2}
# Government:    {3,4,8,10,11,12,13}
OWN_FOR_PROFIT = {"5","6","7","9"}
OWN_NONPROFIT  = {"1","2"}
OWN_GOVT       = {"3","4","8","10","11","12","13"}

def collapse_ownership_code(code: str):
    """Map raw ownership code string (e.g., '4', '11', or '4.0') to 3 buckets or <NA>."""
    if code is None:
        return pd.NA
    s = str(code).strip()
    if s == "" or s.lower() in {"nan","na","none"}:
        return pd.NA
    if s.endswith(".0"):
        s = s[:-2]
    if s in OWN_FOR_PROFIT:
        return "For-profit"
    if s in OWN_NONPROFIT:
        return "Nonprofit"
    if s in OWN_GOVT:
        return "Government"
    return pd.NA

def pick(colnames, *candidates):
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand and cand.lower() in lower:
            return lower[cand.lower()]
    return None

def coerce_state(s):
    if s is None:
        return pd.NA
    x = str(s).strip().upper()
    if len(x) == 2 and x.isalpha():
        return x
    return pd.NA

def coerce_urban_rural(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NA
    s = str(x).strip().upper()
    if s in {"U","URBAN","1","Y","YES"}:
        return "Urban"
    if s in {"R","RURAL","0","N","NO"}:
        return "Rural"
    # Sometimes codes are 1/2; treat 2 as Rural as a fallback if seen
    if s == "2":
        return "Rural"
    return pd.NA

# --- Chain (Home Office) helpers --------------------------------------------
HOME_OFFICE_CANDIDATES = [
    "HOME_OFFICE", "HOME_OFFICE_IND", "HOME_OFFICE_FLAG", "MCR_HOME_OFFICE",
    "S2_1_HOME_OFFICE", "S2_2_HOME_OFFICE", "HO_IND", "HOME_OFFICE_INDICATOR"
]

def to_chain_flag(val):
    """
    Convert a raw 'home office' field to {0,1}:
      * numeric -> 1 if != 0
      * string  -> 1 if non-empty and not in {0, '0', 'N', 'NO', 'NONE', 'FALSE', 'F'}
      * otherwise -> 0
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0
    try:
        f = float(str(val).strip())
        return int(f != 0.0)
    except Exception:
        s = str(val).strip().upper()
        if s in {"", "0", "N", "NO", "NONE", "FALSE", "F"}:
            return 0
        return 1

# Columns we try to read (allow case variants & flexible names)
NEEDED_BASE = {
    "PRVDR_NUM",                 # CCN
    "FY_BGN_DT", "FY_END_DT",    # report period
    "MRC_OWNERSHIP", "MRC_ownership", "MRC_ownership_code",  # ownership code
    "S3_1_PATDAYS_TOTAL",
    "S3_1_PATDAYS_MEDICARE",
    "S3_1_PATDAYS_MEDICAID",
}
# Extra for controls:
EXTRA_CANDS = {
    "state": ["MCR_STATE","STATE","PROV_STATE","STATE_CD"],
    "urban": ["MCR_URBAN","URBAN_RURAL","URBAN_IND","URBAN","URBRUR"],
    "bed_days": ["BED_DAYS","S3_1_BED_DAYS","TOT_BED_DAYS","G3_BED_DAYS"],
    "avg_beds": ["AVG_BEDS","AVERAGE_BEDS","AVG_INPT_BEDS","S3_1_AVG_BEDS"],
    "inpt_beds": ["INPATIENT_BEDS","TOT_CERT_BEDS","S3_1_TOT_BEDS","TOTAL_BEDS"],
}

def load_single_year(year: int) -> pd.DataFrame:
    """Load one year, prefer SAS, fallback to CSV. Return subset with canonical columns."""
    sas_fp = MCR_DIR / f"mcr_flatfile_{year}.sas7bdat"
    csv_fp = MCR_DIR / f"mcr_flatfile_{year}.csv"

    df = None
    src = None

    if sas_fp.exists():
        try:
            import pyreadstat
            df, meta = pyreadstat.read_sas7bdat(str(sas_fp))
            src = "sas"
            print(f"[read] {sas_fp.name} (SAS) rows={len(df):,} cols={df.shape[1]}")
        except Exception as e:
            print(f"[warn] {sas_fp.name} failed: {e}")

    if df is None and csv_fp.exists():
        df = pd.read_csv(csv_fp, low_memory=False)
        src = "csv"
        print(f"[read] {csv_fp.name} (CSV) rows={len(df):,} cols={df.shape[1]}")

    if df is None:
        print(f"[warn] {year}: no SAS or CSV file found; skipping")
        return pd.DataFrame(columns=[
            "cms_certification_number","FY_BGN_DT","FY_END_DT",
            "MRC_OWNERSHIP_RAW","S3_1_PATDAYS_TOTAL","S3_1_PATDAYS_MEDICARE","S3_1_PATDAYS_MEDICAID",
            "STATE_RAW","URBAN_RAW","BED_DAYS_RAW","AVG_BEDS_RAW","INPT_BEDS_RAW","HOME_OFFICE_RAW","_src","file_year"
        ])

    cols = list(df.columns)
    # Required-ish
    prv_col = pick(cols, "PRVDR_NUM","prvdr_num","Provider Number")
    bgn_col = pick(cols, "FY_BGN_DT","fy_bgn_dt","Cost Report Fiscal Year beginning date")
    end_col = pick(cols, "FY_END_DT","fy_end_dt","Cost Report Fiscal Year ending date")
    own_col = pick(cols, "MRC_OWNERSHIP","MRC_ownership","mrc_ownership","MRC_ownership_code")
    tot_col = pick(cols, "S3_1_PATDAYS_TOTAL")
    mcr_col = pick(cols, "S3_1_PATDAYS_MEDICARE")
    mcd_col = pick(cols, "S3_1_PATDAYS_MEDICAID")

    # Optional extras
    st_col   = pick(cols, *EXTRA_CANDS["state"])
    urb_col  = pick(cols, *EXTRA_CANDS["urban"])
    bd_col   = pick(cols, *EXTRA_CANDS["bed_days"])
    ab_col   = pick(cols, *EXTRA_CANDS["avg_beds"])
    ip_col   = pick(cols, *EXTRA_CANDS["inpt_beds"])

    # Home office / chain
    ho_col = None
    for c in HOME_OFFICE_CANDIDATES:
        hit = pick(cols, c)
        if hit is not None:
            ho_col = hit
            break
    if ho_col is None:
        ho_like = [c for c in cols if ("HOME" in c.upper() and "OFFICE" in c.upper())]
        if len(ho_like) == 1:
            ho_col = ho_like[0]
        elif len(ho_like) > 1:
            nunique_sorted = sorted(
                [(c, df[c].dropna().nunique()) for c in ho_like],
                key=lambda x: x[1]
            )
            ho_col = nunique_sorted[0][0] if nunique_sorted else None

    keep_map = {
        "PRVDR_NUM": prv_col,
        "FY_BGN_DT": bgn_col,
        "FY_END_DT": end_col,
        "MRC_OWNERSHIP_RAW": own_col,
        "S3_1_PATDAYS_TOTAL": tot_col,
        "S3_1_PATDAYS_MEDICARE": mcr_col,
        "S3_1_PATDAYS_MEDICAID": mcd_col,
        "STATE_RAW": st_col,
        "URBAN_RAW": urb_col,
        "BED_DAYS_RAW": bd_col,
        "AVG_BEDS_RAW": ab_col,
        "INPT_BEDS_RAW": ip_col,
        "HOME_OFFICE_RAW": ho_col,
    }
    # Ensure presence
    for k, v in keep_map.items():
        if v is None:
            keep_map[k] = k
            df[k] = pd.NA

    sub = df[[keep_map[k] for k in keep_map]].copy()
    sub.columns = list(keep_map.keys())
    sub["file_year"] = year
    sub["_src"] = src
    return sub

# ---------------- Load all years ----------------
years = sorted({int(p.stem[-4:]) for p in MCR_DIR.glob("mcr_flatfile_20??.*")})
if not years:
    raise FileNotFoundError(f"No MCR files found in {MCR_DIR}")

parts = [load_single_year(y) for y in years]
raw = pd.concat(parts, ignore_index=True)
print(f"[stack] combined rows={len(raw):,} cols={raw.shape[1]}")

# ---------------- Clean & types ----------------
raw["cms_certification_number"] = normalize_ccn_any(raw["PRVDR_NUM"])
raw["FY_BGN_DT"] = pd.to_datetime(raw["FY_BGN_DT"], errors="coerce")
raw["FY_END_DT"] = pd.to_datetime(raw["FY_END_DT"], errors="coerce")

for c in ["S3_1_PATDAYS_TOTAL","S3_1_PATDAYS_MEDICARE","S3_1_PATDAYS_MEDICAID",
          "BED_DAYS_RAW","AVG_BEDS_RAW","INPT_BEDS_RAW"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")

raw["state"] = raw["STATE_RAW"].apply(coerce_state)
raw["urban_rural"] = raw["URBAN_RAW"].apply(coerce_urban_rural)

# Ownership collapse
raw["ownership_type"] = raw["MRC_OWNERSHIP_RAW"].apply(collapse_ownership_code)

# Chain dummy from home-office field
raw["is_chain"] = raw["HOME_OFFICE_RAW"].apply(to_chain_flag).astype("Int8")

# Quick ownership sanity
code_counts = (
    raw["MRC_OWNERSHIP_RAW"].astype("string").str.strip().replace({"": pd.NA}).value_counts(dropna=False).head(20)
)
print("\n=== Raw ownership code counts (top 20) ===")
print(code_counts)

# ---------------- Compute beds & occupancy on the annual row ----------------
period_days = (raw["FY_END_DT"] - raw["FY_BGN_DT"]).dt.days.add(1)
period_days = period_days.where(period_days > 0, np.nan)

# Number of beds (avg): prefer AVG_BEDS_RAW; else derive from bed-days / period_days
raw["num_beds"] = raw["AVG_BEDS_RAW"]
need_fill = raw["num_beds"].isna() & raw["BED_DAYS_RAW"].notna() & period_days.notna() & (period_days > 0)
raw.loc[need_fill, "num_beds"] = raw.loc[need_fill, "BED_DAYS_RAW"] / period_days.loc[need_fill]

# Occupancy rate (%) = total patient days / (num_beds * period_days) * 100
den = raw["num_beds"] * period_days
raw["occupancy_rate"] = np.where(
    den.notna() & (den > 0) & raw["S3_1_PATDAYS_TOTAL"].notna(),
    (raw["S3_1_PATDAYS_TOTAL"] / den) * 100.0,
    np.nan
)
raw["occupancy_rate"] = raw["occupancy_rate"].clip(lower=0, upper=100)

# Pay mix (patient-day shares)
def _share(numer, denom):
    return np.where(
        denom.notna() & (denom > 0) & numer.notna(),
        (numer / denom) * 100.0,
        np.nan
    )
raw["pct_medicare"] = _share(raw["S3_1_PATDAYS_MEDICARE"], raw["S3_1_PATDAYS_TOTAL"])
raw["pct_medicaid"] = _share(raw["S3_1_PATDAYS_MEDICAID"], raw["S3_1_PATDAYS_TOTAL"])

# ---------------- Expand to monthly ----------------
rows = []
need = raw.dropna(subset=["cms_certification_number","FY_BGN_DT","FY_END_DT"])
for _, r in need.iterrows():
    months = month_range_df(r["FY_BGN_DT"], r["FY_END_DT"])
    if months.empty:
        continue

    block = months.copy()
    block["cms_certification_number"] = r["cms_certification_number"]
    block["ownership_type"] = r["ownership_type"]
    block["pct_medicare"] = r["pct_medicare"]
    block["pct_medicaid"] = r["pct_medicaid"]
    block["num_beds"] = r["num_beds"]
    block["occupancy_rate"] = r["occupancy_rate"]
    block["state"] = r["state"]
    block["urban_rural"] = r["urban_rural"]
    block["is_chain"] = r["is_chain"]
    rows.append(block)

monthly = (pd.concat(rows, ignore_index=True)
           if rows else pd.DataFrame(columns=[
               "cms_certification_number","month_start","ownership_type","pct_medicare","pct_medicaid",
               "num_beds","occupancy_rate","state","urban_rural","is_chain"
           ]))

# Deduplicate overlapping months within CCN:
#  - ownership_type: prefer earliest non-null (mode-ish choice is possible, earliest is fine)
#  - pct_medicare/pct_medicaid/num_beds/occupancy_rate: mean across overlaps
#  - state: first non-null
#  - urban_rural: first non-null
#  - is_chain: max (if any report says chain, mark 1)
monthly = monthly.sort_values(["cms_certification_number","month_start"])
monthly = (monthly
           .groupby(["cms_certification_number","month_start"], as_index=False)
           .agg({
               "ownership_type": lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
               "pct_medicare": "mean",
               "pct_medicaid": "mean",
               "num_beds": "mean",
               "occupancy_rate": "mean",
               "state": lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
               "urban_rural": lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
               "is_chain": "max",
           }))
monthly["is_chain"] = monthly["is_chain"].fillna(0).astype("Int8")

# ---------------- Report & save ----------------
print(f"\n[result] monthly control rows={len(monthly):,}  CCNs={monthly['cms_certification_number'].nunique():,}")

print("\n=== ownership_type distribution ===")
print(monthly["ownership_type"].value_counts(dropna=False))

has_any_pay = ((monthly["pct_medicare"].notna()) | (monthly["pct_medicaid"].notna())).mean()
print("\n% rows with at least one payer % populated:", round(100*has_any_pay, 1), "%")

print("\n=== urban_rural distribution (non-null) ===")
print(monthly.loc[monthly["urban_rural"].notna(), "urban_rural"].value_counts())

print("\n=== state sample (top 10) ===")
print(monthly["state"].value_counts(dropna=True).head(10))

print("\n=== is_chain distribution ===")
print(monthly["is_chain"].value_counts().rename({0:"No (0)",1:"Yes (1)"}))

monthly.to_csv(OUT_FP, index=False)
print(f"[saved] {OUT_FP}")