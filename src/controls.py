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
#   * Output: data/interim/pbj_controls_from_mcr_monthly.csv
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------- Paths ----------------
RAW_DIR = Path(r"C:\Users\wrthj\OneDrive\NursingHomeData")
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
    # Tidy floats like '4.0' → '4'
    if s.endswith(".0"):
        s = s[:-2]
    if s in OWN_FOR_PROFIT:
        return "For-profit"
    if s in OWN_NONPROFIT:
        return "Nonprofit"
    if s in OWN_GOVT:
        return "Government"
    # Unknown code → NA
    return pd.NA

# Columns we need (allow case variants between SAS and CSV)
NEEDED = {
    "PRVDR_NUM",                 # CCN
    "FY_BGN_DT", "FY_END_DT",    # report period
    "MRC_OWNERSHIP", "MRC_ownership", "MRC_ownership_code",  # ownership code (SAS often 'MRC_ownership')
    "S3_1_PATDAYS_TOTAL",
    "S3_1_PATDAYS_MEDICARE",
    "S3_1_PATDAYS_MEDICAID",
}

def pick(colnames, *candidates):
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def load_single_year(year: int) -> pd.DataFrame:
    """Load one year, prefer SAS, fallback to CSV. Return subset with canonical columns."""
    sas_fp = MCR_DIR / f"mcr_flatfile_{year}.sas7bdat"
    csv_fp = MCR_DIR / f"mcr_flatfile_{year}.csv"

    df = None
    src = None

    if sas_fp.exists():
        try:
            import pyreadstat
            # Read full (columns vary across years); subset after read for simplicity/robustness
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
            "MRC_OWNERSHIP_RAW","S3_1_PATDAYS_TOTAL","S3_1_PATDAYS_MEDICARE","S3_1_PATDAYS_MEDICAID"
        ])

    cols = list(df.columns)
    prv_col = pick(cols, "PRVDR_NUM","prvdr_num","Provider Number")
    bgn_col = pick(cols, "FY_BGN_DT","fy_bgn_dt","Cost Report Fiscal Year beginning date")
    end_col = pick(cols, "FY_END_DT","fy_end_dt","Cost Report Fiscal Year ending date")
    own_col = pick(cols, "MRC_OWNERSHIP","MRC_ownership","mrc_ownership","MRC_ownership_code")
    tot_col = pick(cols, "S3_1_PATDAYS_TOTAL")
    mcr_col = pick(cols, "S3_1_PATDAYS_MEDICARE")
    mcd_col = pick(cols, "S3_1_PATDAYS_MEDICAID")

    keep_map = {
        "PRVDR_NUM": prv_col,
        "FY_BGN_DT": bgn_col,
        "FY_END_DT": end_col,
        "MRC_OWNERSHIP_RAW": own_col,
        "S3_1_PATDAYS_TOTAL": tot_col,
        "S3_1_PATDAYS_MEDICARE": mcr_col,
        "S3_1_PATDAYS_MEDICAID": mcd_col,
    }
    # Ensure presence
    for k, v in keep_map.items():
        if v is None:
            keep_map[k] = k  # keep missing placeholder; will fill with NaN
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

for c in ["S3_1_PATDAYS_TOTAL","S3_1_PATDAYS_MEDICARE","S3_1_PATDAYS_MEDICAID"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")

# Ownership collapse (using SAS codes when available)
raw["ownership_type"] = raw["MRC_OWNERSHIP_RAW"].apply(collapse_ownership_code)

# Quick sanity on ownership codes
code_counts = (
    raw["MRC_OWNERSHIP_RAW"].astype("string").str.strip().replace({"": pd.NA}).value_counts(dropna=False).head(20)
)
print("\n=== Raw ownership code counts (top 20) ===")
print(code_counts)

# ---------------- Expand to monthly ----------------
rows = []
need = raw.dropna(subset=["cms_certification_number","FY_BGN_DT","FY_END_DT"])
for _, r in need.iterrows():
    months = month_range_df(r["FY_BGN_DT"], r["FY_END_DT"])
    if months.empty:
        continue
    tot = r["S3_1_PATDAYS_TOTAL"]
    medcr = r["S3_1_PATDAYS_MEDICARE"]
    medcd = r["S3_1_PATDAYS_MEDICAID"]
    if pd.notna(tot) and tot > 0:
        pct_medicare = 100.0 * (0 if pd.isna(medcr) else medcr) / tot
        pct_medicaid = 100.0 * (0 if pd.isna(medcd) else medcd) / tot
    else:
        pct_medicare = np.nan
        pct_medicaid = np.nan

    block = months.copy()
    block["cms_certification_number"] = r["cms_certification_number"]
    block["ownership_type"] = r["ownership_type"]
    block["pct_medicare"] = pct_medicare
    block["pct_medicaid"] = pct_medicaid
    rows.append(block)

monthly = (pd.concat(rows, ignore_index=True)
           if rows else pd.DataFrame(columns=["cms_certification_number","month_start","ownership_type","pct_medicare","pct_medicaid"]))

# Deduplicate overlapping months within CCN (prefer a non-null ownership; average pay mix if duplicates)
monthly = monthly.sort_values(["cms_certification_number","month_start"])
monthly = (monthly
           .groupby(["cms_certification_number","month_start"], as_index=False)
           .agg({
               "ownership_type": lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
               "pct_medicare": "mean",
               "pct_medicaid": "mean",
           }))

# ---------------- Report & save ----------------
print(f"\n[result] monthly control rows={len(monthly):,}  CCNs={monthly['cms_certification_number'].nunique():,}")

print("\n=== ownership_type distribution ===")
print(monthly["ownership_type"].value_counts(dropna=False))

has_any_pay = ((monthly["pct_medicare"].notna()) | (monthly["pct_medicaid"].notna())).mean()
print("\n% rows with at least one payer % populated:", round(100*has_any_pay, 1), "%")

monthly.to_csv(OUT_FP, index=False)
print(f"[saved] {OUT_FP}")