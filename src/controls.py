from pathlib import Path
import pandas as pd
import numpy as np

# --- Paths & files ---
RAW_DIR = Path(r"C:\Users\Owner\OneDrive\NursingHomeData")
MCR_DIR = RAW_DIR / "medicare-cost-reports"
FILES = sorted(MCR_DIR.glob("mcr_flatfile_20??.csv"))

print(f"[paths] RAW_DIR={RAW_DIR}")
print(f"[paths] MCR_DIR={MCR_DIR}")
print(f"[scan] {len(FILES)} files:", [f.name for f in FILES])

# --- Helpers ---
def normalize_ccn_any(s: pd.Series) -> pd.Series:
    s = s.astype("string").fillna("").str.strip().str.upper()
    s = s.str.replace(r"[ \-\/\.]", "", regex=True)
    is_digits = s.str.fullmatch(r"\d+")
    return s.mask(is_digits, s.str.zfill(6)).replace({"": pd.NA})

def month_range_df(start, end):
    """Closed-open monthly range [start, end] -> rows of month_start."""
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame({"month_start":[]})
    s = pd.Period(start, "M").to_timestamp()
    e = pd.Period(end,   "M").to_timestamp()
    # Ensure end >= start; guard for bad rows
    if e < s: 
        s, e = e, s
    months = pd.period_range(s, e, freq="M").to_timestamp()
    return pd.DataFrame({"month_start": months})

# --- Load, keep only needed columns, and stack ---
use_cols = [
    "PRVDR_NUM",        # CCN
    "FY_BGN_DT",        # fiscal begin date
    "FY_END_DT",        # fiscal end date
    "MRC_OWNERSHIP",    # ownership (numeric-coded)
    "S3_1_PATDAYS_TOTAL",
    "S3_1_PATDAYS_MEDICARE",
    "S3_1_PATDAYS_MEDICAID",
]
frames = []
for fp in FILES:
    df = pd.read_csv(fp, low_memory=False)
    df.columns = [c.upper().strip() for c in df.columns]
    keep = [c for c in use_cols if c in df.columns]
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        print(f"[warn] {fp.name} missing: {missing}")
    sub = df[keep].copy()
    sub["file_year"] = int(fp.stem[-4:])
    frames.append(sub)

raw = pd.concat(frames, ignore_index=True)
print(f"[read] combined rows={len(raw):,}, cols={raw.shape[1]}")

# --- Basic cleaning ---
raw["cms_certification_number"] = normalize_ccn_any(raw["PRVDR_NUM"])
raw["FY_BGN_DT"] = pd.to_datetime(raw["FY_BGN_DT"], errors="coerce")
raw["FY_END_DT"] = pd.to_datetime(raw["FY_END_DT"], errors="coerce")

# Patient-day based pay mix (preferred)
for c in ["S3_1_PATDAYS_TOTAL","S3_1_PATDAYS_MEDICARE","S3_1_PATDAYS_MEDICAID"]:
    if c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    else:
        raw[c] = np.nan

# --- Build a codebook of ownership codes found ---
codebook = (
    raw["MRC_OWNERSHIP"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.replace(".0$", "", regex=True)   # tidy "4.0" -> "4"
    .value_counts()
    .rename_axis("MRC_OWNERSHIP_code")
    .reset_index(name="count")
)
print("\n=== Ownership codebook (from MRC_OWNERSHIP) ===")
print(codebook.to_string(index=False))

# --- Define your mapping to 3 buckets ---
# Fill this dict once you decide the mapping.
# Keep it conservative first, then iterate based on the codebook printout.
OWN_MAP = {
    # EXAMPLES (adjust/expand these as you confirm meanings):
    # "1": "Nonprofit",
    # "2": "Nonprofit",
    # "3": "Government",
    # "4": "For-profit",
    # "5": "For-profit",
    # "6": "For-profit",
    # "7": "For-profit",
    # "8": "Government",
    # "9": "For-profit",
    # "10": "Government",
    # "11": "Government",
    # "12": "Government",
    # "13": "Government",
    # "0": None,  # unknown/missing
}

# Standardize codes to strings w/o trailing .0
raw["_own_code"] = (
    raw["MRC_OWNERSHIP"].astype(str).str.strip().str.replace(".0$", "", regex=True)
)

raw["ownership_type"] = raw["_own_code"].map(OWN_MAP)

# Report unmapped so you can finish the dictionary above
unmapped = (
    raw.loc[raw["ownership_type"].isna(), "_own_code"]
       .dropna().value_counts()
       .rename_axis("code").reset_index(name="rows")
)
print("\n=== Unmapped ownership codes (fill OWN_MAP for these) ===")
print(unmapped.head(30).to_string(index=False) if not unmapped.empty else "None 🎉")

# --- Expand each cost-report year to monthly rows and attach controls ---
rows = []
for _, r in raw.dropna(subset=["cms_certification_number","FY_BGN_DT","FY_END_DT"]).iterrows():
    months = month_range_df(r["FY_BGN_DT"], r["FY_END_DT"])
    if months.empty:
        continue
    tot = r.get("S3_1_PATDAYS_TOTAL", np.nan)
    medcr = r.get("S3_1_PATDAYS_MEDICARE", np.nan)
    medcd = r.get("S3_1_PATDAYS_MEDICAID", np.nan)
    # compute shares (stay on the row; monthly expansion just repeats annual share)
    if pd.notna(tot) and tot > 0:
        pct_medicare = 100.0 * (medcr or 0) / tot
        pct_medicaid = 100.0 * (medcd or 0) / tot
    else:
        pct_medicare = np.nan
        pct_medicaid = np.nan

    block = months.copy()
    block["cms_certification_number"] = r["cms_certification_number"]
    block["ownership_type"] = r["ownership_type"]
    block["pct_medicare"] = pct_medicare
    block["pct_medicaid"] = pct_medicaid
    rows.append(block)

monthly = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
    columns=["cms_certification_number","month_start","ownership_type","pct_medicare","pct_medicaid"]
)

# De-duplicate if overlapping reports yield duplicate months; prefer non-null ownership
monthly = monthly.sort_values(["cms_certification_number","month_start"])
monthly = (monthly
           .groupby(["cms_certification_number","month_start"], as_index=False)
           .agg({
               "ownership_type": lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
               "pct_medicare": "mean",
               "pct_medicaid": "mean",
           }))

print(f"\n[result] monthly rows={len(monthly):,}  CCNs={monthly['cms_certification_number'].nunique():,}")
print("\n=== ownership_type distribution (after mapping) ===")
print(monthly["ownership_type"].value_counts(dropna=False).head(10))

print("\n% rows with at least one payer% populated:",
      round(100 * ((monthly["pct_medicare"].notna()) | (monthly["pct_medicaid"].notna())).mean(), 1), "%")