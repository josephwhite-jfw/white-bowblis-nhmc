#!/usr/bin/env python
# coding: utf-8
# ─────────────────────────────────────────────────────────────────────────────
# Monthly payer mix (% Medicare / % Medicaid) + ownership type from MCR
# Uses explicit columns (validated on your files):
#   CCN: PRVDR_NUM
#   Period: FY_BGN_DT, FY_END_DT
#   Ownership: MRC_OWNERSHIP (1=For-profit, 2=Nonprofit, 3=Government)
#   Patient days: S3_1_PATDAYS_TOTAL / _MEDICARE / _MEDICAID
#
# Output: data/interim/pbj_controls_from_mcr_monthly.csv
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import os, warnings, re
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- Paths ----------------
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").is_dir() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

RAW_DIR = Path(os.getenv("NH_DATA_DIR", PROJECT_ROOT / "data" / "raw")).resolve()
MCR_DIR = RAW_DIR / "medicare-cost-reports"
GLOB = "mcr_flatfile_20??.csv"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

OUT_FP = INTERIM_DIR / "pbj_controls_from_mcr_monthly.csv"

print(f"[paths] RAW_DIR={RAW_DIR}")
print(f"[paths] MCR_DIR={MCR_DIR}")
print(f"[out]   {OUT_FP}")

# ---------------- Helpers ----------------
def normalize_ccn_any(series: pd.Series) -> pd.Series:
    """Alphanumeric-safe CCN normalization: strip separators, pad only if all digits."""
    s = series.astype("string").fillna("").str.strip().str.upper()
    s = s.str.replace(r"[ \-\/\.]", "", regex=True)
    is_digits = s.str.fullmatch(r"\d+")
    s = s.mask(is_digits, s.str.zfill(6))
    s = s.replace({"": pd.NA})
    return s

OWN_MAP = {
    1: "For-profit",  # Proprietary
    2: "Nonprofit",
    3: "Government",
}

def map_ownership(val):
    if pd.isna(val): return pd.NA
    try:
        i = int(float(str(val)))
        return OWN_MAP.get(i, pd.NA)
    except Exception:
        s = str(val).strip().upper()
        if any(k in s for k in ["PROPRIETARY", "FOR PROFIT", "FOR-PROFIT", "PROFIT", "FP"]):
            return "For-profit"
        if any(k in s for k in ["NONPROFIT", "NOT-FOR-PROFIT", "NOT FOR PROFIT", "NON-PROFIT", "NFP"]):
            return "Nonprofit"
        if any(k in s for k in ["GOV", "GOVERNMENT", "PUBLIC", "STATE", "COUNTY", "CITY", "FEDERAL"]):
            return "Government"
        return pd.NA

def month_span(start_ts, end_ts):
    if pd.isna(start_ts) or pd.isna(end_ts):
        return []
    s = pd.Period(start_ts, "M").to_timestamp("s")
    e = pd.Period(end_ts,   "M").to_timestamp("s")
    return pd.period_range(s, e, freq="M").to_timestamp("s").tolist()

# ---------------- Main ----------------
files = sorted(MCR_DIR.glob(GLOB))
if not files:
    raise FileNotFoundError(f"No files matched {MCR_DIR / GLOB}")

rows = []
for fp in files:
    df = pd.read_csv(fp, low_memory=False)
    df.columns = [c.upper().strip() for c in df.columns]
    print(f"[read] {fp.name} rows={len(df):,} cols={df.shape[1]}")

    # Required columns (explicit)
    req = ["PRVDR_NUM", "FY_BGN_DT", "FY_END_DT", "MRC_OWNERSHIP",
           "S3_1_PATDAYS_TOTAL", "S3_1_PATDAYS_MEDICARE", "S3_1_PATDAYS_MEDICAID"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        print(f"[warn] {fp.name}: missing expected columns {missing} — attempting to continue where possible.")

    # CCN
    if "PRVDR_NUM" not in df.columns:
        print(f"[warn] {fp.name}: no PRVDR_NUM; skipping file.")
        continue
    df["cms_certification_number"] = normalize_ccn_any(df["PRVDR_NUM"])

    # Period
    beg = pd.to_datetime(df.get("FY_BGN_DT"), errors="coerce")
    end = pd.to_datetime(df.get("FY_END_DT"), errors="coerce")

    # Ownership
    own = df.get("MRC_OWNERSHIP")
    ownership_type = own.map(map_ownership) if own is not None else pd.Series(pd.NA, index=df.index)

    # Patient days → payer mix
    tot = pd.to_numeric(df.get("S3_1_PATDAYS_TOTAL"), errors="coerce")
    mdcr = pd.to_numeric(df.get("S3_1_PATDAYS_MEDICARE"), errors="coerce") if "S3_1_PATDAYS_MEDICARE" in df.columns else np.nan
    mdcd = pd.to_numeric(df.get("S3_1_PATDAYS_MEDICAID"), errors="coerce") if "S3_1_PATDAYS_MEDICAID" in df.columns else np.nan

    # Avoid division by zero; compute percents
    tot_nonzero = tot.replace(0, np.nan)
    pct_medicare = (mdcr / tot_nonzero * 100).round(2)
    pct_medicaid = (mdcd / tot_nonzero * 100).round(2)

    # Assemble per-row summary
    keep = pd.DataFrame({
        "cms_certification_number": df["cms_certification_number"],
        "period_begin": beg,
        "period_end": end,
        "ownership_type": ownership_type,
        "pct_medicare": pct_medicare,
        "pct_medicaid": pct_medicaid,
        "src_file": fp.name
    }).dropna(subset=["cms_certification_number", "period_begin", "period_end"])

    # Expand to monthly
    for _, r in keep.iterrows():
        months = month_span(r["period_begin"], r["period_end"])
        for m in months:
            rows.append({
                "cms_certification_number": r["cms_certification_number"],
                "year_month": pd.Period(m, "M").strftime("%Y-%m"),
                "month": pd.Period(m, "M").to_timestamp("s"),
                "pct_medicare": r["pct_medicare"],
                "pct_medicaid": r["pct_medicaid"],
                "ownership_type": r["ownership_type"],
                "src_file": r["src_file"]
            })

# Finalize
out = pd.DataFrame(rows)
if out.empty:
    print("[result] No rows created. Check that the expected columns exist in the flatfiles.")
else:
    out = (out.sort_values(["cms_certification_number","year_month","src_file"])
             .drop_duplicates(["cms_certification_number","year_month"], keep="last")
             .reset_index(drop=True))
    print(f"[result] monthly control rows={len(out):,}  CCNs={out['cms_certification_number'].nunique():,}")

    print("\n=== ownership_type distribution ===")
    print(out["ownership_type"].value_counts(dropna=False))

    ok_basis = out[["pct_medicare","pct_medicaid"]].notna().any(axis=1).mean()
    print(f"\n% rows with at least one payer % populated: {ok_basis:.1%}")

    out.to_csv(OUT_FP, index=False)
    print(f"[saved] {OUT_FP}")
