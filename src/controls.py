#!/usr/bin/env python
# coding: utf-8
# -----------------------------------------------------------------------------
# CMS Provider Info — TEST HARVEST for case-mix, CCRC, SFF (by header variants)
#
# - Copies your provider-info pipeline (paths, zip walking, CCN cleaning)
# - Adds flexible extraction for:
#     • case_mix_total
#         2017–2018: "exp_total"
#         later: "CM_TOTAL", "CM TOTAL",
#                "Case-Mix Total Nurse Staffing Hours per Resident per Day",
#                "CASE-MIX TOTAL NURSE STAFFING HOURS PER RESIDENT PER DAY",
#                sometimes "CM_TOTAL_SRC"/"CM TOTAL SRC" appear; we record if present
#     • ccrc_facility (boolean-like)
#         "CCRC_FACIL", "Continuing Care Retirement Community",
#         "CONTINUING CARE RETIREMENT COMMUNITY"
#     • sff_flag (boolean-like)  — special focus status flag
#         "SFF", "Special Focus Status", "SPECIAL FOCUS STATUS"
#     • sff_status (string label)
#         "Special Focus Facility", "SPECIAL FOCUS FACILITY"
#
# - Writes monthly CSVs (provider_info_test_YYYY_MM.csv) and a combined CSV
#   with the new columns + the usual cleaned CCN, provider_name, etc.
# - Keeps audit columns recording which raw header each field came from.
# -----------------------------------------------------------------------------

import os, re, zipfile
from io import BytesIO
from pathlib import Path
import pandas as pd
import numpy as np

# ============================== Config / Paths ================================
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").is_dir() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

RAW_DIR     = Path(os.getenv("NH_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
NH_ZIP_DIR  = RAW_DIR / "nh-compare"
PROV_DIR    = RAW_DIR / "provider-info-files"
PROV_DIR.mkdir(parents=True, exist_ok=True)

TEST_COMBINED_CSV = PROV_DIR / "provider_info_testvars_combined.csv"

print(f"[paths] NH_ZIP_DIR={NH_ZIP_DIR}")
print(f"[paths] PROV_DIR  ={PROV_DIR}")

# ============================ File selection ==================================
PRIORITY = [
    "providerinfo_download.csv",
    "providerinfo_display.csv",
    "nh_providerinfo",
]

MONTH_RE = r"(0[1-9]|1[0-2])"; YEAR_RE = r"(20\d{2})"
INNER_PATTERNS = [
    re.compile(rf"nh_archive_{MONTH_RE}_{YEAR_RE}\.zip", re.I),
    re.compile(rf"nh_archive_{YEAR_RE}_{MONTH_RE}\.zip", re.I),
    re.compile(rf"nursing_homes_including_rehab_services_archive_{MONTH_RE}_{YEAR_RE}\.zip", re.I),
    re.compile(rf"(?:^|[_-]){MONTH_RE}[_-]{YEAR_RE}\.zip$", re.I),
    re.compile(rf"(?:^|[_-]){YEAR_RE}[_-]{MONTH_RE}\.zip$", re.I),
]

def parse_mm_yyyy_from_inner(name: str):
    for pat in INNER_PATTERNS:
        m = pat.search(name)
        if m:
            nums = [int(x) for x in m.groups() if x and x.isdigit()]
            if len(nums) >= 2:
                a, b = nums[0], nums[1]
                if a <= 12 and b >= 2000: return a, b
                if b <= 12 and a >= 2000: return b, a
    return (None, None)

# ============================ IO & helpers ====================================
def safe_read_csv(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8","utf-8-sig","cp1252","latin-1"):
        try:
            return pd.read_csv(BytesIO(raw), dtype=str, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(BytesIO(raw), dtype=str, encoding="utf-8", encoding_errors="replace", low_memory=False)

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = (pd.Index(df.columns).str.strip().str.lower()
            .str.replace(r"\s+","_", regex=True)
            .str.replace(r"[^0-9a-z_]", "", regex=True))
    df.columns = cols
    return df

def to_boolish(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    return s.map({
        "1": True, "y": True, "yes": True, "true": True, "t": True,
        "0": False,"n": False,"no": False,"false": False,"f": False
    }).astype("boolean")

# ============================ CCN candidate columns ===========================
PRIMARY_CCN_ORDER = [
    "cms_certification_number",
    "cms_certification_number_ccn",
    "federal_provider_number",
    "provnum",
    "provider_id",
    "provider_number",
]
NAME_CANDIDATES = ["provider_name", "provname"]
HOSP_CANDIDATES = [
    "provider_resides_in_hospital",
    "resides_in_hospital",
    "provider_resides_in_hospital_",
    "inhosp"
]

# ============================ New variable candidates =========================
# NOTE: these are the normalized header forms (after norm_cols).
CASE_MIX_CANDS = [
    "exp_total",
    "cm_total",
    "cm_total_src",                    # record if present (source indicator)
    "cm_total_src",                    # duplicate safe; harmless
    "cm_total_src",                    # keep to emphasize we record SRC if present
    "cm_total",                        # again to favor cm_total over long label
    "case_mix_total_nurse_staffing_hours_per_resident_per_day",
]
# We’ll search for SRC separately so it doesn't overwrite the actual measure.
CASE_MIX_SRC_CANDS = [
    "cm_total_src",
    "cm_total_src",        # variants normalize the same; left for clarity
    "cm_total_src"
]

CCRC_CANDS = [
    "ccrc_facil",
    "continuing_care_retirement_community"
]

SFF_FLAG_CANDS = [
    "sff",
    "special_focus_status"
]

SFF_STATUS_CANDS = [
    "special_focus_facility"
]

# ============================ CCN cleaning logic ==============================
ALNUM_6_7 = re.compile(r"^[0-9A-Z]{6,7}$")

def clean_primary_ccn(val: str) -> str | float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return np.nan
    s = str(val).strip().upper()
    if "+" in s or "." in s:
        return np.nan
    if s.isdigit():
        return s.zfill(6)
    if ALNUM_6_7.fullmatch(s):
        return s
    return np.nan

# ============================ Value pickers ===================================
def pick_first_nonnull(df: pd.DataFrame, candidates: list[str]) -> tuple[pd.Series, str | None]:
    """
    Returns a (series, raw_colname_used) where series is the first non-null among candidates.
    raw_colname_used is the normalized header name that supplied the *first non-null* per row.
    """
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    used = pd.Series(pd.NA, index=df.index, dtype="object")

    for c in candidates:
        if c not in df.columns: 
            continue
        fill = out.isna() & df[c].notna()
        out = out.mask(fill, df[c])
        used = used.mask(fill, c)

    # if nothing found anywhere, used will stay NA
    # compress to a single colname if same for all non-null, but we keep row-level used for audit
    return out, used

def pick_bool(df: pd.DataFrame, candidates: list[str]) -> tuple[pd.Series, pd.Series]:
    raw, used = pick_first_nonnull(df, candidates)
    # Convert to boolean where possible, keep <NA> where unmapped
    mapped = to_boolish(raw)
    return mapped, used

# ============================ Standardize one month ===========================
def standardize_provider_info_test(df: pd.DataFrame, yyyy: int, mm: int, source_name: str) -> pd.DataFrame:
    df = norm_cols(df)

    # Provider name
    pname = pd.Series(pd.NA, index=df.index, dtype="object")
    for cand in NAME_CANDIDATES:
        if cand in df.columns:
            fill = pname.isna() & df[cand].notna()
            pname = pname.mask(fill, df[cand])

    # Hospital flag (kept for convenience context)
    hosp = pd.Series(pd.NA, index=df.index, dtype="object")
    for cand in HOSP_CANDIDATES:
        if cand in df.columns:
            mapped = to_boolish(df[cand])
            hosp = hosp.mask(hosp.isna() & mapped.notna(), mapped)

    # CCN primary (pre-clean)
    present_cands = [c for c in PRIMARY_CCN_ORDER if c in df.columns]
    primary = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in present_cands:
        fill = primary.isna() & df[c].notna()
        primary = primary.mask(fill, df[c])
    cleaned = primary.map(clean_primary_ccn)

    # ---- New fields ---------------------------------------------------------
    # Case-mix total hours per resident per day (string -> we can leave as str; downstream can numeric-coerce)
    cm_series, cm_used = pick_first_nonnull(df, CASE_MIX_CANDS)
    cm_src_series, cm_src_used = pick_first_nonnull(df, CASE_MIX_SRC_CANDS)

    # CCRC and SFF flag (boolean-like)
    ccrc_series, ccrc_used = pick_bool(df, CCRC_CANDS)
    sff_flag_series, sff_flag_used = pick_bool(df, SFF_FLAG_CANDS)

    # SFF status (string label, not coerced)
    sff_status_series, sff_status_used = pick_first_nonnull(df, SFF_STATUS_CANDS)

    # ---- Build output frame -------------------------------------------------
    df_out = pd.DataFrame({
        "cms_certification_number": cleaned,
        "provider_name": pname,
        "provider_resides_in_hospital": hosp,
        "year": int(yyyy),
        "month": int(mm),
        "date": pd.Timestamp(year=int(yyyy), month=int(mm), day=1).strftime("%Y-%m-%d"),
        "source_file": Path(source_name).name,

        # New vars
        "case_mix_total": cm_series,                        # as found (string)
        "case_mix_total_src": cm_src_series,                # if present
        "ccrc_facility": ccrc_series,                       # boolean/NA
        "sff_flag": sff_flag_series,                        # boolean/NA
        "sff_status": sff_status_series,                    # string/NA

        # Audit: which header supplied each field (normalized names)
        "raw_case_mix_total_col": cm_used,
        "raw_case_mix_total_src_col": cm_src_used,
        "raw_ccrc_col": ccrc_used,
        "raw_sff_flag_col": sff_flag_used,
        "raw_sff_status_col": sff_status_used,
    })

    # Attach raw CCN columns (for audit)
    for c in present_cands:
        df_out[f"ccn_raw_{c}"] = df[c]

    # Drop invalid CCNs
    before = len(df_out)
    df_out = df_out.dropna(subset=["cms_certification_number"])
    if len(df_out) != before:
        print(f"  [clean-ccn] dropped {before - len(df_out):,} row(s) with invalid CCN in {source_name}")

    # Dedup within month
    before = len(df_out)
    df_out = df_out.drop_duplicates()
    if len(df_out) != before:
        print(f"  [dedup-month] removed {before - len(df_out):,} exact dupe row(s) in {source_name}")

    # Order & sort
    audit_cols = [
        "raw_case_mix_total_col","raw_case_mix_total_src_col",
        "raw_ccrc_col","raw_sff_flag_col","raw_sff_status_col"
    ]
    out_cols = (
        ["cms_certification_number"] +
        [c for c in df_out.columns if c.startswith("ccn_raw_")] +
        ["provider_name","provider_resides_in_hospital","year","month","date","source_file"] +
        ["case_mix_total","case_mix_total_src","ccrc_facility","sff_flag","sff_status"] +
        audit_cols
    )
    df_out = df_out[out_cols].sort_values(["cms_certification_number","date"], kind="mergesort").reset_index(drop=True)
    return df_out

# ============================ Extract → Standardize → Write ===================
def extract_and_standardize_test():
    yearly = sorted(p for p in NH_ZIP_DIR.glob("nh_archive_*.zip") if p.is_file())
    if not yearly:
        raise FileNotFoundError(f"No yearly zips found in {NH_ZIP_DIR}")

    written = 0
    for yzip in yearly:
        with zipfile.ZipFile(yzip, "r") as yz:
            inner_zips = [n for n in yz.namelist() if n.lower().endswith(".zip")]
            for inner in inner_zips:
                mm, yyyy = parse_mm_yyyy_from_inner(Path(inner).name)
                if not (mm and yyyy):
                    continue
                with yz.open(inner) as inner_bytes:
                    try:
                        with zipfile.ZipFile(BytesIO(inner_bytes.read()), "r") as mz:
                            entries = mz.namelist()
                            chosen = None
                            for pat in PRIORITY:
                                for e in entries:
                                    if pat in Path(e).name.lower() and Path(e).suffix.lower() == ".csv":
                                        chosen = e
                                        break
                                if chosen:
                                    break
                            if not chosen:
                                continue

                            raw = mz.read(chosen)
                            df = safe_read_csv(raw)
                            std = standardize_provider_info_test(
                                df, yyyy, mm,
                                f"{yzip.name}!{Path(inner).name}!{Path(chosen).name}"
                            )

                            out_name = f"provider_info_test_{yyyy:04d}_{mm:02d}.csv"
                            out_path = PROV_DIR / out_name
                            std.to_csv(out_path, index=False)
                            print(f"[save] {out_name:>26}  rows={len(std):,}")
                            written += 1
                    except zipfile.BadZipFile:
                        continue
    print(f"\n[extract+standardize:test] wrote {written} monthly test CSV(s).")

# ============================ Combine (testvars) ==============================
def combine_testvars():
    monthly = sorted(PROV_DIR.glob("provider_info_test_*.csv"))
    if not monthly:
        raise FileNotFoundError(f"No provider_info_test_*.csv files found in {PROV_DIR}")

    frames = []
    for p in monthly:
        try:
            df = pd.read_csv(p, dtype=str, low_memory=False)
            df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
            frames.append(df)
        except Exception as e:
            print(f"[warn] failed reading {p.name}: {e}")

    prov = pd.concat(frames, ignore_index=True)

    # De-dupe across months (rare but keep consistent with your style)
    before = len(prov)
    prov = prov.drop_duplicates()
    if len(prov) != before:
        print(f"[dedup-combined:test] removed {before - len(prov):,} exact dupe row(s)")

    prov = prov.sort_values(["cms_certification_number","date","source_file"], kind="mergesort").reset_index(drop=True)
    prov.to_csv(TEST_COMBINED_CSV, index=False)
    print(f"[save] combined (testvars) → {TEST_COMBINED_CSV}  ({len(prov):,} rows)")

# =============================== RUN ==========================================
if __name__ == "__main__":
    extract_and_standardize_test()
    combine_testvars()