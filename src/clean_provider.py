#!/usr/bin/env python
# coding: utf-8
# -----------------------------------------------------------------------------
# CMS Provider Info — Extract → Standardize (lean) → Combine (with 2Q lead)
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
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

NH_ZIP_DIR  = RAW_DIR / "nh-compare"
PROV_DIR    = RAW_DIR / "provider-info-files"
PROV_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_CSV = INTERIM_DIR / "provider.csv"

print(f"[paths] NH_ZIP_DIR={NH_ZIP_DIR}")
print(f"[paths] PROV_DIR  ={PROV_DIR}")
print(f"[paths] INTERIM   ={INTERIM_DIR}")

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
    dash_chars = r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212-]"
    cols = pd.Index(df.columns)
    cols = cols.str.replace("\u00A0", " ", regex=False)     # NBSP → space
    cols = cols.str.replace(dash_chars, " ", regex=True)    # any dash → space
    cols = cols.str.strip().str.lower()
    cols = cols.str.replace(r"\s+", "_", regex=True)
    cols = cols.str.replace(r"[^0-9a-z_]", "", regex=True)
    df.columns = cols
    return df

def to_boolish(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    return s.map({
        "1": True, "y": True, "yes": True, "true": True, "t": True,
        "0": False,"n": False,"no": False,"false": False,"f": False
    }).astype("boolean")

# ============================ CCN cleaning ====================================
ALNUM_6_7 = re.compile(r"^[0-9A-Z]{6,7}$")
def clean_primary_ccn(val: str) -> str | float:
    if val is None or (isinstance(val, float) and pd.isna(val)): return np.nan
    s = str(val).strip().upper()
    if "+" in s or "." in s:  # junk artifacts
        return np.nan
    if s.isdigit():
        return s.zfill(6)
    if ALNUM_6_7.fullmatch(s):
        return s
    return np.nan

# ============================ Column candidates ===============================
PRIMARY_CCN_ORDER = [
    "cms_certification_number",
    "cms_certification_number_ccn",
    "federal_provider_number",
    "provnum",
    "provider_id",
    "provider_number",
]

HOSP_CANDIDATES = [
    "provider_resides_in_hospital",
    "resides_in_hospital",
    "provider_resides_in_hospital_",
    "inhosp"
]

CASE_MIX_CANDS = [
    "exp_total",
    "cm_total",
    "case_mix_total_nurse_staffing_hours_per_resident_per_day",
    "casemix_total_nurse_staffing_hours_per_resident_per_day",
]

CCRC_CANDS = ["ccrc_facil","continuing_care_retirement_community"]

# SFF: derive facility status from text and facility Y/N (ignore sff_flag for output)
SFF_STATUS_TEXT_CANDS = ["special_focus_status"]     # e.g., "SFF", "SFF Candidate", "Former SFF"
SFF_FACILITY_CANDS    = ["special_focus_facility"]   # Y/N → current

def classify_sff_text(text: str | float) -> str | None:
    if text is None or (isinstance(text, float) and pd.isna(text)): return None
    t = str(text).strip()
    if t == "" or t.lower() == "nan": return None
    t = (t.replace("\u00A0", " ").replace("—", "-").replace("–", "-"))
    tl = t.lower()
    if tl in {"y","yes"}: return "current"
    if tl in {"n","no"}:  return "none"
    if "candidate" in tl: return "candidate"
    if "former" in tl or "graduated" in tl or "terminated" in tl or "no longer" in tl: return "former"
    if "not" in tl and "sff" in tl: return "none"
    if tl == "sff" or tl.startswith("sff") or (" sff" in tl): return "current"
    return "unknown"

def coalesce_sff_class(text_cls: pd.Series, facility_bool: pd.Series) -> pd.Series:
    out = text_cls.copy()
    mask = out.isna() | (out == "unknown")
    if mask.any():
        tmp = pd.Series(pd.NA, index=out.index, dtype="object")
        tmp.loc[facility_bool == True]  = "current"
        tmp.loc[facility_bool == False] = "none"
        out = out.mask(mask & tmp.notna(), tmp)
    return out.fillna("unknown").astype("string")

# ============================ Period inference (from file contents) ===========
def infer_period_from_file(df: pd.DataFrame) -> tuple[int,int]:
    """
    Use FILEDATE or Processing Date inside the CSV (after norm_cols -> filedate/processing_date).
    Values look like 2023-09-01. Returns (yyyy, mm).
    """
    cand_cols = ["filedate", "processing_date"]
    for c in cand_cols:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notna().any():
                d0 = dt[dt.notna()].iloc[0]
                return int(d0.year), int(d0.month)
    raise ValueError("Could not infer period from FILEDATE / Processing Date in CSV.")

# ============================ Standardize one month ===========================
def standardize_provider_info(df: pd.DataFrame, yyyy_hint: int, mm_hint: int) -> pd.DataFrame:
    df = norm_cols(df)

    # --- Period from file contents ---
    yyyy_use, mm_use = infer_period_from_file(df)

    # Primary CCN
    present_cands = [c for c in PRIMARY_CCN_ORDER if c in df.columns]
    primary = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in present_cands:
        primary = primary.mask(primary.isna() & df[c].notna(), df[c])
    cleaned_ccn = primary.map(clean_primary_ccn)

    # Hospital flag → 0/1 Int8 (no FutureWarning)
    hosp = pd.Series(pd.NA, index=df.index, dtype="object")
    for cand in HOSP_CANDIDATES:
        if cand in df.columns:
            mapped = to_boolish(df[cand])
            hosp = hosp.mask(hosp.isna() & mapped.notna(), mapped)
    hosp = hosp.astype("boolean")
    hosp01 = hosp.fillna(False).astype("Int8")

    # CCRC → 0/1 Int8 (no FutureWarning)
    ccrc_bool = pd.Series(pd.NA, index=df.index, dtype="object")
    for cand in CCRC_CANDS:
        if cand in df.columns:
            mapped = to_boolish(df[cand])
            ccrc_bool = ccrc_bool.mask(ccrc_bool.isna() & mapped.notna(), mapped)
    ccrc_bool = ccrc_bool.astype("boolean")
    ccrc01 = ccrc_bool.fillna(False).astype("Int8")

    # SFF (from status text and facility Y/N only) → 0/1 Int8
    sff_status_text = None
    for cand in SFF_STATUS_TEXT_CANDS:
        if c in df.columns:
            sff_status_text = df[c]
            break
    sff_text_cls = pd.Series(pd.NA, index=df.index, dtype="object")
    if sff_status_text is not None:
        sff_text_cls = sff_status_text.map(classify_sff_text)

    sff_facility_bool = pd.Series(pd.NA, index=df.index, dtype="object")
    for cand in SFF_FACILITY_CANDS:
        if cand in df.columns:
            mapped = to_boolish(df[cand])
            sff_facility_bool = sff_facility_bool.mask(sff_facility_bool.isna() & mapped.notna(), mapped)
    sff_facility_bool = sff_facility_bool.astype("boolean")
    sff01 = coalesce_sff_class(sff_text_cls, sff_facility_bool).isin(["current","candidate"]).astype("Int8")

    # Case-mix (verbatim) — first non-null among candidates
    case_mix = pd.Series(pd.NA, index=df.index, dtype="object")
    for cand in CASE_MIX_CANDS:
        if cand in df.columns:
            case_mix = case_mix.mask(case_mix.isna() & df[cand].notna(), df[cand])

    # Period fields from file contents
    quarter    = f"{yyyy_use:04d}Q{(mm_use - 1)//3 + 1}"
    year_month = f"{yyyy_use:04d}/{mm_use:02d}"

    out = pd.DataFrame({
        "cms_certification_number": cleaned_ccn,
        "quarter": quarter,
        "year_month": year_month,
        "provider_resides_in_hospital": hosp01,
        "ccrc_facility": ccrc01,
        "sff_facility": sff01,
        "case_mix_total": case_mix,
    })

    # drop invalid CCN & duplicates
    out = out.dropna(subset=["cms_certification_number"]).drop_duplicates()
    out = out.sort_values(["cms_certification_number", "year_month"], kind="mergesort").reset_index(drop=True)
    return out

# ============================ Extract → Standardize → Write ===================
def extract_and_standardize():
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
                                        chosen = e; break
                                if chosen: break
                            if not chosen: continue

                            raw = mz.read(chosen)
                            df = safe_read_csv(raw)
                            std = standardize_provider_info(df, yyyy, mm)
                            out_name = f"provider_info_{yyyy:04d}_{mm:02d}.csv"
                            std.to_csv(PROV_DIR / out_name, index=False)
                            print(f"[save] {out_name:>22}  rows={len(std):,}")
                            written += 1
                    except zipfile.BadZipFile:
                        continue
    print(f"\n[extract+standardize] wrote {written} monthly provider_info CSV(s).")

# ============================ Combine (de-dupe then 2Q lead) ==================
def combine_monthlies_and_save():
    monthly = sorted(PROV_DIR.glob("provider_info_*.csv"))
    if not monthly:
        raise FileNotFoundError(f"No provider_info_*.csv files found in {PROV_DIR}")

    frames = []
    for p in monthly:
        try:
            # Ensure CCN is string to avoid mixed-type warnings
            df = pd.read_csv(p, dtype={"cms_certification_number":"string"}, low_memory=False)
            # enforce types on dummies (no FutureWarning)
            for col in ["provider_resides_in_hospital","ccrc_facility","sff_facility"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("Int8")
            frames.append(df)
        except Exception as e:
            print(f"[warn] failed reading {p.name}: {e}")

    prov = pd.concat(frames, ignore_index=True)

    # Keep only the required schema (and in that order)
    keep_cols = [
        "cms_certification_number",
        "quarter",
        "year_month",
        "provider_resides_in_hospital",
        "ccrc_facility",
        "sff_facility",
        "case_mix_total",
    ]
    prov = prov[keep_cols]

    # --- strict sanity: drop rows missing essentials
    prov = prov.dropna(subset=["cms_certification_number","quarter","year_month"])

    # --- de-duplicate BEFORE lead to prevent misalignment
    prov = (prov
            .drop_duplicates(["cms_certification_number","year_month"], keep="first")
            .reset_index(drop=True))

    # --- build a proper month order for stable sorting
    prov["_ord"] = pd.to_datetime(prov["year_month"] + "/01", format="%Y/%m/%d", errors="coerce")

    # --- sort then apply 2-quarter (6-month) LEAD to case_mix_total by CCN
    prov = prov.sort_values(["cms_certification_number","_ord"], kind="mergesort")
    prov["case_mix_total"] = (
        prov.groupby("cms_certification_number", sort=False)["case_mix_total"]
            .shift(-6)   # for each month t, take value from t+6 months
    )

    # --- finalize
    prov = (prov
            .drop(columns=["_ord"])
            .sort_values(["cms_certification_number","year_month"], kind="mergesort")
            .reset_index(drop=True))

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    prov.to_csv(COMBINED_CSV, index=False)
    print(f"[save] combined provider panel → {COMBINED_CSV}  ({len(prov):,} rows)")

# =============================== RUN ==========================================
if __name__ == "__main__":
    extract_and_standardize()
    combine_monthlies_and_save()