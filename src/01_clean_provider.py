#!/usr/bin/env python
# coding: utf-8
# -----------------------------------------------------------------------------
# CMS Provider Info — Extract → Standardize → Combine
#
# Changes:
#   • sff_facility dummy from status text:
#       sff_facility = 1 if status ∈ {"SFF","SFF Candidate"} (i.e., class ∈ {current,candidate})
#                    = 0 otherwise (former/none/unknown)
#   • case_mix_total_num is LEADed by TWO QUARTERS (6 months) in the COMBINED output
#       (we overwrite the column with its 6-month lead; name unchanged)
#   • ccrc_facility stored as 0/1 (Int8) in combined output
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

COMBINED_CSV   = PROV_DIR / "provider_info_combined.csv"
HOSP_CSV       = PROV_DIR / "provider_resides_in_hospital_by_ccn.csv"     # latest TRUE-only per CCN
HOSP_PANEL_CSV = PROV_DIR / "provider_resides_in_hospital_panel.csv"      # TRUE-only month-by-month rows

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
    dash_chars = r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212-]"
    cols = pd.Index(df.columns)
    cols = cols.str.replace("\u00A0", " ", regex=False)
    cols = cols.str.replace(dash_chars, " ", regex=True)
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

def pick_first_nonnull(df: pd.DataFrame, candidates: list[str]) -> tuple[pd.Series, pd.Series]:
    out  = pd.Series(pd.NA, index=df.index, dtype="object")
    used = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in candidates:
        if c not in df.columns:
            continue
        fill = out.isna() & df[c].notna()
        out = out.mask(fill, df[c])
        used = used.mask(fill, c)
    return out, used

def pick_bool(df: pd.DataFrame, candidates: list[str]) -> tuple[pd.Series, pd.Series]:
    raw, used = pick_first_nonnull(df, candidates)
    mapped = to_boolish(raw)
    return mapped, used

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
CASE_MIX_CANDS = [
    "exp_total",
    "cm_total",
    "case_mix_total_nurse_staffing_hours_per_resident_per_day",
    "casemix_total_nurse_staffing_hours_per_resident_per_day",
]
CASE_MIX_SRC_CANDS = ["cm_total_src"]

CCRC_CANDS = [
    "ccrc_facil",
    "continuing_care_retirement_community",
]

SFF_STATUS_TEXT_CANDS = ["special_focus_status"]
SFF_FACILITY_CANDS    = ["special_focus_facility"]
SFF_FLAG_CANDS        = ["sff"]

# ============================ CCN cleaning ====================================
ALNUM_6_7 = re.compile(r"^[0-9A-Z]{6,7}$")
def clean_primary_ccn(val: str) -> str | float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return np.nan
    s = str(val).strip().upper()
    if "+" in s or "." in s:  # junk
        return np.nan
    if s.isdigit():
        return s.zfill(6)
    if ALNUM_6_7.fullmatch(s):
        return s
    return np.nan

# ============================ SFF classification ==============================
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

def coalesce_sff_class(text_cls: pd.Series,
                       facility_bool: pd.Series,
                       simple_bool: pd.Series) -> pd.Series:
    out = text_cls.copy()
    mask = out.isna() | (out == "unknown")
    if mask.any():
        tmp = pd.Series(pd.NA, index=out.index, dtype="object")
        tmp.loc[facility_bool == True]  = "current"
        tmp.loc[facility_bool == False] = "none"
        out = out.mask(mask & tmp.notna(), tmp)
    mask = out.isna() | (out == "unknown")
    if mask.any():
        tmp2 = pd.Series(pd.NA, index=out.index, dtype="object")
        tmp2.loc[simple_bool == True]  = "current"
        tmp2.loc[simple_bool == False] = "none"
        out = out.mask(mask & tmp2.notna(), tmp2)
    return out.fillna("unknown").astype("string")

# ============================ Standardize one month ===========================
def standardize_provider_info(df: pd.DataFrame, yyyy: int, mm: int, source_name: str) -> pd.DataFrame:
    df = norm_cols(df)

    # Hospital flag
    hosp = pd.Series(pd.NA, index=df.index, dtype="object")
    for cand in HOSP_CANDIDATES:
        if cand in df.columns:
            mapped = to_boolish(df[cand])
            hosp = hosp.mask(hosp.isna() & mapped.notna(), mapped)

    # Provider name
    pname = pd.Series(pd.NA, index=df.index, dtype="object")
    for cand in NAME_CANDIDATES:
        if cand in df.columns:
            pname = pname.mask(pname.isna() & df[cand].notna(), df[cand])

    # Primary CCN
    present_cands = [c for c in PRIMARY_CCN_ORDER if c in df.columns]
    primary = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in present_cands:
        primary = primary.mask(primary.isna() & df[c].notna(), df[c])
    cleaned = primary.map(clean_primary_ccn)
    raw_ccn_cols = {f"ccn_raw_{c}": df[c] for c in present_cands}

    # Case-mix (unlagged here; we lead in combine step)
    case_mix, raw_cm_used       = pick_first_nonnull(df, CASE_MIX_CANDS)
    case_mix_src, raw_cm_src_used = pick_first_nonnull(df, CASE_MIX_SRC_CANDS)
    case_mix_num = pd.to_numeric(case_mix, errors="coerce")

    # CCRC
    ccrc_bool, raw_ccrc_used = pick_bool(df, CCRC_CANDS)

    # SFF bits
    sff_status_text, raw_sff_status_text_used = pick_first_nonnull(df, SFF_STATUS_TEXT_CANDS)
    sff_facility_bool, raw_sff_facility_used  = pick_bool(df, SFF_FACILITY_CANDS)
    sff_flag_bool,     raw_sff_flag_used      = pick_bool(df, SFF_FLAG_CANDS)
    sff_text_cls = sff_status_text.map(classify_sff_text)
    sff_class    = coalesce_sff_class(sff_text_cls, sff_facility_bool, sff_flag_bool)

    # >>> SFF facility dummy from status text (current/candidate = 1) <<<
    sff_facility_dummy = sff_class.isin(["current","candidate"]).astype("Int8")

    df_out = pd.DataFrame({
        "cms_certification_number": cleaned,
        "provider_name": pname,
        "provider_resides_in_hospital": hosp,
        "year": int(yyyy),
        "month": int(mm),
        "date": pd.Timestamp(year=int(yyyy), month=int(mm), day=1).strftime("%Y-%m-%d"),
        "source_file": Path(source_name).name,

        "case_mix_total": case_mix,
        "case_mix_total_num": case_mix_num,
        "case_mix_total_src": case_mix_src,

        # keep original bool for reference; will be cast to 0/1 in combine
        "ccrc_facility": ccrc_bool,

        "sff_facility": sff_facility_dummy,
        "sff_flag": sff_flag_bool,
        "sff_status_text": sff_status_text,
        "sff_class": sff_class,
        "sff_current": sff_class.eq("current"),
        "sff_candidate": sff_class.eq("candidate"),
        "sff_former": sff_class.eq("former"),

        "raw_case_mix_total_col": raw_cm_used,
        "raw_case_mix_total_src_col": raw_cm_src_used,
        "raw_ccrc_col": raw_ccrc_used,
        "raw_sff_facility_col": raw_sff_facility_used,
        "raw_sff_flag_col": raw_sff_flag_used,
        "raw_sff_status_text_col": raw_sff_status_text_used,
    })

    for k, v in raw_ccn_cols.items():
        df_out[k] = v

    # drop invalid CCN & dups
    df_out = df_out.dropna(subset=["cms_certification_number"]).drop_duplicates()
    df_out = df_out.sort_values(["cms_certification_number","date"], kind="mergesort").reset_index(drop=True)
    return df_out

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
                            std = standardize_provider_info(
                                df, yyyy, mm,
                                f"{yzip.name}!{Path(inner).name}!{Path(chosen).name}"
                            )
                            out_name = f"provider_info_{yyyy:04d}_{mm:02d}.csv"
                            (PROV_DIR / out_name).write_text(std.to_csv(index=False))
                            print(f"[save] {out_name:>22}  rows={len(std):,}"); written += 1
                    except zipfile.BadZipFile:
                        continue
    print(f"\n[extract+standardize] wrote {written} monthly provider_info CSV(s).")

# ====================== TRUE-ONLY hospital panel & latest list =================
def build_hospital_flags(prov: pd.DataFrame):
    map_bool = {"True":True, "False":False, True:True, False:False, "true":True, "false":False}
    prov = prov.copy()
    prov["provider_resides_in_hospital"] = prov["provider_resides_in_hospital"].map(map_bool).fillna(False)

    panel_true = (
        prov.loc[prov["provider_resides_in_hospital"] == True,
                 ["cms_certification_number", "date", "provider_name"]]
          .dropna(subset=["cms_certification_number","date"])
          .drop_duplicates(["cms_certification_number","date"])
          .sort_values(["cms_certification_number","date"], kind="mergesort")
          .reset_index(drop=True)
    )
    panel_true["provider_resides_in_hospital"] = True
    panel_true.to_csv(HOSP_PANEL_CSV, index=False)
    print(f"[save] hospital panel (TRUE-only) → {HOSP_PANEL_CSV}  ({len(panel_true):,} rows, {panel_true['cms_certification_number'].nunique():,} CCNs)")

    last_true = (
        panel_true.sort_values(["cms_certification_number","date"], kind="mergesort")
                  .groupby("cms_certification_number", as_index=False, sort=False)
                  .last()[["cms_certification_number","provider_resides_in_hospital","provider_name"]]
    )
    last_true.to_csv(HOSP_CSV, index=False)
    print(f"[save] hospital list (latest TRUE only) → {HOSP_CSV}  ({len(last_true):,} CCNs)")
    return panel_true, last_true

# ============================ Case-mix: 2-quarter LEAD ========================
def apply_case_mix_two_quarter_lead(prov: pd.DataFrame) -> pd.DataFrame:
    """
    Overwrite case_mix_total_num with its 6-month LEAD (2 quarters), by CCN.
    Reported in month t corresponds to actual case-mix for t+6 months.
    """
    prov = prov.copy()
    prov["date"] = pd.to_datetime(prov["date"], errors="coerce")
    # build lookup of value at (ccn, date+6m)
    look = prov[["cms_certification_number","date","case_mix_total_num"]].copy()
    look["date"] = look["date"] + pd.DateOffset(months=6)  # shift forward
    look = look.rename(columns={"case_mix_total_num": "_lead_val"})
    prov = prov.merge(look, on=["cms_certification_number","date"], how="left")
    prov["case_mix_total_num"] = prov["_lead_val"]
    prov = prov.drop(columns=["_lead_val"])
    return prov

# ============================ Combine + hospital list =========================
def combine_and_make_hospital_list():
    monthly = sorted(PROV_DIR.glob("provider_info_*.csv"))
    if not monthly:
        raise FileNotFoundError(f"No provider_info_*.csv files found in {PROV_DIR}")

    frames = []
    for p in monthly:
        try:
            df = pd.read_csv(p, dtype=str, low_memory=False)
            df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
            # types we altered
            if "sff_facility" in df.columns:
                df["sff_facility"] = pd.to_numeric(df["sff_facility"], errors="coerce").fillna(0).astype("Int8")
            if "case_mix_total_num" in df.columns:
                df["case_mix_total_num"] = pd.to_numeric(df["case_mix_total_num"], errors="coerce")
            # ccrc_facility to 0/1 Int8
            if "ccrc_facility" in df.columns:
                if str(df["ccrc_facility"].dtype) == "boolean":
                    df["ccrc_facility"] = df["ccrc_facility"].astype("Int8")
                else:
                    s = df["ccrc_facility"].astype("string").str.strip().str.lower()
                    df["ccrc_facility"] = s.map({
                        "1":1,"true":1,"t":1,"yes":1,"y":1,
                        "0":0,"false":0,"f":0,"no":0,"n":0
                    }).astype("Int8")
            frames.append(df)
        except Exception as e:
            print(f"[warn] failed reading {p.name}: {e}")

    prov = pd.concat(frames, ignore_index=True).drop_duplicates()

    # >>> apply 6-month LEAD to case_mix_total_num <<<
    if "case_mix_total_num" in prov.columns:
        prov = apply_case_mix_two_quarter_lead(prov)

    prov = prov.sort_values(["cms_certification_number","date","source_file"], kind="mergesort").reset_index(drop=True)
    prov.to_csv(COMBINED_CSV, index=False)
    print(f"[save] combined (with 2-quarter LEAD on case_mix_total_num) → {COMBINED_CSV}  ({len(prov):,} rows)")

    build_hospital_flags(prov)

# =============================== RUN ==========================================
if __name__ == "__main__":
    extract_and_standardize()
    combine_and_make_hospital_list()