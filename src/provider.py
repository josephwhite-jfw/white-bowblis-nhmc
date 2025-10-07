# nhmc/provider.py
# Extract + standardize CMS Provider Info → combined file + hospital flags

from __future__ import annotations
import re, zipfile
from io import BytesIO
from pathlib import Path
import pandas as pd
import numpy as np

from .paths import RAW_DIR, PROV_DIR, PROV_COMBINED, HOSP_PANEL, HOSP_BY_CCN
from .utils import normalize_ccn_any, bool_from_any, to_monthstart

NH_ZIP_DIR = RAW_DIR / "nh-compare"
PROV_DIR.mkdir(parents=True, exist_ok=True)

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

def _parse_mm_yyyy(name: str):
    for pat in INNER_PATTERNS:
        m = pat.search(name)
        if m:
            nums = [int(x) for x in m.groups() if x and x.isdigit()]
            if len(nums) >= 2:
                a, b = nums[0], nums[1]
                if a <= 12 and b >= 2000: return a, b
                if b <= 12 and a >= 2000: return b, a
    return (None, None)

def _safe_read_csv(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8","utf-8-sig","cp1252","latin-1"):
        try:
            return pd.read_csv(BytesIO(raw), dtype=str, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(BytesIO(raw), dtype=str, encoding="utf-8", encoding_errors="replace", low_memory=False)

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    dash_chars = r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212-]"
    cols = pd.Index(df.columns)
    cols = cols.str.replace("\u00A0", " ", regex=False)
    cols = cols.str.replace(dash_chars, " ", regex=True)
    cols = cols.str.strip().str.lower()
    cols = cols.str.replace(r"\s+", "_", regex=True)
    cols = cols.str.replace(r"[^0-9a-z_]", "", regex=True)
    df.columns = cols
    return df

def _pick_first(df: pd.DataFrame, cands: list[str]):
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    used = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in cands:
        if c in df.columns:
            fill = out.isna() & df[c].notna()
            out = out.mask(fill, df[c])
            used = used.mask(fill, c)
    return out, used

def _pick_bool(df: pd.DataFrame, cands: list[str]):
    raw, used = _pick_first(df, cands)
    mapped = bool_from_any(raw)
    return mapped, used

CASE_MIX_CANDS = [
    "exp_total",
    "cm_total",
    "case_mix_total_nurse_staffing_hours_per_resident_per_day",
    "casemix_total_nurse_staffing_hours_per_resident_per_day"
]
CCRC_CANDS = ["ccrc_facil","continuing_care_retirement_community"]
SFF_STATUS = ["special_focus_status"]
SFF_FAC    = ["special_focus_facility"]
SFF_FLAG   = ["sff"]

def _classify_sff_text(x: str | float) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)): return "unknown"
    s = str(x).strip().upper()
    if s in {"Y","YES","TRUE","T","1","CURRENT","SFF"}: return "current"
    if "CANDIDATE" in s:  return "candidate"
    if "FORMER" in s or "GRADUATED" in s or "TERMINATED" in s: return "former"
    if s in {"N","NO","FALSE","F","0","NONE"}: return "none"
    if "SPECIAL" in s and "FOCUS" in s: return "current"
    return "unknown"

def _standardize_provider_month(df: pd.DataFrame, yyyy: int, mm: int, source_name: str) -> pd.DataFrame:
    df = _norm_cols(df)

    # CCN (pick any plausible id-like)
    ccn_cands = [c for c in df.columns if c in {
        "cms_certification_number","cms_certification_number_ccn","federal_provider_number",
        "provnum","provider_id","provider_number"
    }]
    ccn = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in ccn_cands:
        ccn = ccn.mask(ccn.isna() & df[c].notna(), df[c])
    ccn = normalize_ccn_any(ccn)

    # Provider name / hospital flag
    pname, _ = _pick_first(df, ["provider_name","provname"])
    hosp,  _ = _pick_bool(df, ["provider_resides_in_hospital","resides_in_hospital","inhosp","provider_resides_in_hospital_"])

    # New vars
    case_mix, _ = _pick_first(df, CASE_MIX_CANDS)
    cm_num = pd.to_numeric(case_mix, errors="coerce")
    ccrc, _    = _pick_bool(df, CCRC_CANDS)
    sff_text, _= _pick_first(df, SFF_STATUS)
    sff_fac, _ = _pick_bool(df, SFF_FAC)
    sff_flag,_ = _pick_bool(df, SFF_FLAG)

    # sff_class coalesce
    sff_text_cls = sff_text.map(_classify_sff_text)
    # prefer text, then facility/flag
    sff = sff_text_cls.copy()
    sff = sff.mask((sff.isna()) | (sff=="unknown"), sff_fac.map({True:"current", False:"none"}))
    sff = sff.mask((sff.isna()) | (sff=="unknown"), sff_flag.map({True:"current", False:"none"}))
    sff = sff.fillna("unknown").astype("string")

    out = pd.DataFrame({
        "cms_certification_number": ccn,
        "provider_name": pname,
        "provider_resides_in_hospital": hosp,
        "month": pd.Timestamp(year=int(yyyy), month=int(mm), day=1),
        "case_mix_total_num": cm_num,
        "ccrc_facility": ccrc,
        "sff_class": sff,
        "source_file": source_name
    })
    out = out.dropna(subset=["cms_certification_number"]).drop_duplicates()
    return out

def build_provider_info(write_monthlies: bool = False) -> None:
    yearlies = sorted(p for p in NH_ZIP_DIR.glob("nh_archive_*.zip") if p.is_file())
    if not yearlies:
        raise FileNotFoundError(f"No yearly zips in {NH_ZIP_DIR}")
    frames = []
    monthly_written = 0

    for yzip in yearlies:
        with zipfile.ZipFile(yzip, "r") as yz:
            inner = [n for n in yz.namelist() if n.lower().endswith(".zip")]
            for inzip in inner:
                mm, yyyy = _parse_mm_yyyy(Path(inzip).name)
                if not (mm and yyyy): 
                    continue
                with yz.open(inzip) as ib:
                    try:
                        with zipfile.ZipFile(BytesIO(ib.read()), "r") as mz:
                            entries = mz.namelist()
                            chosen = None
                            for pat in PRIORITY:
                                for e in entries:
                                    if pat in Path(e).name.lower() and Path(e).suffix.lower()==".csv":
                                        chosen = e; break
                                if chosen: break
                            if not chosen: 
                                continue
                            raw = mz.read(chosen)
                            df = _safe_read_csv(raw)
                            std = _standardize_provider_month(df, yyyy, mm, f"{yzip.name}!{Path(inzip).name}!{Path(chosen).name}")
                            if write_monthlies:
                                out_name = f"provider_info_{yyyy:04d}_{mm:02d}.csv"
                                (PROV_DIR / out_name).write_text(std.to_csv(index=False), encoding="utf-8")
                                monthly_written += 1
                            frames.append(std)
                    except zipfile.BadZipFile:
                        continue

    if not frames:
        raise RuntimeError("No provider files standardized.")
    prov = (pd.concat(frames, ignore_index=True)
              .drop_duplicates()
              .sort_values(["cms_certification_number","month","source_file"])
              .reset_index(drop=True))
    prov.to_csv(PROV_COMBINED, index=False)
    print(f"[provider] combined → {PROV_COMBINED} (rows={len(prov):,}, CCNs={prov['cms_certification_number'].nunique():,})")
    if write_monthlies:
        print(f"[provider] monthly CSVs written: {monthly_written}")

def build_hospital_flags_from_combined() -> None:
    df = pd.read_csv(PROV_COMBINED, low_memory=False)
    df["cms_certification_number"] = normalize_ccn_any(df["cms_certification_number"])
    df["month"] = to_monthstart(df["month"])
    df["provider_resides_in_hospital"] = bool_from_any(df["provider_resides_in_hospital"]).fillna(False)

    panel_true = (df.loc[df["provider_resides_in_hospital"]==True,
                 ["cms_certification_number","month","provider_name"]]
                    .dropna(subset=["cms_certification_number","month"])
                    .drop_duplicates(["cms_certification_number","month"])
                    .rename(columns={"month":"date"})
                    .sort_values(["cms_certification_number","date"]))
    panel_true["provider_resides_in_hospital"] = True
    panel_true.to_csv(HOSP_PANEL, index=False)
    print(f"[hospital] TRUE-only panel → {HOSP_PANEL} (rows={len(panel_true):,})")

    last_true = (panel_true.sort_values(["cms_certification_number","date"])
                           .groupby("cms_certification_number", as_index=False)
                           .last()[["cms_certification_number","provider_resides_in_hospital","provider_name"]])
    last_true.to_csv(HOSP_BY_CCN, index=False)
    print(f"[hospital] latest TRUE list → {HOSP_BY_CCN} (CCNs={len(last_true):,})")