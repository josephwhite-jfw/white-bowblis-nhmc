# 01_provider_info_pipeline.py
import re, zipfile, os
from pathlib import Path
from io import BytesIO
import pandas as pd, numpy as np
from config import NH_COMPARE_DIR, PROV_DIR, PROV_COMBINED, HOSP_PANEL_CSV, HOSP_LATEST_CSV

# ---------- helpers ----------
def norm_cols(df):
    dash = r"[\u2010-\u2015\u2212-]"
    c = (pd.Index(df.columns)
         .str.replace("\u00A0", " ", regex=False)
         .str.replace(dash, " ", regex=True)
         .str.strip().str.lower()
         .str.replace(r"\s+", "_", regex=True)
         .str.replace(r"[^0-9a-z_]", "", regex=True))
    df.columns = c
    return df

def to_boolish(s):
    s = s.astype("string").str.strip().str.lower()
    return s.map({"1":True,"y":True,"yes":True,"true":True,"t":True,
                  "0":False,"n":False,"no":False,"false":False,"f":False}).astype("boolean")

def clean_ccn(x):
    if pd.isna(x): return pd.NA
    s = str(x).strip().upper()
    if not s: return pd.NA
    if s.isdigit(): return s.zfill(6)
    if re.fullmatch(r"[0-9A-Z]{6,7}", s): return s
    return pd.NA

def safe_read_csv(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8","utf-8-sig","cp1252","latin-1"):
        try:
            return pd.read_csv(BytesIO(raw), dtype=str, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(BytesIO(raw), dtype=str, encoding="utf-8", encoding_errors="replace", low_memory=False)

MONTH_RE = r"(0[1-9]|1[0-2])"; YEAR_RE = r"(20\d{2})"
INNER_PATTERNS = [re.compile(p, re.I) for p in [
    rf"nh_archive_{MONTH_RE}_{YEAR_RE}\.zip",
    rf"nh_archive_{YEAR_RE}_{MONTH_RE}\.zip",
    rf"nursing_homes_including_rehab_services_archive_{MONTH_RE}_{YEAR_RE}\.zip",
    rf"(?:^|[_-]){MONTH_RE}[_-]{YEAR_RE}\.zip$",
    rf"(?:^|[_-]){YEAR_RE}[_-]{MONTH_RE}\.zip$",
]]

def parse_mm_yyyy(name):
    for pat in INNER_PATTERNS:
        m = pat.search(name)
        if m:
            nums = [int(x) for x in m.groups() if x.isdigit()]
            if len(nums)>=2:
                a,b=nums[0],nums[1]
                if a<=12: return a,b
                if b<=12: return b,a
    return (None,None)

# ---------- standardize one monthly provider-info ----------
PRI_CCN = ["cms_certification_number","cms_certification_number_ccn","federal_provider_number","provnum","provider_id","provider_number"]
NAME_CANDS = ["provider_name","provname"]
HOSP_CANDS = ["provider_resides_in_hospital","resides_in_hospital","provider_resides_in_hospital_","inhosp"]
CASE_MIX_CANDS = ["exp_total","cm_total","case_mix_total_nurse_staffing_hours_per_resident_per_day","casemix_total_nurse_staffing_hours_per_resident_per_day"]
SFF_STATUS = ["special_focus_status"]
SFF_FAC    = ["special_focus_facility"]
SFF_FLAG   = ["sff"]
CCRC_CANDS = ["ccrc_facil","continuing_care_retirement_community"]

def classify_sff_text(text):
    if text is None or pd.isna(text): return None
    t = str(text).strip().lower()
    if t in {"y","yes","sff"}: return "current"
    if t in {"n","no"}: return "none"
    if "candidate" in t: return "candidate"
    if "former" in t or "graduated" in t or "terminated" in t: return "former"
    if "special" in t and "focus" in t: return "current"
    return "unknown"

def pick_first(df, cols):
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    used = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in cols:
        if c in df.columns:
            mask = out.isna() & df[c].notna()
            out = out.mask(mask, df[c])
            used = used.mask(mask, c)
    return out, used

def standardize_provider_info(df, yyyy, mm, source_name):
    df = norm_cols(df)

    # CCN, name, hosp
    ccn_series,_ = pick_first(df, [c for c in PRI_CCN if c in df.columns])
    ccn = ccn_series.map(clean_ccn)
    name,_ = pick_first(df, NAME_CANDS)
    hosp,_ = pick_first(df, HOSP_CANDS); hosp = to_boolish(hosp)

    # case-mix (numeric; LEAD applied later)
    cm_raw,_ = pick_first(df, CASE_MIX_CANDS)
    cm_num = pd.to_numeric(cm_raw, errors="coerce")

    # CCRC
    ccrc_raw,_ = pick_first(df, CCRC_CANDS)
    ccrc = to_boolish(ccrc_raw)

    # SFF class and facility dummy (current/candidate = 1)
    sff_status,_ = pick_first(df, SFF_STATUS)
    sff_fac_raw,_= pick_first(df, SFF_FAC); sff_fac = to_boolish(sff_fac_raw)
    sff_flag_raw,_= pick_first(df, SFF_FLAG); sff_flag = to_boolish(sff_flag_raw)
    sff_class = sff_status.map(classify_sff_text).astype("string")
    sff_class = sff_class.fillna("unknown")
    sff_facility = sff_class.isin(["current","candidate"]).astype("Int8")

    out = pd.DataFrame({
        "cms_certification_number": ccn,
        "provider_name": name,
        "provider_resides_in_hospital": hosp,
        "date": pd.Timestamp(year=int(yyyy), month=int(mm), day=1),
        "source_file": Path(source_name).name,
        "case_mix_total_num": cm_num,
        "ccrc_facility": ccrc,              # boolean for now; convert later
        "sff_facility": sff_facility,      # Int8 0/1
        "sff_class": sff_class
    }).dropna(subset=["cms_certification_number"])
    return out.drop_duplicates().sort_values(["cms_certification_number","date"])

# ---------- extract all monthly CSVs from yearly zips ----------
def extract_all():
    yearly = sorted(p for p in NH_COMPARE_DIR.glob("nh_archive_*.zip") if p.is_file())
    if not yearly:
        raise FileNotFoundError(f"No yearly NH Compare zips found in {NH_COMPARE_DIR}")
    frames=[]
    for yzip in yearly:
        with zipfile.ZipFile(yzip, "r") as yz:
            for inner in [n for n in yz.namelist() if n.lower().endswith(".zip")]:
                mm,yyyy = parse_mm_yyyy(Path(inner).name)
                if not (mm and yyyy): continue
                with yz.open(inner) as inner_bytes:
                    try:
                        with zipfile.ZipFile(BytesIO(inner_bytes.read()), "r") as mz:
                            # pick the best CSV inside
                            entries = mz.namelist()
                            candidates = [e for e in entries if Path(e).suffix.lower()==".csv"]
                            # prefer providerinfo_download/display
                            chosen = None
                            for pr in ("providerinfo_download.csv","providerinfo_display.csv","nh_providerinfo"):
                                chosen = next((e for e in candidates if pr in Path(e).name.lower()), None)
                                if chosen: break
                            if not chosen and candidates:
                                chosen = candidates[0]
                            if not chosen: continue
                            raw = mz.read(chosen)
                            df = safe_read_csv(raw)
                            std = standardize_provider_info(df, yyyy, mm, f"{yzip.name}!{Path(inner).name}!{Path(chosen).name}")
                            frames.append(std)
                    except zipfile.BadZipFile:
                        continue
    prov = (pd.concat(frames, ignore_index=True)
              .drop_duplicates()
              .sort_values(["cms_certification_number","date","source_file"]))
    # convert ccrc -> Int8(0/1)
    if "ccrc_facility" in prov.columns:
        prov["ccrc_facility"] = prov["ccrc_facility"].astype("boolean").fillna(False).astype("Int8")

    # apply 6-month LEAD to case_mix_total_num (shift forward by +6 months within CCN)
    prov["_date_plus_6"] = prov["date"] + pd.DateOffset(months=6)
    look = prov[["cms_certification_number","_date_plus_6","case_mix_total_num"]].rename(
        columns={"_date_plus_6":"date","case_mix_total_num":"_lead"})
    prov = prov.merge(look, on=["cms_certification_number","date"], how="left")
    prov["case_mix_total_num"] = prov["_lead"]; prov = prov.drop(columns=["_lead","_date_plus_6"])

    # write combined
    prov.to_csv(PROV_COMBINED, index=False)
    print(f"[provider] combined rows={len(prov):,} CCNs={prov['cms_certification_number'].nunique():,} → {PROV_COMBINED}")

    # hospital TRUE-only panel and latest list
    hosp_true = (prov.loc[prov["provider_resides_in_hospital"]==True,
                          ["cms_certification_number","date","provider_name"]]
                    .dropna(subset=["cms_certification_number","date"])
                    .drop_duplicates()
                    .sort_values(["cms_certification_number","date"]))
    hosp_true["provider_resides_in_hospital"] = True
    hosp_true.to_csv(HOSP_PANEL_CSV, index=False)
    latest = (hosp_true.groupby("cms_certification_number", as_index=False)
                      .last()[["cms_certification_number","provider_resides_in_hospital","provider_name"]])
    latest.to_csv(HOSP_LATEST_CSV, index=False)
    print(f"[hospital] panel rows={len(hosp_true):,} latest CCNs={len(latest):,}")

if __name__ == "__main__":
    extract_all()