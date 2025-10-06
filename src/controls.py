# %% Provider acuity test — OneDrive yearly zips → monthly zips → provider CSVs
import re, zipfile
from io import BytesIO
from pathlib import Path
import pandas as pd
import numpy as np

# -------------------- Paths (explicit, as requested) --------------------
NH_ZIP_DIR  = Path(r"C:\Users\Owner\OneDrive\NursingHomeData\nh-compare")
INTERIM_DIR = Path(r"C:\Repositories\white-bowblis-nhmc\data\interim"); INTERIM_DIR.mkdir(parents=True, exist_ok=True)
OUT_FP      = INTERIM_DIR / "provider_acuity_test_only.csv"

print(f"[paths] NH_ZIP_DIR={NH_ZIP_DIR}")
print(f"[paths] INTERIM_DIR={INTERIM_DIR}")
print(f"[out]   {OUT_FP}")

# -------------------- Small utils --------------------
def safe_read_csv(raw: bytes) -> pd.DataFrame:
    attempts = [
        dict(dtype=str, low_memory=False),
        dict(dtype=str, low_memory=False, encoding="utf-8-sig"),
        dict(dtype=str, low_memory=False, encoding="cp1252"),
        dict(dtype=str, low_memory=False, engine="python", sep=None),
        dict(dtype=str, low_memory=False, engine="python", sep=None, encoding="cp1252"),
    ]
    last_err = None
    for kw in attempts:
        try:
            return pd.read_csv(BytesIO(raw), **kw)
        except Exception as e:
            last_err = e
    raise last_err

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = pd.Index(df.columns).str.strip()
    return df

def normalize_ccn_any(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("").str.strip().str.upper()
    s = s.str.replace(r"[ \-\/\.]", "", regex=True)
    digits = s.str.fullmatch(r"\d+")
    s = s.mask(digits, s.str.zfill(6)).replace({"": pd.NA})
    return s

def to_month(x):
    dt = pd.to_datetime(x, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp("s")

def to_yes_no_flag(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.upper()
    s = s.replace({"": pd.NA, "NA": pd.NA, "NAN": pd.NA})
    yes = {"Y","YES","1","TRUE","T"}
    no  = {"N","NO","0","FALSE","F"}
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    out = out.mask(s.isin(yes), "YES").mask(s.isin(no), "NO")
    return out

def pick(cols, candidates):
    lookup = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand and cand.lower() in lookup:
            return lookup[cand.lower()]
    # loose token match
    for cand in candidates:
        if not cand: continue
        toks = [t for t in re.split(r"[^A-Za-z0-9]+", cand) if t]
        for c in cols:
            if all(t.lower() in c.lower() for t in toks):
                return c
    return None

# Column candidates
PRIMARY_CCN_ORDER = [
    "CMS Certification Number (CCN)",
    "cms certification number",
    "Federal Provider Number",
    "Provider Number",
    "PROVNUM",
    "PRVDR_NUM",
]

CCRC_CANDS = [
    "CCRC_FACIL",
    "Continuing Care Retirement Community",
    "CCRC",
]

SFF_CANDS = [
    "SFF",                      # older flat “SFF” column (Y/N)
    "Special Focus Status",     # newer “NONE / CANDIDATE / SFF”
    "SFF Status",
    "SFF_STATUS",
]

CM_TOTAL_CANDS = [
    "CM_TOTAL",
    "Case-Mix Total Nurse Staffing Hours per Resident per Day",
    "CASE-MIX TOTAL NURSE STAFFING HOURS PER RESIDENT PER DAY",
    "CM TOTAL",
]

MONTH_CANDS = [
    "Processing Date", "FILEDATE", "Filedate", "PROCESSING_DATE", "FILE_DATE", "Month"
]

# Parse MM/YYYY from inner zip names
MONTH_RE = r"(0[1-9]|1[0-2])"; YEAR_RE = r"(20\d{2})"
FN_PATTERNS = [
    re.compile(rf"nh_archive_{MONTH_RE}_{YEAR_RE}", re.I),
    re.compile(rf"nh_archive_{YEAR_RE}_{MONTH_RE}", re.I),
    re.compile(rf"nursing_homes_including_rehab_services_archive_{MONTH_RE}_{YEAR_RE}", re.I),
    re.compile(rf"(?:^|[_-]){MONTH_RE}[_-]{YEAR_RE}(?:[_-]|$)", re.I),
    re.compile(rf"(?:^|[_-]){YEAR_RE}[_-]{MONTH_RE}(?:[_-]|$)", re.I),
]
def parse_mm_yyyy_from_name(name: str):
    for pat in FN_PATTERNS:
        m = pat.search(name)
        if m:
            nums = [int(x) for x in m.groups() if x and x.isdigit()]
            if len(nums) >= 2:
                a, b = nums[0], nums[1]
                if a <= 12 and b >= 2000: return a, b
                if b <= 12 and a >= 2000: return b, a
    return (None, None)

def std_month(df: pd.DataFrame, yyyy: int|None, mm: int|None) -> pd.DataFrame:
    df = norm_cols(df)
    cols = list(df.columns)

    # Map essentials
    ccn_col = pick(cols, PRIMARY_CCN_ORDER)
    if not ccn_col:
        return pd.DataFrame()

    month_col = pick(cols, MONTH_CANDS)
    ccrc_col  = pick(cols, CCRC_CANDS)
    sff_col   = pick(cols, SFF_CANDS)
    cm_col    = pick(cols, CM_TOTAL_CANDS)

    out = pd.DataFrame({
        "cms_certification_number": normalize_ccn_any(df[ccn_col]),
        "date": pd.NaT
    })
    if month_col:
        out["date"] = to_month(df[month_col])
    if out["date"].isna().all() and (yyyy and mm):
        out["date"] = pd.Timestamp(year=int(yyyy), month=int(mm), day=1)

    # Variables
    out["ccrc_facil"] = (
        to_yes_no_flag(df[ccrc_col]).map({"YES":1, "NO":0}).astype("Int8")
        if ccrc_col else pd.NA
    )

    if sff_col:
        raw = df[sff_col].astype("string").str.strip()
        up  = raw.str.upper()
        sff_norm = pd.Series(pd.NA, index=raw.index, dtype="object")
        # If a booleanish “SFF” column → map YES/NO
        # If a label column (NONE/CANDIDATE/SFF) → preserve (and normalize case)
        sff_norm = sff_norm.mask(up.isin({"SFF","YES","Y","1"}), "SFF")
        sff_norm = sff_norm.mask(up.isin({"CANDIDATE","CAND","POTENTIAL"}), "CANDIDATE")
        sff_norm = sff_norm.mask(up.isin({"NONE","NO","N","0"}), "NONE")
        # If still NA, pass through original (for any unexpected label)
        out["sff_status_full"] = sff_norm.fillna(raw).replace({"": pd.NA})
    else:
        out["sff_status_full"] = pd.NA

    out["cm_total"] = pd.to_numeric(df[cm_col], errors="coerce") if cm_col else pd.NA

    # Final hygiene
    out = out.dropna(subset=["cms_certification_number"])
    if out["date"].isna().all():
        return pd.DataFrame()
    out["date"] = to_month(out["date"])
    out = out.dropna(subset=["date"])
    out["year"]  = out["date"].dt.year.astype("Int64")
    out["month"] = out["date"].dt.month.astype("Int64")
    return out.sort_values(["cms_certification_number","date"]).reset_index(drop=True)

# -------------------- Ingest yearly zips → monthly zips --------------------
yearly_zips = sorted([p for p in NH_ZIP_DIR.glob("*.zip") if p.is_file()])
if not yearly_zips:
    print("[scan] no yearly zips found in the folder you specified.")
    # Early save (empty) so you can see the file exists
    pd.DataFrame(columns=["cms_certification_number","date","ccrc_facil","sff_status_full","cm_total","year","month"]).to_csv(OUT_FP, index=False)
else:
    print(f"[scan] found {len(yearly_zips)} yearly zip(s)")
    frames = []
    month_files_seen = 0

    for yzip in yearly_zips:
        with zipfile.ZipFile(yzip, "r") as yz:
            inner_zips = [n for n in yz.namelist() if n.lower().endswith(".zip")]
            if not inner_zips:
                print(f"  [warn] {yzip.name} has no inner monthly zips")
                continue
            for inner in inner_zips:
                mm, yyyy = parse_mm_yyyy_from_name(Path(inner).name)
                with yz.open(inner) as inner_bytes:
                    try:
                        with zipfile.ZipFile(BytesIO(inner_bytes.read()), "r") as mz:
                            entries = mz.namelist()
                            # Prefer files with "providerinfo" in name; else any CSV/TXT
                            chosen = None
                            for e in entries:
                                nm = Path(e).name.lower()
                                if ("providerinfo" in nm) and re.search(r"\.(csv|txt)$", nm, re.I):
                                    chosen = e; break
                            if not chosen:
                                cands = [e for e in entries if re.search(r"\.(csv|txt)$", e, re.I)]
                                chosen = cands[0] if cands else None
                            if not chosen:
                                continue
                            raw = mz.read(chosen)
                            df  = safe_read_csv(raw)
                            std = std_month(df, yyyy, mm)
                            if not std.empty:
                                frames.append(std)
                                month_files_seen += 1
                    except zipfile.BadZipFile:
                        continue

    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not panel.empty:
        panel = panel.drop_duplicates(["cms_certification_number","date"]).reset_index(drop=True)

    print(f"[ingested] month-files={month_files_seen} rows={len(panel):,} "
          f"CCNs={panel['cms_certification_number'].nunique() if not panel.empty else 0:,}")

    # -------------------- Save + quick diagnostics --------------------
    panel.to_csv(OUT_FP, index=False)
    print(f"[save] {OUT_FP}  rows={len(panel):,}")

    if panel.empty:
        print("[warn] panel is empty. Two common causes:")
        print("  1) provider CSV inside monthly zips is named differently — print(mz.namelist()) to check")
        print("  2) CCN header variant not matched — print(df.columns[:50]) for a month to add an alias")
    else:
        panel["year"] = panel["date"].dt.year
        print("\n=== Missingness by year (non-null counts) ===")
        miss = (panel.assign(
                    cm_total_n=panel["cm_total"].notna().astype(int),
                    ccrc_n    =panel["ccrc_facil"].notna().astype(int),
                    sff_n     =panel["sff_status_full"].notna().astype(int))
                .groupby("year")[["cm_total_n","ccrc_n","sff_n"]].sum().astype(int))
        print(miss)

        if panel["ccrc_facil"].notna().any():
            ccrc_share = (panel.dropna(subset=["ccrc_facil"])
                               .groupby("year")["ccrc_facil"].mean().mul(100).round(1))
            print("\n=== CCRC share by year (among non-null, %) ===")
            print(ccrc_share)

        if panel["sff_status_full"].notna().any():
            sff_tab = (panel.dropna(subset=["sff_status_full"])
                             .groupby(["year","sff_status_full"]).size()
                             .groupby(level=0).apply(lambda s: (s/s.sum()*100))
                             .unstack(fill_value=0)
                             .reindex(columns=[c for c in ["NONE","CANDIDATE","SFF"]
                                               if c in panel["sff_status_full"].dropna().unique()])
                             .round(1))
            print("\n=== SFF status % by year ===")
            print(sff_tab)

        if panel["cm_total"].notna().any():
            cm = (panel.groupby("year")["cm_total"]
                        .agg(count="count", mean="mean", std="std",
                             min="min", q10=lambda s: s.quantile(.10),
                             q25=lambda s: s.quantile(.25), q50="median",
                             q75=lambda s: s.quantile(.75), q90=lambda s: s.quantile(.90), max="max")
                        .round(2))
            print("\n=== CM_TOTAL summary by year ===")
            print(cm)