#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ─────────────────────────────────────────────────────────────────────────────
# CMS Ownership — Extract (filename matching) → Standardize In-Place → Combine
#   * Overwrites existing monthly CSVs (no numbered duplicates)
#   * No profiling/QA outputs; deletes any existing profiling/qa_reports folders
#   * Dates stored as YYYY-MM-DD STRINGS (no integer nanoseconds)
# ─────────────────────────────────────────────────────────────────────────────
import os, re, csv, zipfile, shutil, tempfile, warnings
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

# ============================== Config / Paths ================================
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").is_dir() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

RAW_DIR     = Path(os.getenv("NH_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
NH_ZIP_DIR  = RAW_DIR / "nh-compare"
OWN_DIR     = RAW_DIR / "ownership-files"
OWN_DIR.mkdir(parents=True, exist_ok=True)

# Behavior flags
DRY_RUN        = False          # True = preview only
NAME_STYLE     = "yyyy_mm"      # "mm_yyyy" or "yyyy_mm"
DO_STANDARDIZE = True           # standardize in-place before combine
DO_COMBINE     = True           # write final combined CSV
COMBINED_CSV   = OWN_DIR / "ownership_combined.csv"

print(f"[paths] NH_ZIP_DIR={NH_ZIP_DIR}")
print(f"[paths] OWN_DIR={OWN_DIR}")

# ============================ Housekeeping ====================================
# Delete any old profiling / qa_reports folders
for junk in (OWN_DIR / "profiling", OWN_DIR / "qa_reports"):
    if junk.exists() and junk.is_dir():
        shutil.rmtree(junk, ignore_errors=True)

# =========================== Shared helpers ==================================
def sniff_delim(fp: Path, nbytes=8192):
    raw = fp.read_bytes()
    sample = raw[:nbytes]
    try:
        dialect = csv.Sniffer().sniff(sample.decode("utf-8", errors="ignore"))
        return dialect.delimiter
    except Exception:
        return "\t" if sample.count(b"\t") > sample.count(b",") else ","

def read_csv_any(fp: Path, nrows=None):
    """Robust reader (keeps strings) for arbitrary monthly files."""
    delim = sniff_delim(fp)
    encs = ("utf-8","utf-8-sig","cp1252","latin-1")
    for enc in encs:
        try:
            return pd.read_csv(fp, dtype=str, sep=delim, encoding=enc,
                               engine="c", low_memory=False, nrows=nrows)
        except Exception:
            try:
                return pd.read_csv(fp, dtype=str, sep=delim, encoding=enc,
                                   engine="python", on_bad_lines="skip", nrows=nrows)
            except Exception:
                continue
    return pd.read_csv(fp, dtype=str, sep=delim, encoding="utf-8",
                       encoding_errors="replace", engine="python",
                       on_bad_lines="skip", nrows=nrows)

def norm_header(h: str) -> str:
    return re.sub(r"\s+"," ", str(h or "").strip().lower().replace("_"," ")).strip()

def safe_to_datetime(series: pd.Series) -> pd.Series:
    """
    Robust date parser:
      - strips 'since'
      - tries explicit formats; falls back only on strings that look like dates
      - returns datetime64[ns]
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"(?i)^\s*since[:\-]?\s*", "", regex=True)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    m = s.str.match(r"^\d{4}-\d{2}-\d{2}$")
    if m.any():
        out.loc[m] = pd.to_datetime(s[m], format="%Y-%m-%d", errors="coerce")

    m = s.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$")
    if m.any():
        out.loc[m] = pd.to_datetime(s[m], format="%m/%d/%Y", errors="coerce")

    m = s.str.match(r"^\d{1,2}/\d{1,2}/\d{2}$")
    if m.any():
        out.loc[m] = pd.to_datetime(s[m], format="%m/%d/%y", errors="coerce")

    remaining = out.isna() & s.notna()
    looks_like_date = remaining & s.str.contains(r"\d") & s.str.contains(r"[-/]")
    if looks_like_date.any():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`."
            )
            out.loc[looks_like_date] = pd.to_datetime(s[looks_like_date], errors="coerce")

    return out

# =============================== 1) EXTRACT ===================================
MONTH_RE = r"(0[1-9]|1[0-2])"; YEAR_RE = r"(20\d{2})"
INNER_PATTERNS = [
    re.compile(rf"nh_archive_{MONTH_RE}_{YEAR_RE}\.zip", re.I),
    re.compile(rf"nh_archive_{YEAR_RE}_{MONTH_RE}\.zip", re.I),
    re.compile(rf"nursing_homes_including_rehab_services_archive_{MONTH_RE}_{YEAR_RE}\.zip", re.I),
    re.compile(rf"(?:^|[_-]){MONTH_RE}[_-]{YEAR_RE}\.zip$", re.I),
    re.compile(rf"(?:^|[_-]){YEAR_RE}[_-]{MONTH_RE}\.zip$", re.I),
]

def is_ownership_basename(name: str) -> bool:
    b = Path(name).name.strip().lower()
    if not re.search(r"(ownership|owner)", b):
        return False
    return bool(re.search(r"\.(csv|txt|tsv)$", b))

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

def std_name(mm: int, yyyy: int):
    return (f"ownership_{yyyy:04d}_{mm:02d}.csv"
            if NAME_STYLE == "yyyy_mm" else
            f"ownership_{mm:02d}_{yyyy:04d}.csv")

def write_overwrite(path: Path, data: bytes):
    """Always overwrite target file."""
    path.write_bytes(data)
    return path

def extract_by_filename_only():
    yearlies = sorted(p for p in NH_ZIP_DIR.glob("nh_archive_*.zip") if p.is_file())
    if not yearlies:
        raise FileNotFoundError(f"No yearly zips found in {NH_ZIP_DIR}")

    extracted, skipped = 0, 0
    notes = []
    for yearly in yearlies:
        with zipfile.ZipFile(yearly, "r") as yz:
            inner_zips = [n for n in yz.namelist() if n.lower().endswith(".zip")]
            for inner in inner_zips:
                mm, yyyy = parse_mm_yyyy_from_inner(Path(inner).name)
                if not (mm and yyyy):
                    skipped += 1
                    notes.append((yearly.name, inner, "no_mm_yyyy_in_inner_zip_name"))
                    continue
                with yz.open(inner) as inner_bytes:
                    with zipfile.ZipFile(BytesIO(inner_bytes.read()), "r") as mz:
                        names = mz.namelist()
                        candidates = [n for n in names if is_ownership_basename(n)]
                        if not candidates:
                            skipped += 1
                            preview = ", ".join(Path(n).name for n in names[:8])
                            notes.append((yearly.name, inner, f"no_ownership_like_name; sample: {preview}"))
                            continue

                        def sort_key(n):
                            bn = Path(n).name.strip().lower()
                            size = mz.getinfo(n).file_size
                            return (0 if "download" in bn else (1 if "display" in bn else 2),
                                    -len(bn), -size, bn)
                        candidates.sort(key=sort_key)
                        target = candidates[0]

                        out_name = std_name(mm, yyyy)
                        out_path = OWN_DIR / out_name
                        print(f"[{yyyy}-{str(mm).zfill(2)}] {Path(inner).name} → {Path(target).name}  ⇒  {out_path.name}")
                        if not DRY_RUN:
                            data = mz.read(target)
                            write_overwrite(out_path, data)
                        extracted += 1

    print(f"\n[extract] extracted={extracted}, skipped={skipped}")
    if notes:
        print("\n[notes] first 25 skip reasons:")
        for yzip, inner, reason in notes[:25]:
            print(f"  - {yzip} :: {inner} → {reason}")
        if len(notes) > 25:
            print(f"  ... and {len(notes)-25} more")

# =================== 2) STANDARDIZE IN-PLACE (7 cols + roles) =================
CANON_MAP = {
    # CCN
    "provnum": "cms_certification_number",
    "federal provider number": "cms_certification_number",
    "cms certification number (ccn)": "cms_certification_number",
    "cms certification number": "cms_certification_number",
    "provider id": "cms_certification_number",
    # Provider (not kept now, but map anyway)
    "provider name": "provider_name",
    "provname": "provider_name",
    # Role
    "role": "role",
    "role desc": "role",
    "role_desc": "role",
    "role played by owner or manager in facility": "role",
    "role played by owner in facility": "role",
    "role of owner or manager": "role",
    # Ownership %
    "ownership percentage": "ownership_percentage",
    "owner percentage": "ownership_percentage",
    "pct ownership": "ownership_percentage",
    "percent ownership": "ownership_percentage",
    # Owner
    "owner name": "owner_name",
    "ownership name": "owner_name",
    "owner": "owner_name",
    "owner type": "owner_type",
    "type of owner": "owner_type",
    "ownership type": "owner_type",
    # Dates
    "processing date": "processing_date",
    "process date": "processing_date",
    "processingdate": "processing_date",
    "processdate": "processing_date",
    "filedate": "processing_date",
    "association date": "association_date",
    "assoc date": "association_date",
}

KEEP_COLS = [
    "cms_certification_number","role","owner_type","owner_name",
    "ownership_percentage","association_date","processing_date"
]

ROLE_KEEP_MAP = {
    "DIRECT":      r"5%\s*OR\s*GREATER\s+DIRECT\s+OWNERSHIP\s+INTEREST",
    "INDIRECT":    r"5%\s*OR\s*GREATER\s+INDIRECT\s+OWNERSHIP\s+INTEREST",
    "PARTNERSHIP": r"\bPARTNERSHIP\s+INTEREST\b",
}

RE_MM_YYYY = re.compile(r"ownership_(0[1-9]|1[0-2])_(20\d{2})\.csv$", re.I)
RE_YYYY_MM = re.compile(r"ownership_(20\d{2})_(0[1-9]|1[0-2])\.csv$", re.I)

def parse_ym_from_fname(name: str):
    m = RE_MM_YYYY.search(name)
    if m: return int(m.group(2)), int(m.group(1))
    m = RE_YYYY_MM.search(name)
    if m: return int(m.group(1)), int(m.group(2))
    return None, None

def normalize_month_df(df: pd.DataFrame, fname: str) -> pd.DataFrame:
    # 1) rename to canonical
    ren = {c: CANON_MAP.get(norm_header(c), c) for c in df.columns}
    df = df.rename(columns=ren)

    # 2) role filter → DIRECT / INDIRECT / PARTNERSHIP
    role_raw = df.get("role")
    if role_raw is None:
        role_raw = pd.Series(pd.NA, index=df.index)
    role_up = role_raw.fillna("").astype(str).str.upper()

    role_out = pd.Series(pd.NA, index=df.index, dtype="object")
    for canon, pat in ROLE_KEEP_MAP.items():
        role_out = role_out.mask(role_up.str.contains(pat, regex=True, na=False)==True, canon)

    mask_keep = role_out.isin(list(ROLE_KEEP_MAP.keys()))
    df = df.loc[mask_keep].copy()
    df["role"] = role_out.loc[mask_keep].values

    # 3) ownership % → numeric 0..100 (auto-scale 0..1 → 0..100)
    if "ownership_percentage" in df.columns:
        pct = (df["ownership_percentage"].astype(str)
               .str.replace("%","",regex=False)
               .str.replace(",","",regex=False)
               .str.strip())
        pct = pct.mask(pct.eq("") | pct.str.contains("NO PERCENTAGE", case=False))
        val = pd.to_numeric(pct, errors="coerce")
        if val.dropna().between(0,1).mean() > 0.85 and val.dropna().between(0,100).mean() < 0.9:
            val = val * 100.0
        df["ownership_percentage"] = val

    # 4) CCN → 6-digit string
    if "cms_certification_number" in df.columns:
        df["cms_certification_number"] = (
            df["cms_certification_number"].astype(str)
              .str.replace(r"\D","", regex=True).str.zfill(6)
              .mask(lambda s: ~s.str.fullmatch(r"\d{6}"))
        )

    # 5) dates → datetime64
    if "processing_date" in df.columns:
        df["processing_date"] = safe_to_datetime(df["processing_date"])
    if "association_date" in df.columns:
        df["association_date"] = safe_to_datetime(df["association_date"])

    # 6) synth processing_date from filename if missing anywhere
    y, m = parse_ym_from_fname(fname)
    if y and m:
        synth = pd.Timestamp(year=y, month=m, day=1)
        if "processing_date" in df.columns:
            df.loc[df["processing_date"].isna(), "processing_date"] = synth
        else:
            df["processing_date"] = synth

    # 6b) enforce association_date never null: fill from processing_date
    if "association_date" in df.columns:
        df["association_date"] = df["association_date"].fillna(df["processing_date"])
    else:
        df["association_date"] = df["processing_date"]

    # 6c) FINALIZE DATE COLUMNS AS STRINGS (YYYY-MM-DD) to avoid ns integers anywhere
    for dc in ["association_date", "processing_date"]:
        if dc in df.columns:
            if not np.issubdtype(df[dc].dtype, np.datetime64):
                df[dc] = safe_to_datetime(df[dc])
            df[dc] = df[dc].dt.strftime("%Y-%m-%d")

    # 7) keep only requested columns (create if missing)
    for col in KEEP_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[KEEP_COLS]

    # 8) drop exact duplicates on the keep set (within file)
    df = df.drop_duplicates(KEEP_COLS)

    return df

def atomic_overwrite_csv(fp: Path, df: pd.DataFrame):
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_fp = tmp_dir / ("~" + fp.name)
    df.to_csv(tmp_fp, index=False, date_format="%Y-%m-%d")
    shutil.move(str(tmp_fp), str(fp))
    shutil.rmtree(tmp_dir, ignore_errors=True)

def standardize_in_place_and_combine():
    files = sorted(p for p in OWN_DIR.glob("ownership_*.csv"))
    if not files:
        raise FileNotFoundError(f"No ownership_*.csv files found in {OWN_DIR}")

    # In-place rewrite (overwrite file contents)
    rewritten = 0
    for fp in files:
        try:
            raw = read_csv_any(fp)
            clean = normalize_month_df(raw, fp.name)
            atomic_overwrite_csv(fp, clean)
            rewritten += 1
            print(f"[rewrite] {fp.name}  rows={len(clean)}")
        except Exception as e:
            print(f"[warn] {fp.name} failed: {e}")

    print(f"\n[standardize] Rewrote {rewritten} file(s).")

    if DO_COMBINE:
        frames = []
        files = sorted(p for p in OWN_DIR.glob("ownership_*.csv"))  # refresh list
        for fp in files:
            try:
                frames.append(pd.read_csv(fp, dtype=str, low_memory=False))
            except Exception as e:
                print(f"[warn] combine read failed {fp.name}: {e}")

        if frames:
            combined = pd.concat(frames, ignore_index=True)

            # ---- EXACT DEDUP ACROSS ALL FILES (before save) ----
            # Light canonicalization to avoid whitespace/float format dupes
            for c in ["role","owner_type","owner_name"]:
                combined[c] = combined[c].astype(str).str.strip()
            # ownership_percentage was written as numeric earlier; ensure consistent rounding-as-string
            combined["ownership_percentage"] = pd.to_numeric(
                combined["ownership_percentage"], errors="coerce"
            ).round(2)

            # Drop exact duplicates on canonical keep set
            before = len(combined)
            combined = combined.drop_duplicates(
                subset=["cms_certification_number","role","owner_type","owner_name",
                        "ownership_percentage","association_date","processing_date"],
                keep="first"
            ).reset_index(drop=True)
            removed = before - len(combined)
            print(f"[dedup-combined] removed {removed:,} exact duplicate row(s) across all months")

            # sort by CCN as string (six-digit with NaNs last)
            if "cms_certification_number" in combined.columns:
                key = combined["cms_certification_number"].fillna("~")
                combined = combined.iloc[key.argsort(kind="mergesort")].reset_index(drop=True)

            # Write combined as strings; dates are already "YYYY-MM-DD"
            combined.to_csv(COMBINED_CSV, index=False)
            print(f"[save] combined CSV (sorted by CCN) → {COMBINED_CSV}  ({len(combined):,} rows)")

# =============================== RUN PIPELINE =================================
# 1) Extract monthly ownership CSVs (overwrites if present)
extract_by_filename_only()

# 2) Standardize in place + Combine (with global exact-dedup)
if DO_STANDARDIZE:
    standardize_in_place_and_combine()


# In[ ]:




