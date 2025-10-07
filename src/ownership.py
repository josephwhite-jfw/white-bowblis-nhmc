# nhmc/ownership.py
# Extract ownership files by name → standardize → combine → drop in-hospital months

from __future__ import annotations
import re, csv, zipfile
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd

from .paths import RAW_DIR, OWN_DIR, PROV_DIR, HOSP_PANEL, OWN_COMBINED
from .utils import normalize_ccn_any, to_monthstart, drop_hospital_months

NH_ZIP_DIR = RAW_DIR / "nh-compare"
OWN_DIR.mkdir(parents=True, exist_ok=True)

MONTH_RE = r"(0[1-9]|1[0-2])"; YEAR_RE = r"(20\d{2})"
INNER_PATTERNS = [
    re.compile(rf"nh_archive_{MONTH_RE}_{YEAR_RE}\.zip", re.I),
    re.compile(rf"nh_archive_{YEAR_RE}_{MONTH_RE}\.zip", re.I),
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

def _is_ownership_name(name: str) -> bool:
    b = Path(name).name.lower()
    return ("owner" in b or "ownership" in b) and b.endswith((".csv",".tsv",".txt"))

def _sniff_delim(raw: bytes):
    sample = raw[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample.decode("utf-8","ignore"))
        return dialect.delimiter
    except Exception:
        return "\t" if sample.count(b"\t") > sample.count(b",") else ","

def _read_any(raw: bytes) -> pd.DataFrame:
    sep = _sniff_delim(raw)
    for enc in ("utf-8","utf-8-sig","cp1252","latin1"):
        try:
            return pd.read_csv(BytesIO(raw), dtype=str, sep=sep, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(BytesIO(raw), dtype=str, sep=sep, encoding="utf-8", encoding_errors="replace", low_memory=False)

CANON = {
    "provnum":"cms_certification_number",
    "federal provider number":"cms_certification_number",
    "cms certification number (ccn)":"cms_certification_number",
    "cms certification number":"cms_certification_number",
    "provider id":"cms_certification_number",
    "provider name":"provider_name",
    "provname":"provider_name",
    "role":"role",
    "role desc":"role","role_desc":"role","role played by owner or manager in facility":"role",
    "ownership percentage":"ownership_percentage","owner percentage":"ownership_percentage",
    "pct ownership":"ownership_percentage","percent ownership":"ownership_percentage",
    "owner name":"owner_name","ownership name":"owner_name","owner":"owner_name",
    "owner type":"owner_type","type of owner":"owner_type","ownership type":"owner_type",
    "processing date":"processing_date","process date":"processing_date","processdate":"processing_date","processingdate":"processing_date","filedate":"processing_date",
    "association date":"association_date","assoc date":"association_date",
}
KEEP = ["cms_certification_number","role","owner_type","owner_name","ownership_percentage","association_date","processing_date"]

ROLE_KEEP = {
    "DIRECT":      r"5%\s*OR\s*GREATER\s+DIRECT",
    "INDIRECT":    r"5%\s*OR\s*GREATER\s+INDIRECT",
    "PARTNERSHIP": r"\bPARTNERSHIP\s+INTEREST\b",
}

def _norm_header(h: str) -> str:
    return re.sub(r"\s+"," ", str(h or "").strip().lower().replace("_"," ")).strip()

def _standardize_month(df: pd.DataFrame, fname: str) -> pd.DataFrame:
    # rename
    ren = {c: CANON.get(_norm_header(c), c) for c in df.columns}
    df = df.rename(columns=ren)

    # role filter
    role_raw = df.get("role", pd.Series(pd.NA, index=df.index)).astype(str).str.upper()
    role_out = pd.Series(pd.NA, index=df.index, dtype="object")
    for lab, pat in ROLE_KEEP.items():
        role_out = role_out.mask(role_raw.str.contains(pat, regex=True, na=False), lab)
    df = df.loc[role_out.isin(list(ROLE_KEEP.keys()))].copy()
    df["role"] = role_out.loc[df.index].values

    # ccns
    df["cms_certification_number"] = normalize_ccn_any(df.get("cms_certification_number"))

    # ownership percentage → numeric (auto-scale)
    if "ownership_percentage" in df.columns:
        pct = (df["ownership_percentage"].astype(str)
               .str.replace("%","",regex=False)
               .str.replace(",","",regex=False).str.strip())
        pct = pct.mask(pct.eq("") | pct.str.contains("NO PERCENTAGE", case=False))
        val = pd.to_numeric(pct, errors="coerce")
        # detect 0..1
        if val.dropna().between(0,1).mean() > 0.6 and val.dropna().between(0,100).mean() < 0.6:
            val = val * 100.0
        df["ownership_percentage"] = val

    # dates
    for c in ["association_date","processing_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    # fill processing_date from filename if missing (YYYY_MM in name)
    m = re.search(r"(20\d{2})[_-](0[1-9]|1[0-2])", fname)
    if m:
        synth = pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
        if "processing_date" in df.columns:
            df.loc[df["processing_date"].isna(), "processing_date"] = synth
        else:
            df["processing_date"] = synth
    if "association_date" in df.columns:
        df["association_date"] = df["association_date"].fillna(df["processing_date"])
    else:
        df["association_date"] = df["processing_date"]

    # to str for stability
    for c in ["association_date","processing_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d")

    # keep / dropna CCN
    for col in KEEP:
        if col not in df.columns: df[col] = pd.NA
    df = df[KEEP].copy()
    df = df.dropna(subset=["cms_certification_number"]).drop_duplicates(KEEP)
    return df

def build_ownership_combined() -> None:
    yearlies = sorted(p for p in NH_ZIP_DIR.glob("nh_archive_*.zip") if p.is_file())
    if not yearlies:
        raise FileNotFoundError(f"No yearly zips in {NH_ZIP_DIR}")
    frames = []

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
                            cand = [n for n in mz.namelist() if _is_ownership_name(n)]
                            if not cand: 
                                continue
                            # prefer "*download*.csv"
                            cand.sort(key=lambda n: (0 if "download" in n.lower() else (1 if "display" in n.lower() else 2), -len(n)))
                            raw = mz.read(cand[0])
                            df = _read_any(raw)
                            std = _standardize_month(df, f"ownership_{yyyy:04d}_{mm:02d}.csv")
                            frames.append(std)
                    except zipfile.BadZipFile:
                        continue

    if not frames:
        raise RuntimeError("No ownership files standardized.")
    combined = pd.concat(frames, ignore_index=True)

    # helper month for filtering exact in-hospital months if panel exists
    combined["date"] = to_monthstart(combined["processing_date"])
    combined["cms_certification_number"] = normalize_ccn_any(combined["cms_certification_number"])
    combined = drop_hospital_months(combined, ccn_col="cms_certification_number", month_col="date")
    combined = combined.drop(columns=["date"]).drop_duplicates().reset_index(drop=True)

    combined.to_csv(OWN_COMBINED, index=False)
    print(f"[ownership] combined → {OWN_COMBINED} (rows={len(combined):,})")
