#!/usr/bin/env python
# coding: utf-8
# -----------------------------------------------------------------------------
# chow.py — Minimal CHOW exports, with explicit universes
#
# Outputs:
#   1) data/interim/chow_nh_compare.csv  (universe = CCNs in ownership.csv)
#   2) data/interim/chow_mcr.csv         (universe = CCNs in MCR files)
#   3) data/interim/chow.csv             (inner join of the two universes)
#
# Ownership CHOW criteria: UNCHANGED from your logic
#   - turnover >= 50% (percent-overlap when available; else names-based fallback)
#   - surname override: if >=80% surname-family control persists → NOT a CHOW
#   - window: to_start >= 2017-01-01
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, re, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# ============================== Paths / Config ================================
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").is_dir() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

RAW_DIR     = Path(os.getenv("NH_DATA_DIR", PROJECT_ROOT / "data" / "raw")).resolve()
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# Inputs
OWNERSHIP_FP = INTERIM_DIR / "ownership.csv"  # produced by your ownership pipeline
MCR_DIR      = RAW_DIR / "medicare-cost-reports"
MCR_PATTERNS = ["mcr_flatfile_20??.csv", "mcr_flatfile_20??.sas7bdat", "mcr_flatfile_20??.xpt", "mcr_flatfile_20??.XPT"]

# Outputs
OWN_OUT   = INTERIM_DIR / "chow_nh_compare.csv"
MCR_OUT   = INTERIM_DIR / "chow_mcr.csv"
MERGE_OUT = INTERIM_DIR / "chow.csv"

# Window & thresholds (match your existing logic)
CUTOFF_DATE = pd.Timestamp("2017-01-01")
TURNOVER_THRESH = 0.50  # 50% or greater → CHOW
SURNAME_MIN_FRACTION_KEEP = 0.80
USE_SURNAME_OVERRIDE = True
LEVEL_PRIORITY = ["indirect", "direct", "partnership"]  # prefer indirect snapshots

# ============================== Common helpers ================================
ORG_MARKERS_RE = re.compile(r"\b(LLC|INC|CORP|CORPORATION|L\.L\.C\.|L\.P\.|LP|LLP|PLC|COMPANY|CO\.?|HOLDINGS?|GROUP|TRUST|FUND|CAPITAL|PARTNERS(hip)?|HEALTH|CARE|AUTHORITY|HOSPITAL|CENTER|NURSING|HOME|OPERATING|MANAGEMENT)\b", re.I)
TOKEN_RE = re.compile(r"[^\w\s]")
SUFFIXES = r'\b(INC|INCORPORATED|CORP|CORPORATION|LLC|L\.L\.C\.|L\.P\.|LP|LLP|PLC|CO|COMPANY|HOLDINGS?|PARTNERS?|PARTNERSHIP|CAPITAL|INVESTMENTS?|TRUST|GROUP)\b'

def normalize_ccn_any(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("").str.strip().str.upper()
    s = s.str.replace(r"[ \-\/\.]", "", regex=True)
    is_digits = s.str.fullmatch(r"\d+")
    s = s.mask(is_digits, s.str.zfill(6))
    s = s.replace({"": pd.NA})
    return s

def clean_owner_name(s: str) -> str:
    if pd.isna(s) or not str(s).strip():
        return ""
    x = str(s).upper()
    x = re.sub(r"[.,&/()\-']", " ", x)
    x = re.sub(SUFFIXES, "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def level_bucket(role_val: str) -> str:
    s = str(role_val).lower()
    if "indirect" in s:  return "indirect"
    if "direct"   in s:  return "direct"
    if "partner"  in s:  return "partnership"
    return ""

def normalize_weights_allow_missing(df_block: pd.DataFrame) -> pd.DataFrame:
    g = df_block.copy()
    g["ownership_percentage"] = pd.to_numeric(g["ownership_percentage"], errors="coerce")
    agg = g.groupby("owner_name_norm", as_index=False)["ownership_percentage"].sum(min_count=1)
    has_num = agg["ownership_percentage"].notna().any()
    tot = float(agg["ownership_percentage"].fillna(0).sum())
    if has_num and tot > 0:
        vec = agg[agg["ownership_percentage"].notna()].copy()
        vec["ownership_percentage"] = vec["ownership_percentage"] * (100.0 / tot)
    else:
        owners = agg["owner_name_norm"].tolist()
        if not owners:
            return pd.DataFrame(columns=["owner_name_norm","ownership_percentage"])
        equal = 100.0 / len(owners)
        vec = pd.DataFrame({"owner_name_norm": owners, "ownership_percentage": [equal]*len(owners)})
    vec["ownership_percentage"] = vec["ownership_percentage"].round(1)
    tot2 = float(vec["ownership_percentage"].sum())
    if tot2 > 0:
        vec["ownership_percentage"] = (vec["ownership_percentage"] * (100.0 / tot2)).round(1)
    return vec[vec["ownership_percentage"] > 0].sort_values(["ownership_percentage","owner_name_norm"], ascending=[False, True]).reset_index(drop=True)

def pct_overlap(prev_map: dict, curr_map: dict) -> float:
    names = set(prev_map) | set(curr_map)
    overlap = 0.0
    for n in names:
        overlap += min(prev_map.get(n, 0.0), curr_map.get(n, 0.0))
    return max(0.0, min(overlap / 100.0, 1.0))

def parse_list(j):
    try:
        if pd.isna(j): return []
        out = json.loads(j)
        return out if isinstance(out, list) else []
    except Exception:
        return []

def weight_map(names_list, pcts_list):
    wm = defaultdict(float)
    for n, p in zip(names_list, pcts_list):
        try:
            f = float(p)
        except Exception:
            continue
        if pd.isna(f):
            continue
        wm[str(n)] += f
    return dict(wm)

def jaccard_names(prev_names, curr_names):
    a, b = set(prev_names), set(curr_names)
    if not a and not b: return np.nan
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union

def looks_like_person(name: str) -> bool:
    if not name or ORG_MARKERS_RE.search(name):
        return False
    toks = TOKEN_RE.sub(" ", str(name)).split()
    toks = [t for t in toks if t]
    return 1 <= len(toks) <= 3

def surname_of(name: str) -> str:
    toks = TOKEN_RE.sub(" ", str(name)).split()
    toks = [t for t in toks if t]
    return toks[-1].upper() if toks else ""

def surname_weight_map(wm: dict) -> dict:
    agg = defaultdict(float)
    for n, p in wm.items():
        if looks_like_person(n):
            s = surname_of(n)
            if s:
                agg[s] += p
            else:
                agg["_PERSON_"] += p
        else:
            agg["_ORG_"] += p
    return dict(agg)

def surname_family_overlap(prev_wm: dict, curr_wm: dict) -> float:
    ps = surname_weight_map(prev_wm)
    cs = surname_weight_map(curr_wm)
    owners = set(ps) | set(cs)
    overlap = sum(min(ps.get(k, 0.0), cs.get(k, 0.0)) for k in owners)
    denom   = max(sum(ps.values()), sum(cs.values()), 100.0)
    return max(0.0, min(100.0 * overlap / denom, 100.0))

def month_floor(d: pd.Timestamp | pd.Series) -> pd.Timestamp | pd.Series:
    return pd.to_datetime(d, errors="coerce").dt.to_period("M").dt.to_timestamp()

def pivot_dates_wide(df: pd.DataFrame, id_col: str, date_col: str, prefix: str, order_col: str | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[id_col])
    g = df.copy()
    if order_col:
        g["_rank"] = g.groupby(id_col)[order_col].rank(method="first").astype(int)
    else:
        g["_rank"] = g.groupby(id_col)[date_col].rank(method="first").astype(int)
    g = g[[id_col, date_col, "_rank"]].drop_duplicates().sort_values([id_col, "_rank"])
    wide = g.pivot(index=id_col, columns="_rank", values=date_col).reset_index()
    if wide.shape[1] > 1:
        wide.columns = [id_col] + [f"{prefix}_chow_{k}_date" for k in wide.columns[1:]]
    else:
        wide.columns = [id_col]
    for c in wide.columns:
        if c.endswith("_date"):
            wide[c] = pd.to_datetime(wide[c], errors="coerce").dt.strftime("%Y-%m-%d")
    return wide

# ============================== MCR helpers/reader ============================
def read_mcr_three_cols() -> pd.DataFrame:
    files = []
    seen = set()
    for pat in MCR_PATTERNS:
        for f in sorted(MCR_DIR.glob(pat)):
            if f not in seen:
                files.append(f); seen.add(f)
    if not files:
        raise FileNotFoundError(f"No MCR files matched in {MCR_DIR}")

    try:
        import pyreadstat
        HAS_PYREADSTAT = True
    except Exception:
        HAS_PYREADSTAT = False

    def _sniff_csv(fp: Path):
        seps = [",","|","\t",";","~"]; encs=["utf-8","utf-8-sig","cp1252","latin1"]
        for enc in encs:
            for sep in seps:
                try:
                    hdr = pd.read_csv(fp, sep=sep, nrows=0, engine="python", encoding=enc)
                    if hdr.shape[1] > 0:
                        return sep, enc
                except Exception:
                    pass
        return ",", "utf-8"

    def _select_three(df: pd.DataFrame) -> pd.DataFrame:
        up = {c: c.upper().strip() for c in df.columns}
        rev = {v:k for k,v in up.items()}
        out = pd.DataFrame()
        for t in ["PRVDR_NUM","S2_2_CHOW","S2_2_CHOWDATE"]:
            src = rev.get(t)
            if src is not None:
                out[t] = df[src]
            else:
                out[t] = pd.NA
        return out

    frames = []
    for fp in files:
        suffix = fp.suffix.lower()
        try:
            if suffix == ".csv":
                sep, enc = _sniff_csv(fp)
                df = pd.read_csv(fp, sep=sep, encoding=enc, dtype=str, low_memory=False, engine=("c" if sep=="," else "python"))
                frames.append(_select_three(df))
            elif suffix in {".sas7bdat",".xpt"}:
                if HAS_PYREADSTAT:
                    try:
                        if suffix == ".sas7bdat":
                            _, meta = pyreadstat.read_sas7bdat(fp, metadataonly=True)
                        else:
                            _, meta = pyreadstat.read_xport(fp, metadataonly=True)
                        avail = {c.lower().strip(): c for c in meta.column_names}
                        def find_one(cands):
                            for c in cands:
                                if c in avail: return avail[c]
                            return None
                        m = {
                            "PRVDR_NUM":     find_one(["prvdr_num","provider_number","provnum","prvdrnum"]),
                            "S2_2_CHOW":     find_one(["s2_2_chow","s22_chow","chow","s2_2_chow_cd","s2_2_chow_flag","s2_2_chow_ind"]),
                            "S2_2_CHOWDATE": find_one(["s2_2_chowdate","s2_2_chow_date","s22_chow_date","chow_date"]),
                        }
                        usecols = [c for c in m.values() if c]
                        if not usecols: raise RuntimeError("no mapped cols")
                        if suffix == ".sas7bdat":
                            df, _ = pyreadstat.read_sas7bdat(fp, usecols=usecols)
                        else:
                            df, _ = pyreadstat.read_xport(fp, usecols=usecols)
                        inv = {v:k for k,v in m.items() if v}
                        df = df.rename(columns=inv)
                        frames.append(_select_three(df))
                    except Exception:
                        fmt = "sas7bdat" if suffix==".sas7bdat" else "xport"
                        df = pd.read_sas(fp, format=fmt, encoding="latin1")
                        frames.append(_select_three(df))
                else:
                    fmt = "sas7bdat" if suffix==".sas7bdat" else "xport"
                    df = pd.read_sas(fp, format=fmt, encoding="latin1")
                    frames.append(_select_three(df))
            else:
                print(f"[skip] {fp.name}")
        except Exception as e:
            print(f"[warn] {fp.name}: {e}")

    if not frames:
        return pd.DataFrame(columns=["PRVDR_NUM","S2_2_CHOW","S2_2_CHOWDATE"])
    out = pd.concat(frames, ignore_index=True)
    return out

def coerce_maybe_sas_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    num = pd.to_numeric(s, errors="coerce")
    has_num = num.notna().sum()
    if has_num and has_num >= max(1, int(0.5 * s.notna().sum())):
        base = pd.Timestamp("1960-01-01")
        return (base + pd.to_timedelta(num, unit="D")).astype("datetime64[ns]")
    return pd.to_datetime(s.astype("string").str.strip(), errors="coerce")

# ============================== OWNERSHIP CHOW ================================
def build_chow_from_ownership() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not OWNERSHIP_FP.exists():
        raise FileNotFoundError(f"{OWNERSHIP_FP} not found (run ownership pipeline first).")

    own = pd.read_csv(OWNERSHIP_FP, low_memory=False)
    need = {"cms_certification_number","role","owner_name","ownership_percentage","association_date"}
    miss = need - set(own.columns)
    if miss:
        raise ValueError(f"ownership.csv missing {miss}")

    own["cms_certification_number"] = normalize_ccn_any(own["cms_certification_number"])
    own_ccns = own["cms_certification_number"].dropna().drop_duplicates().to_frame(name="cms_certification_number")

    own["association_date"] = pd.to_datetime(own["association_date"], errors="coerce")
    own = own.dropna(subset=["association_date"]).copy()

    own["owner_name_norm"] = own["owner_name"].map(clean_owner_name)
    own["level"] = own["role"].map(level_bucket)

    # Build snapshots at each (CCN, association_date), prefer Indirect→Direct→Partnership
    rows = []
    for (ccn, adate), g in own.groupby(["cms_certification_number","association_date"], sort=True):
        chosen = None
        for lvl in LEVEL_PRIORITY:
            gl = g[g["level"] == lvl]
            if len(gl):
                chosen = (lvl, gl); break
        if chosen is None:
            continue
        lvl, gl = chosen
        vec = normalize_weights_allow_missing(gl[["owner_name_norm","ownership_percentage"]])
        if vec.empty:
            owners = gl["owner_name_norm"].dropna().unique().tolist()
            if not owners: 
                continue
            equal = 100.0 / len(owners)
            vec = pd.DataFrame({"owner_name_norm": owners, "ownership_percentage": [equal]*len(owners)})
        wm = dict(zip(vec["owner_name_norm"], vec["ownership_percentage"].astype(float)))
        rows.append({"cms_certification_number": ccn, "association_date": adate, "source_level": lvl, "weights": wm})

    snaps = pd.DataFrame(rows).sort_values(["cms_certification_number","association_date"]).reset_index(drop=True)
    if snaps.empty:
        # still return a shell so the base-universe join works
        return pd.DataFrame(columns=["cms_certification_number","n_chow_nh_compare"]), own_ccns

    # Compute transitions & CHOW flags
    trans_rows = []
    for ccn, g in snaps.groupby("cms_certification_number", sort=True):
        g = g.sort_values("association_date").reset_index(drop=True)
        if g.empty or len(g) == 1:
            continue
        prev_w = g.loc[0, "weights"]
        for i in range(1, len(g)):
            curr_w = g.loc[i, "weights"]
            to_start = g.loc[i, "association_date"]
            ov = pct_overlap(prev_w, curr_w)   # 0..1
            turnover = 1.0 - ov if ov is not None else None
            method = 0  # 0=percent overlap, 1=names fallback
            if turnover is None or np.isnan(turnover):
                prev_names = list(prev_w.keys()); curr_names = list(curr_w.keys())
                j = jaccard_names(prev_names, curr_names)
                turnover = None if pd.isna(j) else (1.0 - float(j))
                method = 1
            surname_keep_pct = surname_family_overlap(prev_w, curr_w)  # [0,100]
            surname_override = (USE_SURNAME_OVERRIDE and pd.notna(turnover)
                                and surname_keep_pct >= 100.0 * SURNAME_MIN_FRACTION_KEEP)
            is_in_window = pd.notna(to_start) and (to_start >= CUTOFF_DATE)
            is_chow = bool(is_in_window and (turnover is not None) and (turnover >= TURNOVER_THRESH) and (not surname_override))
            trans_rows.append({
                "cms_certification_number": ccn,
                "to_start": to_start,
                "turnover": None if turnover is None else float(turnover),
                "method": method,  # 0=percent, 1=names
                "surname_keep_pct": float(surname_keep_pct) if pd.notna(surname_keep_pct) else np.nan,
                "surname_override": bool(surname_override),
                "is_chow": is_chow
            })
            prev_w = curr_w

    trans = pd.DataFrame(trans_rows)
    if trans.empty:
        return pd.DataFrame(columns=["cms_certification_number","n_chow_nh_compare"]), own_ccns

    ch = trans[trans["is_chow"] == True].copy()
    ch["event_month"] = pd.to_datetime(ch["to_start"]).dt.to_period("M").dt.to_timestamp()

    meth = (ch.assign(method_name=lambda d: d["method"].map({0:"percent",1:"names"}))
              .groupby(["cms_certification_number","method_name"], as_index=False)
              .size().pivot(index="cms_certification_number", columns="method_name", values="size")
              .fillna(0).astype(int).reset_index())
    if "percent" not in meth.columns: meth["percent"] = 0
    if "names"   not in meth.columns: meth["names"]   = 0
    meth = meth.rename(columns={"percent":"n_chow_nh_compare_percent","names":"n_chow_nh_compare_names"})

    wide = pivot_dates_wide(ch[["cms_certification_number","event_month"]],
                            "cms_certification_number", "event_month", "nh_compare")
    n = ch.groupby("cms_certification_number", as_index=False).size().rename(columns={"size":"n_chow_nh_compare"})

    out = (n.merge(wide, on="cms_certification_number", how="left")
             .merge(meth, on="cms_certification_number", how="left"))
    out["n_chow_nh_compare"] = out["n_chow_nh_compare"].astype("Int16")
    for c in ["n_chow_nh_compare_percent","n_chow_nh_compare_names"]:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype("Int16")
    out = out.sort_values("cms_certification_number").reset_index(drop=True)
    return out, own_ccns

# ============================== MCR CHOW ======================================
def build_chow_from_mcr() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_mcr_three_cols()
    if raw.empty:
        return pd.DataFrame(columns=["cms_certification_number","n_chow_mcr"]), pd.DataFrame(columns=["cms_certification_number"])
    df = raw.rename(columns={c: c.upper().strip() for c in raw.columns})
    df["PRVDR_NUM"] = normalize_ccn_any(df["PRVDR_NUM"])
    mcr_ccns = df["PRVDR_NUM"].dropna().drop_duplicates().to_frame(name="cms_certification_number")

    df["S2_2_CHOWDATE"] = coerce_maybe_sas_date(df["S2_2_CHOWDATE"])
    events = (df.loc[df["S2_2_CHOWDATE"].notna(), ["PRVDR_NUM","S2_2_CHOWDATE"]]
                .drop_duplicates()
                .rename(columns={"PRVDR_NUM":"cms_certification_number"}))
    if events.empty:
        return pd.DataFrame(columns=["cms_certification_number","n_chow_mcr"]), mcr_ccns

    events["event_month"] = month_floor(events["S2_2_CHOWDATE"])
    events = events.loc[events["event_month"] >= CUTOFF_DATE].copy()
    if events.empty:
        return pd.DataFrame(columns=["cms_certification_number","n_chow_mcr"]), mcr_ccns

    n = events.groupby("cms_certification_number", as_index=False).size().rename(columns={"size":"n_chow_mcr"}).astype({"n_chow_mcr":"Int16"})
    wide = pivot_dates_wide(events[["cms_certification_number","event_month"]],
                            "cms_certification_number", "event_month", "mcr")
    out = n.merge(wide, on="cms_certification_number", how="left").sort_values("cms_certification_number").reset_index(drop=True)
    return out, mcr_ccns

# ============================== MERGE & SAVE ==================================
def main():
    # Build OWN and MCR tables + capture each source universe
    own_wide, own_ccns = build_chow_from_ownership()
    mcr_wide, mcr_ccns = build_chow_from_mcr()

    # --- Export OWNERSHIP with OWNERSHIP universe base (includes zeros) ---
    own_full = own_ccns.merge(own_wide, on="cms_certification_number", how="left")
    if "n_chow_nh_compare" not in own_full.columns:
        own_full["n_chow_nh_compare"] = 0
    own_full["n_chow_nh_compare"] = pd.to_numeric(own_full["n_chow_nh_compare"], errors="coerce").fillna(0).astype("Int16")
    for c in ["n_chow_nh_compare_percent","n_chow_nh_compare_names"]:
        if c in own_full.columns:
            own_full[c] = pd.to_numeric(own_full[c], errors="coerce").fillna(0).astype("Int16")
    for c in [c for c in own_full.columns if c.endswith("_date")]:
        own_full[c] = own_full[c].astype("string")
    own_full = own_full.sort_values("cms_certification_number").reset_index(drop=True)
    own_full.to_csv(OWN_OUT, index=False)
    print(f"[saved] ownership CHOWs → {OWN_OUT}  (rows={len(own_full):,})")

    # --- Export MCR with MCR universe base (includes zeros) ---
    mcr_full = mcr_ccns.merge(mcr_wide, on="cms_certification_number", how="left")
    if "n_chow_mcr" not in mcr_full.columns:
        mcr_full["n_chow_mcr"] = 0
    mcr_full["n_chow_mcr"] = pd.to_numeric(mcr_full["n_chow_mcr"], errors="coerce").fillna(0).astype("Int16")
    for c in [c for c in mcr_full.columns if c.endswith("_date")]:
        mcr_full[c] = mcr_full[c].astype("string")
    mcr_full = mcr_full.sort_values("cms_certification_number").reset_index(drop=True)
    mcr_full.to_csv(MCR_OUT, index=False)
    print(f"[saved] MCR CHOWs → {MCR_OUT}  (rows={len(mcr_full):,})")

    # --- MERGED = intersection of universes (inner join) ---
    merged = own_full.merge(mcr_full, on="cms_certification_number", how="inner")
    for c in ["n_chow_nh_compare","n_chow_mcr"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype("Int16")
    merged["n_chow_any"] = ((merged["n_chow_nh_compare"] > 0) | (merged["n_chow_mcr"] > 0)).astype("Int8")
    merged = merged.sort_values("cms_certification_number").reset_index(drop=True)
    merged.to_csv(MERGE_OUT, index=False)
    print(f"[saved] merged CHOWs  → {MERGE_OUT}  (rows={len(merged):,})")

if __name__ == "__main__":
    main()