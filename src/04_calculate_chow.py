#!/usr/bin/env python
# coding: utf-8
# ─────────────────────────────────────────────────────────────────────────────
# Facility Signatures + CHOW Engine (single script)
#   * Reads ownership_combined.csv (already hospital-filtered upstream)
#   * Alphanumeric-safe CCN normalization (pad only if numeric)
#   * Builds facility signatures (long + wide QC preview)
#   * Computes CHOW transitions + summary (with surname override)
#   * Outputs to data/interim/
# Run with: %run 04_build_facility_signatures_and_chow.py
# ─────────────────────────────────────────────────────────────────────────────

import os, re, json, pathlib, warnings
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------- Config / Paths -----------------------------------
ENV_DIR      = os.getenv("NH_DATA_DIR", r"C:\Users\Owner\OneDrive\NursingHomeData")
OWN_DIR      = pathlib.Path(ENV_DIR) / "ownership-files"
INPUT_FP     = OWN_DIR / "ownership_combined.csv"   # already hospital-filtered upstream

PROJECT_ROOT = pathlib.Path.cwd()
while not (PROJECT_ROOT / "data").is_dir() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "data").is_dir():
    PROJECT_ROOT = OWN_DIR.parent

INTERIM_DIR  = PROJECT_ROOT / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

OUT_LONG     = INTERIM_DIR / "facility_signatures_long.csv"
OUT_WIDE     = INTERIM_DIR / "facility_signatures_wide_preview.csv"
OUT_TRANS    = INTERIM_DIR / "ccn_group_transitions.csv"
OUT_SUM      = INTERIM_DIR / "ccn_chow_summary.csv"
OUT_LITE     = INTERIM_DIR / "ccn_chow_lite.csv"

# -------------------------- Tunables -----------------------------------------
LEVEL_PRIORITY   = ["indirect", "direct", "partnership"]
ROUND_PCT        = 1
TURNOVER_THRESH  = 0.50           # overlap-based turnover ≥ 50% → new group
WIDE_MAX_GROUPS  = 8

# CHOW config
THRESH_CHOW      = 50.0           # percent turnover to count as CHOW
CUTOFF_DATE      = pd.Timestamp("2017-01-01")
BUCKET_LABELS    = {
    0: "<50",
    1: "90–100",
    2: "80–90",
    3: "70–80",
    4: "60–70",
    5: "50–60",
    6: "inconclusive"
}
ORG_MARKERS_RE   = re.compile(r"\b(LLC|INC|CORP|CORPORATION|L\.L\.C\.|L\.P\.|LP|LLP|PLC|COMPANY|CO\.?|HOLDINGS?|GROUP|TRUST|FUND|CAPITAL|PARTNERS(hip)?|HEALTH|CARE|AUTHORITY|HOSPITAL|CENTER|NURSING|HOME|OPERATING|MANAGEMENT)\b", re.I)
TOKEN_RE         = re.compile(r"[^\w\s]")
SURNAME_MIN_FRACTION_KEEP = 0.80   # ≥80% surname family keeps control → override CHOW
USE_SURNAME_OVERRIDE       = True

# -------------------------- Helpers ------------------------------------------
SUFFIXES = r'\b(INC|INCORPORATED|CORP|CORPORATION|LLC|L\.L\.C\.|L\.P\.|LP|LLP|PLC|CO|COMPANY|HOLDINGS?|PARTNERS?|PARTNERSHIP|CAPITAL|INVESTMENTS?|TRUST|GROUP)\b'

def normalize_ccn_any(series: pd.Series) -> pd.Series:
    """
    Preserve alphanumeric CCNs; strip separators; pad only if numeric.
      - uppercase, remove spaces/-/./
      - if ALL digits → zfill(6)
      - if has letters → keep as-is
    """
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

    agg_num = g.groupby("owner_name_norm", as_index=False)["ownership_percentage"].sum(min_count=1)
    has_numeric   = agg_num["ownership_percentage"].notna().any()
    total_numeric = agg_num["ownership_percentage"].fillna(0).sum()

    if has_numeric and total_numeric > 0:
        vec = agg_num[agg_num["ownership_percentage"].notna()].copy()
        vec["ownership_percentage"] = vec["ownership_percentage"] * (100.0 / total_numeric)
    else:
        owners = agg_num["owner_name_norm"].tolist()
        if not owners:
            return pd.DataFrame(columns=["owner_name_norm","ownership_percentage"])
        equal = 100.0 / len(owners)
        vec = pd.DataFrame({"owner_name_norm": owners,
                            "ownership_percentage": [equal]*len(owners)})

    vec["ownership_percentage"] = vec["ownership_percentage"].round(ROUND_PCT)
    tot2 = vec["ownership_percentage"].sum()
    if tot2 > 0:
        vec["ownership_percentage"] = (vec["ownership_percentage"] * (100.0 / tot2)).round(ROUND_PCT)

    vec = vec[vec["ownership_percentage"] > 0].copy()
    if vec.empty:
        owners = agg_num["owner_name_norm"].tolist()
        equal = 100.0 / len(owners)
        vec = pd.DataFrame({"owner_name_norm": owners,
                            "ownership_percentage": [round(equal, ROUND_PCT)]*len(owners)})
    return vec.sort_values(["ownership_percentage","owner_name_norm"], ascending=[False, True]).reset_index(drop=True)

def pct_overlap(prev_map: dict, curr_map: dict) -> float:
    names = set(prev_map) | set(curr_map)
    overlap = 0.0
    for n in names:
        overlap += min(prev_map.get(n, 0.0), curr_map.get(n, 0.0))
    return max(0.0, min(overlap / 100.0, 1.0))

# CHOW helpers
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

def pct_overlap100(prev_map, curr_map):
    if not prev_map and not curr_map:
        return np.nan
    owners = set(prev_map) | set(curr_map)
    overlap = sum(min(prev_map.get(o, 0.0), curr_map.get(o, 0.0)) for o in owners)
    denom   = max(sum(prev_map.values()), sum(curr_map.values()), 100.0)
    return max(0.0, min(100.0 * overlap / denom, 100.0))

def jaccard_names(prev_names, curr_names):
    a, b = set(prev_names), set(curr_names)
    if not a and not b:
        return np.nan
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

def bucket_code(turnover_pct: float) -> int:
    if pd.isna(turnover_pct): return 6
    t = float(turnover_pct)
    if t < 50:  return 0
    if t >= 90: return 1
    if t >= 80: return 2
    if t >= 70: return 3
    if t >= 60: return 4
    return 5  # 50–60

# -------------------------- Facility Signatures -------------------------------
print("[load]", INPUT_FP)
df = pd.read_csv(INPUT_FP, low_memory=False)

needed = {"cms_certification_number", "role", "owner_name", "ownership_percentage", "association_date"}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Normalize CCN & dates (alphanumeric-safe; no dropping by format)
df["cms_certification_number"] = normalize_ccn_any(df["cms_certification_number"])
df["association_date"] = pd.to_datetime(df["association_date"], errors="coerce")

# Keep rows w/ valid date (do not require ownership %)
df = df.dropna(subset=["association_date"]).copy()

# Normalize owner names and levels
df["owner_name_norm"] = df["owner_name"].map(clean_owner_name)
df["level"] = df["role"].map(level_bucket)

# --- Build snapshots at each (CCN, association_date), preferring Indirect→Direct→Partnership ---
snapshots = []
for (ccn, adate), g in df.groupby(["cms_certification_number","association_date"], sort=True):
    chosen = None
    for lvl in LEVEL_PRIORITY:
        gl = g[g["level"] == lvl]
        if len(gl):
            chosen = (lvl, gl)
            break
    if chosen is None:
        continue

    lvl, gl = chosen
    vec = normalize_weights_allow_missing(gl[["owner_name_norm","ownership_percentage"]].copy())
    if vec.empty:
        owners = gl["owner_name_norm"].dropna().unique().tolist()
        if not owners:
            continue
        equal = 100.0 / len(owners)
        vec = pd.DataFrame({"owner_name_norm": owners,
                            "ownership_percentage": [round(equal, ROUND_PCT)]*len(owners)})

    weight_map_snapshot = dict(zip(vec["owner_name_norm"], vec["ownership_percentage"].astype(float)))
    snapshots.append({
        "cms_certification_number": ccn,
        "association_date": adate,
        "source_level": lvl,
        "weights": weight_map_snapshot
    })

snapshots_df = pd.DataFrame(snapshots).sort_values(["cms_certification_number","association_date"]).reset_index(drop=True)
print("[snapshots built] rows:", len(snapshots_df))

# --- Group into stable regimes using turnover threshold ---
long_rows = []
for ccn, g in snapshots_df.groupby("cms_certification_number", sort=True):
    g = g.sort_values("association_date").reset_index(drop=True)
    if g.empty: 
        continue

    group_n     = 1
    group_start = g.loc[0, "association_date"]
    group_level = g.loc[0, "source_level"]
    prev_weights = g.loc[0, "weights"]

    def hhi_from_map(wm: dict) -> float:
        return round(sum((p/100.0)**2 for p in wm.values()), 4)

    long_rows.append({
        "cms_certification_number": ccn,
        "group_n": group_n,
        "start": group_start,
        "end": pd.NaT,
        "source_level": group_level,
        "names_list": json.dumps(list(prev_weights.keys()), separators=(",", ":")),
        "pcts_list": json.dumps(list(prev_weights.values()), separators=(",", ":")),
        "owner_count": len(prev_weights),
        "hhi": hhi_from_map(prev_weights),
    })

    for i in range(1, len(g)):
        curr_weights = g.loc[i, "weights"]
        ov = pct_overlap(prev_weights, curr_weights)     # 0..1
        turnover = 1.0 - ov

        if turnover >= TURNOVER_THRESH:
            long_rows[-1]["end"] = g.loc[i-1, "association_date"]
            group_n += 1
            group_start = g.loc[i, "association_date"]
            group_level = g.loc[i, "source_level"]
            long_rows.append({
                "cms_certification_number": ccn,
                "group_n": group_n,
                "start": group_start,
                "end": pd.NaT,
                "source_level": group_level,
                "names_list": json.dumps(list(curr_weights.keys()), separators=(",", ":")),
                "pcts_list": json.dumps(list(curr_weights.values()), separators=(",", ":")),
                "owner_count": len(curr_weights),
                "hhi": hhi_from_map(curr_weights),
            })
            prev_weights = curr_weights
        else:
            prev_weights = curr_weights

    long_rows[-1]["end"] = g.loc[len(g)-1, "association_date"]

long_df = pd.DataFrame(long_rows).sort_values(["cms_certification_number","group_n"]).reset_index(drop=True)

# --- Wide QC preview ---
def as_label(names_json, pcts_json, k=12):
    names = json.loads(names_json)
    pcts  = json.loads(pcts_json)
    pairs = [f"{n} ({round(p, ROUND_PCT)}%)" for n, p in zip(names, pcts)]
    return "; ".join(pairs[:k])

if not long_df.empty:
    wide_blocks = []
    for ccn, g in long_df.groupby("cms_certification_number"):
        g = g.sort_values("group_n")
        row = {"cms_certification_number": ccn}
        for _, r in g.head(WIDE_MAX_GROUPS).iterrows():
            n = int(r["group_n"])
            row[f"group{n}_start"] = pd.to_datetime(r["start"]).date()
            row[f"group{n}_end"]   = pd.to_datetime(r["end"]).date()
            row[f"group{n}_level"] = r["source_level"]
            row[f"group{n}_names"] = as_label(r["names_list"], r["pcts_list"])
            row[f"group{n}_pcts"]  = ",".join(map(lambda x: str(int(round(x,0))), json.loads(r["pcts_list"])))
        wide_blocks.append(row)
    wide_df = pd.DataFrame(wide_blocks).sort_values("cms_certification_number").reset_index(drop=True)
else:
    wide_df = pd.DataFrame(columns=["cms_certification_number"])

# Save signatures
long_df.to_csv(OUT_LONG, index=False)
wide_df.to_csv(OUT_WIDE, index=False)
print(f"[save] signatures long → {OUT_LONG}  rows={len(long_df):,}")
print(f"[save] signatures wide → {OUT_WIDE}  rows={len(wide_df):,}")

# -------------------------- CHOW Engine --------------------------------------
print("[INPUT ]", OUT_LONG)
print("[OUTPUT]", OUT_TRANS)
print("[OUTPUT]", OUT_SUM)

long = long_df.copy()
# Parse dates & lists
long["start"] = pd.to_datetime(long["start"], errors="coerce")
long["end"]   = pd.to_datetime(long["end"],   errors="coerce")
long["names"] = long["names_list"].apply(parse_list)
long["pcts"]  = long["pcts_list"].apply(parse_list)

# Defensive: ensure lists align
bad_align = (long["names"].str.len() != long["pcts"].str.len())
if int(bad_align.sum()) > 0:
    print(f"[warn] rows with names/pcts length mismatch: {int(bad_align.sum())}")

# Build transitions per CCN
rows = []
for ccn, g in long.groupby("cms_certification_number", sort=True):
    g = g.sort_values("start").reset_index(drop=True)
    if len(g) < 2:
        continue
    for i in range(1, len(g)):
        prev = g.loc[i-1]
        curr = g.loc[i]
        from_start = prev["start"]
        to_start   = curr["start"]

        prev_w = weight_map(prev["names"], prev["pcts"])
        curr_w = weight_map(curr["names"], curr["pcts"])

        ov_pct    = pct_overlap100(prev_w, curr_w)                 # [0,100]
        turn_pct  = None if pd.isna(ov_pct) else (100.0 - ov_pct)
        method    = 0  # 0 = percent-based, 1 = names-based

        if pd.isna(turn_pct):
            jacc = jaccard_names(prev["names"], curr["names"])
            turn_pct = None if pd.isna(jacc) else (100.0 * (1.0 - jacc))
            method = 1

        inconclusive = pd.isna(turn_pct)
        surname_keep_pct = surname_family_overlap(prev_w, curr_w)  # [0,100]
        surname_override = False
        if USE_SURNAME_OVERRIDE and not pd.isna(turn_pct):
            if surname_keep_pct >= 100.0 * SURNAME_MIN_FRACTION_KEEP:
                surname_override = True

        bcode = bucket_code(turn_pct)
        is_in_window = (pd.notna(to_start) and to_start >= CUTOFF_DATE)
        is_chow = bool(is_in_window and (not inconclusive) and (turn_pct >= THRESH_CHOW) and (not surname_override))

        rows.append({
            "cms_certification_number": ccn,
            "from_group": int(prev["group_n"]),
            "to_group":   int(curr["group_n"]),
            "from_start": from_start,
            "from_end":   prev["end"],
            "to_start":   to_start,
            "to_end":     curr["end"],
            "from_level": prev["source_level"],
            "to_level":   curr["source_level"],
            "turnover_pct": None if pd.isna(turn_pct) else round(float(turn_pct), 1),
            "overlap_pct":  None if pd.isna(ov_pct)   else round(float(ov_pct), 1),
            "method":       method,              # 0=percent, 1=names
            "bucket_code":  bcode,
            "bucket_label": BUCKET_LABELS[bcode],
            "surname_keep_pct": round(float(surname_keep_pct), 1) if not pd.isna(surname_keep_pct) else np.nan,
            "surname_override": surname_override,
            "inconclusive": inconclusive,
            "is_chow": is_chow
        })

trans = pd.DataFrame(rows).sort_values(
    ["cms_certification_number","to_start","to_group"]
).reset_index(drop=True)
trans.to_csv(OUT_TRANS, index=False)
print(f"[save] transitions → {OUT_TRANS}  rows={len(trans):,}")

# Universe of CCNs (even those with 0 transitions)
all_ccns = (
    long[["cms_certification_number","start"]]
    .groupby("cms_certification_number", as_index=False)
    .agg(first_seen=("start","min"))
)

def summarize_ccn_safe(ccn, df_ccn, max_events=12):
    out = {
        "cms_certification_number": ccn,
        "num_chows": 0,
        "first_seen_month": pd.NaT,
        "present_at_start": False,
        "entered_after_start": False,
    }
    for k in range(0,7):
        out[f"bucket_{k}_count"] = 0

    if df_ccn is None or df_ccn.empty:
        return out

    df_ccn = df_ccn.sort_values("to_start")
    bc = df_ccn["bucket_code"].value_counts().to_dict()
    for k in range(0,7):
        out[f"bucket_{k}_count"] = int(bc.get(k, 0))

    chow = df_ccn[df_ccn["is_chow"]].copy()
    out["num_chows"] = int(chow.shape[0])

    for i, (_, r) in enumerate(chow.head(max_events).iterrows(), start=1):
        out[f"chow_date_{i}"]      = r["to_start"]
        out[f"chow_magnitude_{i}"] = r.get("turnover_pct", np.nan)
        out[f"chow_method_{i}"]    = r.get("method", np.nan)
        out[f"chow_inconcl_{i}"]   = bool(r.get("inconclusive", False))

    return out

trans_by_ccn = {k: v.copy() for k, v in trans.groupby("cms_certification_number")} if not trans.empty else {}
summary_rows = []
for _, row in all_ccns.iterrows():
    ccn = row["cms_certification_number"]
    s   = summarize_ccn_safe(ccn, trans_by_ccn.get(ccn))
    first_seen = long.loc[long["cms_certification_number"]==ccn, "start"].min()
    s["first_seen_month"]  = first_seen.to_period("M").to_timestamp("M") if pd.notna(first_seen) else pd.NaT
    s["present_at_start"]  = bool(pd.notna(first_seen) and (first_seen <  CUTOFF_DATE))
    s["entered_after_start"] = bool(pd.notna(first_seen) and (first_seen >= CUTOFF_DATE))
    summary_rows.append(s)

summary = (
    pd.DataFrame(summary_rows)
      .sort_values("cms_certification_number")
      .reset_index(drop=True)
)
summary.to_csv(OUT_SUM, index=False)
print(f"[save] summary(all CCNs) → {OUT_SUM}  rows={len(summary):,}")

# LITE export
chow_date_cols = [c for c in summary.columns if c.startswith("chow_date_")]
lite = summary[["cms_certification_number","num_chows"] + chow_date_cols].copy()
lite["is_chow"] = (lite["num_chows"] > 0).astype(int)
lite = lite[["cms_certification_number","num_chows","is_chow"] + chow_date_cols]
lite.to_csv(OUT_LITE, index=False)
print(f"[save] lite → {OUT_LITE}  rows={len(lite):,}  cols={len(lite.columns)}")

# Quick diagnostics
print("\n=== Quick diagnostics ===")
print("[diag] CCNs in long:", long['cms_certification_number'].nunique())
print("[diag] CCNs in trans:", trans['cms_certification_number'].nunique() if not trans.empty else 0)
print("[diag] CCNs in summary:", summary['cms_certification_number'].nunique())
if not trans.empty:
    print("Transitions total:", len(trans))
    print("CHOWs total     :", int(trans['is_chow'].sum()))
    print("Share overridden by surname rule:",
          f"{100.0*trans.loc[trans['surname_override'],'is_chow'].count()/max(1,len(trans)):.2f}% (of all transitions)")
    print("\nCHOWs by year:")
    print(trans[trans["is_chow"]].assign(year=pd.to_datetime(trans["to_start"]).dt.year)
              .groupby("year").size().rename("count"))
print("\nBucket distribution (all transitions):")
print(trans["bucket_label"].value_counts() if not trans.empty else "n/a")