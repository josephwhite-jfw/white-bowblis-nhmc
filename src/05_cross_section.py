#!/usr/bin/env python
# coding: utf-8
# ─────────────────────────────────────────────────────────────────────────────
# Medicare Cost Reports (MCR) CHOW builder + agreement vs. Ownership CHOW-lite
#   * Reads RAW_DIR/medicare-cost-reports/mcr_flatfile_20??.csv (3 cols only)
#   * CCN normalization: alphanumeric-safe (pad only if purely digits)
#   * Removes in-hospital CCNs via provider_resides_in_hospital_by_ccn.csv
#   * Flags MCR CHOWs **only** if CHOW date >= 2017-01-01 (OWN-aligned window)
#   * Outputs:
#       - data/interim/mcr_chow_events_long.csv                (all dated events, audit)
#       - data/interim/mcr_chow_provider_events_all.csv        (per-CCN CHOW counts+dates, POST-2017 only)
#       - data/interim/mcr_chow_provider_events_with_windows.csv (diagnostics: pre/post lists per CCN)
#       - data/interim/chow_agreement_tables.xlsx              (Overview, buckets, Crosstab, Discrepancies)
#       - data/interim/chow_facility_comparison_plus.csv       (rich per-CCN comparison)
#       - data/interim/chow_facility_comparison_1v1.csv        (subset: exactly 1 vs 1)
#       - data/interim/chow_facility_mismatches.csv            (subset: disagreements)
#   * Compares against: data/interim/ccn_chow_lite.csv         (your ownership CHOW-lite)
# Run with: %run 05_build_mcr_chow_and_compare.py
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import os, warnings, re
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- Paths ----------------
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").is_dir() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

RAW_DIR = Path(os.getenv("NH_DATA_DIR", PROJECT_ROOT / "data" / "raw")).resolve()
MCR_DIR = RAW_DIR / "medicare-cost-reports"
MCR_GLOB = "mcr_flatfile_20??.csv"

PROV_DIR = RAW_DIR / "provider-info-files"
HOSP_BY_CCN = PROV_DIR / "provider_resides_in_hospital_by_ccn.csv"  # CCN-level flag

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

OUT_EVENTS_LONG   = INTERIM_DIR / "mcr_chow_events_long.csv"                  # all dated events (audit)
OUT_PROVIDER_ALL  = INTERIM_DIR / "mcr_chow_provider_events_all.csv"          # POST-2017 only (flagged)
OUT_PROVIDER_DIAG = INTERIM_DIR / "mcr_chow_provider_events_with_windows.csv" # diagnostics: pre/post lists
OUT_XLSX          = INTERIM_DIR / "chow_agreement_tables.xlsx"

# Rich comparison outputs
PLUS_FP    = INTERIM_DIR / "chow_facility_comparison_plus.csv"
ONEVONE_FP = INTERIM_DIR / "chow_facility_comparison_1v1.csv"
MIS_FP     = INTERIM_DIR / "chow_facility_mismatches.csv"

OWN_LITE = INTERIM_DIR / "ccn_chow_lite.csv"  # produced by your signatures+CHOW script

# Align to OWN window
CUTOFF_DATE = pd.Timestamp("2017-01-01")

print(f"[paths] RAW_DIR={RAW_DIR}")
print(f"[paths] MCR_DIR={MCR_DIR}")
print(f"[paths] PROV_DIR={PROV_DIR}")
print(f"[paths] INTERIM_DIR={INTERIM_DIR}")

# ---------------- CCN normalization (alphanumeric-safe) ----------------
def normalize_ccn_any(series: pd.Series) -> pd.Series:
    """
    Preserve alphanumeric CCNs; strip separators; pad only if numeric.
      - uppercase, remove spaces / - / / / .
      - if ALL digits → zfill(6)
      - if has letters → keep as-is
    """
    s = series.astype("string").fillna("").str.strip().str.upper()
    s = s.str.replace(r"[ \-\/\.]", "", regex=True)
    is_digits = s.str.fullmatch(r"\d+")
    s = s.mask(is_digits, s.str.zfill(6))
    s = s.replace({"": pd.NA})
    return s

# ------------- Reader (robust & simple) -------------
_TRY_SEPS = [",", "|", "\t", ";", "~"]
_TRY_ENCODINGS = ["utf-8","utf-8-sig","cp1252","latin1"]
TARGET_UP = {"PRVDR_NUM","S2_2_CHOW","S2_2_CHOWDATE"}

def _sniff_sep_enc(fp: Path):
    last_err = None
    for enc in _TRY_ENCODINGS:
        for sep in _TRY_SEPS:
            try:
                hdr = pd.read_csv(fp, sep=sep, nrows=0, engine="python", encoding=enc)
                if hdr.shape[1] > 0:
                    return sep, enc
            except Exception as e:
                last_err = e
    raise last_err or RuntimeError(f"Could not sniff {fp}")

def _usecols_ci(colname: str) -> bool:
    return str(colname).upper().strip() in TARGET_UP

def _read_three_raw(fp: Path) -> pd.DataFrame:
    sep, enc = _sniff_sep_enc(fp)
    engine = None if sep == "," else "python"  # C engine for comma csv
    df = pd.read_csv(fp, sep=sep, encoding=enc, engine=engine,
                     usecols=_usecols_ci, dtype=str)
    print(f"[read] {fp.name} sep='{sep}' enc={enc} -> cols={list(df.columns)} rows={len(df):,}")
    return df

# ------------- Load & Stack (raw) -------------
files = sorted(MCR_DIR.glob(MCR_GLOB))
if not files:
    raise FileNotFoundError(f"No files matched {MCR_DIR / MCR_GLOB}")

frames = []
for fp in files:
    try:
        frames.append(_read_three_raw(fp))
    except Exception as e:
        print(f"[warn] {fp.name}: {e}")

mcr_raw = pd.concat(frames, ignore_index=True)
print(f"[stack] combined rows={len(mcr_raw):,}")
print("[stack] non-null counts:\n", mcr_raw.notna().sum())

# ------------- Normalize -------------
mcr = mcr_raw.rename(columns={c: c.upper().strip() for c in mcr_raw.columns})
mcr["PRVDR_NUM"] = normalize_ccn_any(mcr["PRVDR_NUM"])
mcr["S2_2_CHOWDATE"] = pd.to_datetime(
    mcr["S2_2_CHOWDATE"].astype("string").str.strip(),
    errors="coerce"
)

# ------------- Remove in-hospital CCNs (by CCN) -------------
if HOSP_BY_CCN.exists():
    hosp = pd.read_csv(HOSP_BY_CCN, dtype=str, low_memory=False)
    hosp.columns = [c.strip().lower() for c in hosp.columns]
    ccn_col = next((c for c in ["cms_certification_number","ccn","provnum","prvdr_num"] if c in hosp.columns), None)
    if ccn_col is None:
        raise ValueError(f"{HOSP_BY_CCN} missing CCN column.")
    if "provider_resides_in_hospital" not in hosp.columns:
        raise ValueError(f"{HOSP_BY_CCN} missing 'provider_resides_in_hospital' column.")

    def parse_bool(x):
        s = str(x).strip().lower()
        if s in {"1","true","t","y","yes"}:  return True
        if s in {"0","false","f","n","no"}:  return False
        return pd.NA

    hosp["provider_resides_in_hospital"] = hosp["provider_resides_in_hospital"].map(parse_bool)
    hosp["cms_certification_number"] = normalize_ccn_any(hosp[ccn_col])

    drop_ccns = set(hosp.loc[hosp["provider_resides_in_hospital"] == True, "cms_certification_number"].dropna().unique())
    before = mcr["PRVDR_NUM"].nunique()
    mcr = mcr[~mcr["PRVDR_NUM"].isin(drop_ccns)].copy()
    after  = mcr["PRVDR_NUM"].nunique()
    print(f"[hospital filter] CCNs: {before:,} -> {after:,} (removed {before-after:,})")
else:
    print(f"[hospital filter] WARNING: {HOSP_BY_CCN} not found — no CCNs filtered")

# ---------- Provider universe (after hospital filter) ----------
all_providers = (
    mcr["PRVDR_NUM"].dropna().astype("string").drop_duplicates()
    .to_frame(name="cms_certification_number")
)

# ---------- Long events table (all dated CHOWs; audit) ----------
events_all = (
    mcr.loc[mcr["S2_2_CHOWDATE"].notna(), ["PRVDR_NUM", "S2_2_CHOWDATE"]]
       .drop_duplicates()
       .sort_values(["PRVDR_NUM", "S2_2_CHOWDATE"], kind="mergesort")
       .reset_index(drop=True)
       .rename(columns={"PRVDR_NUM": "cms_certification_number"})
)
if not events_all.empty:
    events_all["event_month"] = pd.to_datetime(events_all["S2_2_CHOWDATE"]).dt.to_period("M").dt.to_timestamp()

# Save all events (no cutoff) for audit
events_all.to_csv(OUT_EVENTS_LONG, index=False)
print(f"[saved] events-long -> {OUT_EVENTS_LONG}  rows={len(events_all):,}  providers={events_all['cms_certification_number'].nunique() if not events_all.empty else 0:,}")

# ---------- Enforce OWN-aligned window for MCR CHOWs ----------
# Keep events that occur on/after 2017-01-01 for counting/flagging
events_post = events_all.loc[events_all["event_month"] >= CUTOFF_DATE].copy() if not events_all.empty else events_all

# ---------- Wide per-provider table (POST-2017 only) ----------
if events_post.empty:
    wide = pd.DataFrame(columns=["cms_certification_number"])
    counts = pd.DataFrame(columns=["cms_certification_number","n_chow"])
else:
    events_post["chow_order"] = events_post.groupby("cms_certification_number")["event_month"].rank(method="first").astype(int)
    wide = (
        events_post.pivot(index="cms_certification_number", columns="chow_order", values="event_month")
                  .sort_index()
                  .reset_index()
    )
    if wide.shape[1] > 1:
        wide.columns = ["cms_certification_number"] + [f"chow_{k}_date" for k in wide.columns[1:]]
    else:
        wide.columns = ["cms_certification_number"]
    counts = events_post.groupby("cms_certification_number", as_index=False).size().rename(columns={"size":"n_chow"})

provider_wide = all_providers.merge(wide, on="cms_certification_number", how="left")
provider_wide = provider_wide.merge(counts, on="cms_certification_number", how="left")
provider_wide["n_chow"] = provider_wide["n_chow"].fillna(0).astype("Int16")
provider_wide["is_chow"] = (provider_wide["n_chow"] > 0).astype("Int8")

# Convert date columns to ISO for CSV readability
date_cols = [c for c in provider_wide.columns if c.startswith("chow_") and c.endswith("_date")]
for c in date_cols:
    provider_wide[c] = pd.to_datetime(provider_wide[c], errors="coerce").dt.strftime("%Y-%m-%d")

# Order columns nicely
ordered = ["cms_certification_number", "n_chow", "is_chow"] + sorted(
    date_cols,
    key=lambda x: int(re.search(r"chow_(\d+)_date", x).group(1)) if re.search(r"chow_(\d+)_date", x) else 0
)
provider_wide = provider_wide[ordered].sort_values("cms_certification_number").reset_index(drop=True)

print(f"[result] providers total={len(provider_wide):,}  with POST-2017 CHOWs={int((provider_wide['n_chow']>0).sum()):,}  max_n_chow={int(provider_wide['n_chow'].max() if not provider_wide.empty else 0)}")
print(provider_wide.head(10))

# ---------- Save provider-wide (POST-2017 aligned) ----------
provider_wide.to_csv(OUT_PROVIDER_ALL, index=False)
print(f"[saved] provider-wide (POST-2017 ONLY) -> {OUT_PROVIDER_ALL}")

# ---------- Optional diagnostics: pre/post collapsed lists per CCN ----------
def collapse_months(df):
    if df is None or df.empty:
        return pd.Series(dtype="string")
    return (df.groupby("cms_certification_number")["event_month"]
              .apply(lambda s: "|".join(pd.to_datetime(sorted(set(s))).strftime("%Y-%m").tolist()))
              .astype("string"))

pre  = events_all.loc[events_all["event_month"] <  CUTOFF_DATE].copy() if not events_all.empty else events_all
post = events_post

diag = all_providers.copy()
diag["pre_dates"]  = collapse_months(pre)
diag["post_dates"] = collapse_months(post)
diag["n_pre"]  = diag["pre_dates"].fillna("").str.count(r"\|").add((diag["pre_dates"].fillna("")!="").astype(int))
diag["n_post"] = diag["post_dates"].fillna("").str.count(r"\|").add((diag["post_dates"].fillna("")!="").astype(int))
diag["n_pre"]  = diag["n_pre"].fillna(0).astype("Int16")
diag["n_post"] = diag["n_post"].fillna(0).astype("Int16")

diag_out = all_providers.merge(provider_wide[["cms_certification_number","n_chow"]], on="cms_certification_number", how="left") \
                        .merge(diag, on="cms_certification_number", how="left") \
                        .rename(columns={"n_chow":"n_chow_post"})
diag_out.to_csv(OUT_PROVIDER_DIAG, index=False)
print(f"[saved] provider-wide with windows (diagnostics) -> {OUT_PROVIDER_DIAG}")

# ---------- Agreement vs. Ownership CHOW-lite ----------
if not OWN_LITE.exists():
    print(f"[warn] {OWN_LITE} not found — skipping agreement/crosstab build.")
else:
    own = pd.read_csv(OWN_LITE, dtype={"cms_certification_number":"string"})
    mcrw = provider_wide.copy()

    if "num_chows" not in own.columns:
        raise KeyError("Ownership CHOW-lite missing 'num_chows'.")
    if "n_chow" not in mcrw.columns:
        raise KeyError("MCR wide missing 'n_chow' (internal bug).")

    lite_comp = own[["cms_certification_number", "num_chows"]].rename(columns={"num_chows": "num_chows_lite"})
    mcr_comp  = mcrw[["cms_certification_number", "n_chow"]].rename(columns={"n_chow": "num_chows_mcr"})

    merged = lite_comp.merge(mcr_comp, on="cms_certification_number", how="outer")

    merged["is_chow_lite"] = merged["num_chows_lite"].fillna(0) > 0
    merged["is_chow_mcr"]  = merged["num_chows_mcr"].fillna(0) > 0

    crosstab = pd.crosstab(merged["is_chow_lite"], merged["is_chow_mcr"],
                           rownames=["Lite (ours)"], colnames=["MCR"])

    # 0/1/2+ categories for buckets
    def to_cat(n):
        try:
            n = int(n)
        except Exception:
            return "0"
        if n <= 0: return "0"
        if n == 1: return "1"
        return "2+"

    merged["own_cat"] = merged["num_chows_lite"].map(to_cat)
    merged["mcr_cat"] = merged["num_chows_mcr"].map(to_cat)

    overlap = merged.dropna(subset=["num_chows_lite","num_chows_mcr"], how="any").copy()

    ctab = pd.crosstab(overlap["own_cat"], overlap["mcr_cat"]).reindex(
        index=["0","1","2+"], columns=["0","1","2+"], fill_value=0
    ).reset_index().rename(columns={"own_cat":"Ownership n_chow"})
    ctab["Total"] = ctab[["0","1","2+"]].sum(axis=1)

    own_match_00 = ((overlap["own_cat"]=="0")  & (overlap["mcr_cat"]=="0")).sum()
    own_match_11 = ((overlap["own_cat"]=="1")  & (overlap["mcr_cat"]=="1")).sum()
    own_match_2p = ((overlap["own_cat"]=="2+") & (overlap["mcr_cat"]=="2+")).sum()
    own_total    = len(overlap)
    own_disc     = own_total - (own_match_00 + own_match_11 + own_match_2p)
    own_bucket = pd.DataFrame({
        "Bucket": [
            "OWN=0 & MCR=0",
            "OWN=1 & MCR=1",
            "OWN=2+ & MCR=2+",
            "Discrepancies (OWN base)",
            "TOTAL (OWN base overlap)"
        ],
        "Count": [int(own_match_00), int(own_match_11), int(own_match_2p), int(own_disc), int(own_total)]
    })
    own_bucket["Share_of_Overlap_%"] = (own_bucket["Count"] / max(1, own_total) * 100).round(2)

    mcr_match_00 = ((overlap["mcr_cat"]=="0")  & (overlap["own_cat"]=="0")).sum()
    mcr_match_11 = ((overlap["mcr_cat"]=="1")  & (overlap["own_cat"]=="1")).sum()
    mcr_match_2p = ((overlap["mcr_cat"]=="2+") & (overlap["own_cat"]=="2+")).sum()
    mcr_total    = len(overlap)
    mcr_disc     = mcr_total - (mcr_match_00 + mcr_match_11 + mcr_match_2p)
    mcr_bucket = pd.DataFrame({
        "Bucket": [
            "MCR=0 & OWN=0",
            "MCR=1 & OWN=1",
            "MCR=2+ & OWN=2+",
            "Discrepancies (MCR base)",
            "TOTAL (MCR base overlap)"
        ],
        "Count": [int(mcr_match_00), int(mcr_match_11), int(mcr_match_2p), int(mcr_disc), int(mcr_total)]
    })
    mcr_bucket["Share_of_Overlap_%"] = (mcr_bucket["Count"] / max(1, mcr_total) * 100).round(2)

    discrepancies = overlap.loc[overlap["own_cat"] != overlap["mcr_cat"], [
        "cms_certification_number","num_chows_lite","num_chows_mcr","own_cat","mcr_cat"
    ]].sort_values("cms_certification_number").reset_index(drop=True)

    overview = pd.DataFrame({
        "Metric": [
            "MCR providers (after hospital filter)",
            "Ownership providers (from CHOW-lite)",
            "Overlap (inner join on CCN for bucket tables)"
        ],
        "Count": [len(mcrw), len(own), len(overlap)]
    })

    print("\n=== Crosstab of CHOW presence (binary, all CCNs) ===")
    print(crosstab)
    print("\n=== OWN-based buckets (overlap base) ===")
    print(own_bucket.to_string(index=False))
    print("\n=== MCR-based buckets (overlap base) ===")
    print(mcr_bucket.to_string(index=False))
    print("\n=== Crosstab 0/1/2+ (Ownership rows × MCR cols; overlap only) ===")
    print(ctab.to_string(index=False))
    print(f"\n[discrepancies] rows = {len(discrepancies):,}")

    # -------- Excel export (with simple autofit) --------
    def compute_col_widths(df, extra=2, min_w=8, max_w=60):
        widths = []
        for col in df.columns:
            series = df[col].astype(str)
            max_len = max([len(col)] + series.map(len).tolist()) + extra
            widths.append(max(min_w, min(max_w, max_len)))
        return widths

    try:
        import xlsxwriter
        engine = "xlsxwriter"
    except Exception:
        engine = "openpyxl"

    with pd.ExcelWriter(OUT_XLSX, engine=engine) as xw:
        overview.to_excel(xw, sheet_name="Overview", index=False)
        own_bucket.to_excel(xw, sheet_name="Buckets_OWN_base", index=False)
        mcr_bucket.to_excel(xw, sheet_name="Buckets_MCR_base", index=False)
        ctab.to_excel(xw, sheet_name="Crosstab_0-1-2plus", index=False)
        discrepancies.to_excel(xw, sheet_name="Discrepancies", index=False)

        if engine == "xlsxwriter":
            for name, df in {
                "Overview": overview,
                "Buckets_OWN_base": own_bucket,
                "Buckets_MCR_base": mcr_bucket,
                "Crosstab_0-1-2plus": ctab,
                "Discrepancies": discrepancies
            }.items():
                ws = xw.sheets[name]
                for i, w in enumerate(compute_col_widths(df)):
                    ws.set_column(i, i, w)
        else:
            from openpyxl.utils import get_column_letter
            wb = xw.book
            for name, df in {
                "Overview": overview,
                "Buckets_OWN_base": own_bucket,
                "Buckets_MCR_base": mcr_bucket,
                "Crosstab_0-1-2plus": ctab,
                "Discrepancies": discrepancies
            }.items():
                ws = wb[name]
                for i, w in enumerate(compute_col_widths(df), start=1):
                    ws.column_dimensions[get_column_letter(i)].width = w

    print(f"\n[saved] Excel -> {OUT_XLSX}")

    # ─────────────────────────────────────────────────────────────────────────
    # Rich comparison datasets (PLUS, 1v1, mismatches)
    # ─────────────────────────────────────────────────────────────────────────
    def collapse_dates_from_cols(df, date_cols, fmt="%Y-%m-%d", month_only=True):
        if not date_cols:
            return pd.Series([""] * len(df), index=df.index, dtype="string")
        dt = df[date_cols].apply(pd.to_datetime, errors="coerce")
        if month_only:
            dt = dt.apply(lambda col: col.dt.to_period("M").dt.to_timestamp())
            out = dt.apply(
                lambda r: "|".join(sorted({d.strftime("%Y-%m") for d in r if pd.notna(d)})),
                axis=1
            )
        else:
            out = dt.apply(
                lambda r: "|".join(sorted({d.strftime(fmt) for d in r if pd.notna(d)})),
                axis=1
            )
        return out.astype("string")

    def first_date_str(s):
        if not isinstance(s, str) or not s:
            return ""
        return s.split("|")[0]

    def last_date_str(s):
        if not isinstance(s, str) or not s:
            return ""
        return s.split("|")[-1]

    def month_diff(a, b):
        if not a or not b:
            return np.nan
        A = pd.Period(a, freq="M")
        B = pd.Period(b, freq="M")
        return (A - B).n

    def count_exact_overlap(a, b):
        A = set(a.split("|")) if isinstance(a, str) and a else set()
        B = set(b.split("|")) if isinstance(b, str) and b else set()
        return len(A & B)

    def count_within_one_month(a, b):
        A = [x for x in (a.split("|") if isinstance(a, str) and a else [])]
        B = [y for y in (b.split("|") if isinstance(b, str) and b else [])]
        if not A or not B:
            return 0
        cnt = 0
        for x in A:
            for y in B:
                try:
                    if abs(month_diff(x, y)) <= 1:
                        cnt += 1
                        break
                except Exception:
                    pass
        return cnt

    def jaccard_months(a, b):
        A = set(a.split("|")) if isinstance(a, str) and a else set()
        B = set(b.split("|")) if isinstance(b, str) and b else set()
        if not A and not B:
            return np.nan
        return len(A & B) / max(1, len(A | B))

    def set_minus(a, b):
        A = set(a.split("|")) if isinstance(a, str) and a else set()
        B = set(b.split("|")) if isinstance(b, str) and b else set()
        return "|".join(sorted(A - B))

    # Prepare compact frames
    lite = own.copy()
    mcrw = provider_wide.copy()

    lite_date_cols = [c for c in lite.columns if re.fullmatch(r"chow_date_\d+", c)]
    mcr_date_cols  = [c for c in mcrw.columns  if re.fullmatch(r"chow_\d+_date", c)]

    lite_dates = collapse_dates_from_cols(lite, lite_date_cols, month_only=True)
    mcr_dates  = collapse_dates_from_cols(mcrw, mcr_date_cols,  month_only=True)

    lite_comp = pd.DataFrame({
        "cms_certification_number": lite["cms_certification_number"].astype("string"),
        "num_chows_lite": pd.to_numeric(lite.get("num_chows", 0), errors="coerce").fillna(0).astype(int),
        "lite_dates": lite_dates
    })

    mcr_comp = pd.DataFrame({
        "cms_certification_number": mcrw["cms_certification_number"].astype("string"),
        "num_chows_mcr": pd.to_numeric(mcrw.get("n_chow", 0), errors="coerce").fillna(0).astype(int),
        "mcr_dates": mcr_dates
    })

    comp = (lite_comp
            .merge(mcr_comp, on="cms_certification_number", how="outer")
            .fillna({"num_chows_lite":0, "num_chows_mcr":0, "lite_dates":"", "mcr_dates":""}))

    comp["first_lite_date"] = comp["lite_dates"].apply(first_date_str)
    comp["first_mcr_date"]  = comp["mcr_dates"].apply(first_date_str)
    comp["last_lite_date"]  = comp["lite_dates"].apply(last_date_str)
    comp["last_mcr_date"]   = comp["mcr_dates"].apply(last_date_str)
    comp["delta_first_months"] = comp.apply(lambda r: month_diff(r["first_lite_date"], r["first_mcr_date"]), axis=1)

    comp["n_common_exact"]      = comp.apply(lambda r: count_exact_overlap(r["lite_dates"], r["mcr_dates"]), axis=1)
    comp["n_common_within_1m"]  = comp.apply(lambda r: count_within_one_month(r["lite_dates"], r["mcr_dates"]), axis=1)
    comp["jaccard_months"]      = comp.apply(lambda r: jaccard_months(r["lite_dates"], r["mcr_dates"]), axis=1)
    comp["only_in_lite_dates"]  = comp.apply(lambda r: set_minus(r["lite_dates"], r["mcr_dates"]), axis=1)
    comp["only_in_mcr_dates"]   = comp.apply(lambda r: set_minus(r["mcr_dates"],  r["lite_dates"]), axis=1)

    def classify_match(r):
        nl, nm = int(r["num_chows_lite"]), int(r["num_chows_mcr"])
        if nl == 0 and nm == 0:
            return "both_zero"
        if nl > 0 and nm == 0:
            return "only_lite"
        if nl == 0 and nm > 0:
            return "only_mcr"
        return "both_yes_same_months" if r["n_common_exact"] > 0 else "both_yes_different_months"

    comp["match_type"] = comp.apply(classify_match, axis=1)
    comp["discrepancy_flag"] = comp["match_type"].isin(["only_lite","only_mcr","both_yes_different_months"]).astype(int)

    cols = [
        "cms_certification_number",
        "num_chows_lite","num_chows_mcr",
        "lite_dates","mcr_dates",
        "first_lite_date","first_mcr_date","last_lite_date","last_mcr_date",
        "n_common_exact","n_common_within_1m","jaccard_months",
        "delta_first_months",
        "only_in_lite_dates","only_in_mcr_dates",
        "match_type","discrepancy_flag"
    ]
    comp = comp[cols].sort_values("cms_certification_number").reset_index(drop=True)

    # Save (complete)
    comp.to_csv(PLUS_FP, index=False)
    print(f"[saved] {PLUS_FP}  rows={len(comp):,}  cols={len(comp.columns)}")

    # (1) 1 vs 1 subset
    one_one = comp[(comp["num_chows_lite"] == 1) & (comp["num_chows_mcr"] == 1)].copy()
    one_one.to_csv(ONEVONE_FP, index=False)
    print(f"[saved] {ONEVONE_FP}  rows={len(one_one):,}  cols={len(one_one.columns)}")

    # (2) Mismatches (only_lite / only_mcr / both_yes_different_months)
    mis = comp[comp["match_type"].isin(["only_lite","only_mcr","both_yes_different_months"])].copy()
    mis.to_csv(MIS_FP, index=False)
    print(f"[saved] {MIS_FP}  rows={len(mis):,}  cols={len(mis.columns)}")

print("\n[done]")