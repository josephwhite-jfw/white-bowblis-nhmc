# 05_build_final_panel.py  — FINAL PANEL (PBJ × agreement × controls × provider-info)
# Creates a clean, 1-row-per (CCN, month) panel with event-time, CHOW flags, controls, and case-mix.
# -----------------------------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    PBJ_MONTHLY_CSV,          # built by 04 script
    PROV_COMBINED,            # provider_info_combined.csv (includes case-mix, ccrc/sff)
    HOSP_LATEST_CSV,          # latest TRUE-only hospital list
    MCR_CONTROLS_CSV,         # monthly ownership + controls
    CHOW_EVENTS_MCR, CHOW_EVENTS_OWNERSHIP, CHOW_OVERLAP_AGREE,
    FINAL_PANEL_CSV,
)

# ----------------------------- Helpers ----------------------------------------
def normalize_ccn_any(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.upper().str.replace(r"[^\dA-Z]", "", regex=True)
    s = s.where(~s.str.fullmatch(r"\d+"), s.str.zfill(6))
    return s

def to_month(s) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp("s")

def months_diff(a, b) -> pd.Series:
    pa, pb = to_month(a), to_month(b)
    out = pd.Series(np.nan, index=pa.index, dtype="float")
    mask = pa.notna() & pb.notna()
    if mask.any():
        dif = (pa[mask].dt.to_period("M") - pb[mask].dt.to_period("M")).astype("int")
        out.loc[mask] = dif.astype("float")
    return out

def ensure_unique(df, key, name):
    """Ensure dataframe has unique rows by key; if not, reduce & warn."""
    dupe_mask = df.duplicated(key, keep=False)
    if dupe_mask.any():
        n = dupe_mask.sum()
        print(f"[warn] {name} has {n:,} duplicate rows on {key}. De-duplicating by keeping first.")
        df = df.drop_duplicates(key, keep="first")
    return df

# ---------------------- Agreement / overlap logic -----------------------------
def load_agreement_flexible() -> pd.DataFrame:
    if Path(CHOW_OVERLAP_AGREE).exists():
        df = pd.read_csv(CHOW_OVERLAP_AGREE, dtype=str, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        ccn_col = next((c for c in df.columns if c in {"cms_certification_number","ccn","provnum","provider_number"}), None)
        if not ccn_col:
            raise ValueError("Agreement file found but no CCN-like column.")

        lite_col = next((c for c in [
            "first_event_month_lite","lite_first_event_month",
            "first_month_lite","first_lite_month","lite_month_first","change_month_lite"
        ] if c in df.columns), None)
        mcr_col = next((c for c in [
            "first_event_month_mcr","mcr_first_event_month",
            "first_month_mcr","first_mcr_month","mcr_month_first","change_month_mcr"
        ] if c in df.columns), None)
        num_lite_col = next((c for c in df.columns if c in {"num_chows","n_lite","num_lite"}), None)
        num_mcr_col  = next((c for c in df.columns if c in {"n_chow","n_mcr","num_mcr"}), None)

        out = pd.DataFrame({
            "cms_certification_number": normalize_ccn_any(df[ccn_col]),
            "first_event_month_lite": to_month(df[lite_col]) if lite_col else pd.NaT,
            "first_event_month_mcr":  to_month(df[mcr_col])  if mcr_col  else pd.NaT,
            "num_chows": pd.to_numeric(df[num_lite_col], errors="coerce").fillna(0).astype(int) if num_lite_col else 0,
            "n_chow":    pd.to_numeric(df[num_mcr_col],  errors="coerce").fillna(0).astype(int) if num_mcr_col  else 0,
        })
        return out

    if not (Path(CHOW_EVENTS_OWNERSHIP).exists() and Path(CHOW_EVENTS_MCR).exists()):
        raise FileNotFoundError(
            "No agreement file and cannot rebuild from CHOW events "
            f"(need {Path(CHOW_EVENTS_OWNERSHIP).name} & {Path(CHOW_EVENTS_MCR).name})."
        )

    lite = pd.read_csv(CHOW_EVENTS_OWNERSHIP, dtype=str, low_memory=False)
    mcr  = pd.read_csv(CHOW_EVENTS_MCR,      dtype=str, low_memory=False)
    for t in (lite, mcr): t.columns = [c.strip().lower() for c in t.columns]

    l_ccn = next((c for c in lite.columns if c in {"cms_certification_number","ccn","provnum","provider_number"}), None)
    m_ccn = next((c for c in mcr.columns  if c in {"cms_certification_number","ccn","provnum","provider_number"}), None)
    l_dt  = next((c for c in ["event_month","first_event_month","month","change_month"] if c in lite.columns), None)
    m_dt  = next((c for c in ["event_month","first_event_month","month","change_month"] if c in mcr.columns),  None)

    lite_grp = (lite.assign(
                    cms_certification_number = normalize_ccn_any(lite[l_ccn]),
                    _month = to_month(lite[l_dt])
                )
                .dropna(subset=["cms_certification_number"])
                .groupby("cms_certification_number", as_index=False)
                .agg(first_event_month_lite=("._month".strip("."), "min"),
                     num_chows=("._month".strip("."), "count")))
    mcr_grp = (mcr.assign(
                    cms_certification_number = normalize_ccn_any(mcr[m_ccn]),
                    _month = to_month(mcr[m_dt])
              )
              .dropna(subset=["cms_certification_number"])
              .groupby("cms_certification_number", as_index=False)
              .agg(first_event_month_mcr=("._month".strip("."), "min"),
                   n_chow=("._month".strip("."), "count")))

    out = lite_grp.merge(mcr_grp, on="cms_certification_number", how="outer").fillna({"num_chows":0,"n_chow":0})
    out["num_chows"] = out["num_chows"].astype(int)
    out["n_chow"]    = out["n_chow"].astype(int)
    return out

def classify_agreement(row) -> str:
    num_l, num_m = row.get("num_chows", 0), row.get("n_chow", 0)
    d_l, d_m     = row.get("first_event_month_lite"), row.get("first_event_month_mcr")
    if (num_l == 0) and (num_m == 0):
        return "match_0"
    if (num_l >= 1) and (num_m >= 1) and pd.notna(d_l) and pd.notna(d_m):
        mdiff = abs((to_month(d_l).to_period("M") - to_month(d_m).to_period("M")).n)
        if mdiff <= 6:
            return "match_1_within_6m"
    return "overlap_other"

def choose_change_month(row, prefer="lite"):
    a = row.get("agreement")
    if a == "match_0": return pd.NaT
    if a == "match_1_within_6m":
        if prefer == "mcr": return to_month(row.get("first_event_month_mcr"))
        return to_month(row.get("first_event_month_lite"))
    return pd.NaT

# ----------------------------- Build panel ------------------------------------
def main(prefer_change="lite"):
    # PBJ base
    pbj = pd.read_csv(PBJ_MONTHLY_CSV, dtype={"cms_certification_number":"string"}, parse_dates=["month"], low_memory=False)
    pbj["cms_certification_number"] = normalize_ccn_any(pbj["cms_certification_number"])
    print(f"[PBJ] rows={len(pbj):,} CCNs={pbj['cms_certification_number'].nunique():,}")

    # Hospital filter
    if Path(HOSP_LATEST_CSV).exists():
        hosp = pd.read_csv(HOSP_LATEST_CSV, dtype={"cms_certification_number":"string"})
        hosp["cms_certification_number"] = normalize_ccn_any(hosp["cms_certification_number"])
        before = pbj["cms_certification_number"].nunique()
        pbj = pbj[~pbj["cms_certification_number"].isin(set(hosp["cms_certification_number"]))]
        after = pbj["cms_certification_number"].nunique()
        print(f"[hospital filter] CCNs {before:,} -> {after:,}")
    print(f"[PBJ (post hospital filter)] rows={len(pbj):,} CCNs={pbj['cms_certification_number'].nunique():,}")

    # Ensure uniqueness of PBJ by (CCN, month)
    pbj = ensure_unique(pbj, ["cms_certification_number","month"], "PBJ base")

    # Agreement / change_month
    agree = load_agreement_flexible()
    agree["cms_certification_number"] = normalize_ccn_any(agree["cms_certification_number"])
    agree["agreement"]    = agree.apply(classify_agreement, axis=1)
    agree["change_month"] = agree.apply(lambda r: choose_change_month(r, prefer_change), axis=1)
    print(f"[agreement] counts: {agree['agreement'].value_counts(dropna=False).to_dict()}")
    print(f"[agreement] non-null change_month CCNs: {agree['change_month'].notna().sum():,}")
    agree = agree[["cms_certification_number","agreement","first_event_month_lite","first_event_month_mcr","num_chows","n_chow","change_month"]]
    agree = ensure_unique(agree, ["cms_certification_number"], "agreement")

    panel = pbj.merge(agree, on="cms_certification_number", how="left", validate="m:1")  # keep all PBJ
    print(f"[merge] after agree: rows={len(panel):,}")

    # Event-time & post
    panel["event_time"] = months_diff(panel["month"], panel["change_month"])
    panel["treat_post"] = ((panel["change_month"].notna()) & (panel["event_time"] >= 0)).astype("Int64").fillna(0)

    # Controls (MCR) — enforce uniqueness on (CCN, month)
    if Path(MCR_CONTROLS_CSV).exists():
        ctrl = pd.read_csv(MCR_CONTROLS_CSV, dtype={"cms_certification_number":"string"}, parse_dates=["month"], low_memory=False)
        ctrl["cms_certification_number"] = normalize_ccn_any(ctrl["cms_certification_number"])
        keep = ["cms_certification_number","month","ownership_type","pct_medicare","pct_medicaid","num_beds","occupancy_rate","state","urban_rural","is_chain"]
        for c in keep:
            if c not in ctrl.columns: ctrl[c] = pd.NA
        ctrl = ctrl[keep]
        ctrl = ensure_unique(ctrl, ["cms_certification_number","month"], "MCR controls")
        before = len(panel)
        panel = panel.merge(ctrl, on=["cms_certification_number","month"], how="left", validate="m:1")
        print(f"[merge] controls: {before:,} -> {len(panel):,}")
    else:
        print("[controls] missing; skipped.")

    # Provider-info (case-mix + ccrc/sff) — enforce uniqueness on (CCN, month)
    if Path(PROV_COMBINED).exists():
        prov = pd.read_csv(PROV_COMBINED, dtype={"cms_certification_number":"string"}, parse_dates=["date"], low_memory=False)
        prov["cms_certification_number"] = normalize_ccn_any(prov["cms_certification_number"])
        prov["month"] = to_month(prov["date"])

        bring = ["cms_certification_number","month"]
        # case-mix fields that existed before
        for c in [
            "case_mix_total_num",
            "case_mix_quartile_nat","case_mix_decile_nat",
            "case_mix_quartile_state","case_mix_decile_state",
            "ccrc_facility","sff_facility","sff_class"
        ]:
            if c in prov.columns: bring.append(c)

        prov_small = prov[bring].dropna(subset=["cms_certification_number","month"])
        prov_small = ensure_unique(prov_small, ["cms_certification_number","month"], "provider-info")
        before = len(panel)
        panel = panel.merge(prov_small, on=["cms_certification_number","month"], how="left", validate="m:1")
        print(f"[merge] provider-info: {before:,} -> {len(panel):,}")
    else:
        print("[provider-info] missing; skipped case-mix/ccrc/sff merge.")

    # Convenience dummies rebuilt here (stable output set)
    # ownership_type → for_profit / non_profit (Gov = 0/0)
    if "ownership_type" in panel.columns:
        ot = panel["ownership_type"].astype("string").str.lower()
        panel["for_profit"] = ot.eq("for-profit").astype("Int64")
        panel["non_profit"] = ot.eq("nonprofit").astype("Int64")
    else:
        panel["for_profit"] = pd.NA
        panel["non_profit"] = pd.NA

    # urban flag from urban_rural
    if "urban_rural" in panel.columns:
        ur = panel["urban_rural"].astype("string").str.lower().str.strip()
        panel["urban"] = ur.eq("urban").astype("Int64")
    else:
        panel["urban"] = pd.NA

    # Type & NA cleanups
    for c in ["is_chain","ccrc_facility","sff_facility","treat_post"]:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0).astype("Int64")

    # Final ordering (optional but helpful)
    front = [
        "cms_certification_number","month",
        "agreement","change_month","event_time","treat_post",
        "for_profit","non_profit","ownership_type","is_chain",
        "num_beds","occupancy_rate","state","urban_rural","urban",
        "pct_medicare","pct_medicaid",
        "ccrc_facility","sff_facility","sff_class",
        "case_mix_total_num","case_mix_quartile_nat","case_mix_decile_nat",
        "case_mix_quartile_state","case_mix_decile_state",
    ]
    existing_front = [c for c in front if c in panel.columns]
    other_cols = [c for c in panel.columns if c not in existing_front]
    panel = panel[existing_front + other_cols]

    # Save
    panel = panel.sort_values(["cms_certification_number","month"]).reset_index(drop=True)
    panel.to_csv(FINAL_PANEL_CSV, index=False)
    print(f"[final] {FINAL_PANEL_CSV} rows={len(panel):,} cols={panel.shape[1]}")

    # Quick sanity: 1 row per (CCN,month)
    dup = panel.duplicated(["cms_certification_number","month"]).sum()
    if dup:
        print(f"[warn] final has {dup:,} duplicate (CCN,month) rows; investigate upstream merges.")
    else:
        print("[final] unique (CCN,month) ✓")

if __name__ == "__main__":
    # prefer_change ∈ {"lite","mcr"} for tie-breaks
    main(prefer_change="lite")
