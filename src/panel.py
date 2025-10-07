# nhmc/panel.py
# Final PBJ panel ← PBJ + CHOW agreement + MCR controls (light) + provider-info vars
# outputs: pbj_panel_with_chow_dummies.csv & pbj_panel_analytic.csv

from __future__ import annotations
import re
import numpy as np
import pandas as pd

from .paths import (INTERIM, CLEAN, PROV_DIR,
                    PBJ_MONTHLY, CHOW_LITE, MCR_WIDE,
                    PROV_COMBINED, PANEL_FULL, PANEL_ANALYTIC, RAW_DIR)
from .utils import normalize_ccn_any, to_monthstart

INTERIM.mkdir(parents=True, exist_ok=True)
CLEAN.mkdir(parents=True, exist_ok=True)

def _first_event_month(df: pd.DataFrame, pat_list) -> pd.Series:
    cols = []
    for pat in pat_list:
        cols += [c for c in df.columns if re.search(pat, c, re.I)]
    cols = sorted(set(cols))
    if not cols:
        return pd.Series(pd.NaT, index=df.index)
    tmp = df[cols].apply(pd.to_datetime, errors="coerce")
    return to_monthstart(tmp.min(axis=1))

def _within_k_months(a, b, k=6) -> bool:
    if pd.isna(a) or pd.isna(b): return False
    pa, pb = pd.Period(a, "M"), pd.Period(b, "M")
    return abs((pa - pb).n) <= k

def _load_provider_info_monthly() -> pd.DataFrame:
    src = PROV_COMBINED
    if not src.exists():
        print("[panel] provider_info_combined.csv not found → skipping provider vars")
        return pd.DataFrame(columns=["cms_certification_number","month","case_mix_total_num","ccrc_facility","sff_class"])
    df = pd.read_csv(src, dtype=str, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    ccn_col = next((c for c in ["cms_certification_number","ccn","provnum","provider_number","federal_provider_number"] if c in df.columns), None)
    date_col= next((c for c in ["date","month","period","as_of_date"] if c in df.columns), None)
    if ccn_col is None or date_col is None:
        print("[panel] provider-info: missing ccn/date columns → skipping")
        return pd.DataFrame(columns=["cms_certification_number","month","case_mix_total_num","ccrc_facility","sff_class"])

    out = pd.DataFrame({
        "cms_certification_number": normalize_ccn_any(df[ccn_col]),
        "month": to_monthstart(df[date_col]),
        "case_mix_total_num": pd.to_numeric(df.get("case_mix_total_num"), errors="coerce")
    })
    # ccrc
    def _boolish(x):
        s = str(x).strip().lower()
        return {"1":1,"y":1,"yes":1,"true":1,"t":1,"0":0,"n":0,"no":0,"false":0,"f":0}.get(s, np.nan)
    out["ccrc_facility"] = pd.Series([_boolish(x) for x in df.get("ccrc_facility", [])]).astype("float").astype("Int8", errors="ignore")
    # sff
    s = df.get("sff_class", pd.Series(index=df.index, dtype="object")).astype(str).str.strip().str.lower()
    def _sff_class(v: str) -> str:
        if v in {"current","candidate","former","none"}: return v
        if v in {"y","yes","true","1","sff"}: return "current"
        if "candidate" in v: return "candidate"
        if "former" in v or "graduated" in v or "terminated" in v: return "former"
        if v in {"n","no","false","0"}: return "none"
        return "unknown"
    out["sff_class"] = s.map(_sff_class)
    out = (out.dropna(subset=["cms_certification_number","month"])
             .drop_duplicates(["cms_certification_number","month"])
             .reset_index(drop=True))
    return out

def _build_mcr_controls_monthly() -> pd.DataFrame:
    """Very light monthly controls from MCR CSVs if present (state, urban/rural, beds, occupancy, payer mix)."""
    MCR_DIR = RAW_DIR / "medicare-cost-reports"
    files = sorted(MCR_DIR.glob("mcr_flatfile_20??.csv"))
    if not files:
        print("[panel] no MCR CSVs found, skipping controls")
        return pd.DataFrame(columns=["cms_certification_number","month","ownership_type","pct_medicare","pct_medicaid",
                                     "num_beds","occupancy_rate","state","urban_rural","is_chain"])
    frames = []
    for fp in files:
        df = pd.read_csv(fp, low_memory=False)
        df.columns = [c.upper().strip() for c in df.columns]
        def pick(*cands): 
            for c in cands:
                if c in df.columns: return c
            return None
        keep = dict(
            PRVDR_NUM = pick("PRVDR_NUM","PROVNUM","PROVIDER NUMBER"),
            FY_BGN_DT = pick("FY_BGN_DT"),
            FY_END_DT = pick("FY_END_DT"),
            STATE     = pick("STATE","PROV_STATE","STATE_CD"),
            URBAN     = pick("URBAN_RURAL","URBAN","MCR_URBAN","URBRUR"),
            TOT_BEDS  = pick("S3_1_TOTALBEDS","TOTAL_BEDS","BEDS"),
            BEDDAYS   = pick("S3_1_BEDDAYS_AVAL","BEDDAYS_AVAL","BED_DAYS_AVAIL"),
            PAT_TOT   = pick("S3_1_PATDAYS_TOTAL","PATDAYS_TOTAL","PATIENT_DAYS_TOTAL"),
            PAT_MCR   = pick("S3_1_PATDAYS_MEDICARE","PATDAYS_MEDICARE"),
            PAT_MCD   = pick("S3_1_PATDAYS_MEDICAID","PATDAYS_MEDICAID"),
            OWN_CODE  = pick("MRC_OWNERSHIP"),
            HOME_OFF  = pick("HOME_OFFICE","MCR_HOME_OFFICE"),
        )
        sub = pd.DataFrame({k: df[v] if v in df.columns else pd.NA for k,v in keep.items()})
        frames.append(sub)

    raw = pd.concat(frames, ignore_index=True)
    raw["cms_certification_number"] = normalize_ccn_any(raw["PRVDR_NUM"])
    raw["FY_BGN_DT"] = pd.to_datetime(raw["FY_BGN_DT"], errors="coerce")
    raw["FY_END_DT"] = pd.to_datetime(raw["FY_END_DT"], errors="coerce")
    for c in ["TOT_BEDS","BEDDAYS","PAT_TOT","PAT_MCR","PAT_MCD","HOME_OFF"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["STATE"] = raw["STATE"].astype("string").str.upper().str.strip()

    period_days = (raw["FY_END_DT"] - raw["FY_BGN_DT"]).dt.days.add(1)
    num_beds = np.where(raw["TOT_BEDS"].notna(), raw["TOT_BEDS"],
                 np.where((raw["BEDDAYS"].notna()) & (period_days>0), raw["BEDDAYS"]/period_days, np.nan))
    raw["num_beds"] = pd.to_numeric(num_beds, errors="coerce")

    occ = np.where((raw["PAT_TOT"].notna()) & (raw["BEDDAYS"].notna()) & (raw["BEDDAYS"]>0),
                   100.0*raw["PAT_TOT"]/raw["BEDDAYS"], np.nan)
    raw["occupancy_rate"] = pd.to_numeric(occ, errors="coerce").clip(0,100)

    def _share(n, d): 
        n = pd.to_numeric(n, errors="coerce"); d = pd.to_numeric(d, errors="coerce")
        return (100.0*n/d).where(d>0)
    raw["pct_medicare"] = _share(raw["PAT_MCR"], raw["PAT_TOT"]).clip(0,100)
    raw["pct_medicaid"] = _share(raw["PAT_MCD"], raw["PAT_TOT"]).clip(0,100)

    def _own_bucket(s):
        s = str(s).strip().upper().replace(".0","")
        if s in {"1","2"}: return "Nonprofit"
        if s in {"3","4","5","6"}: return "For-profit"
        if s in {"7","8","9","10","11","12","13"}: return "Government"
        return None
    raw["ownership_type"] = raw["OWN_CODE"].map(_own_bucket)

    def _urban(s):
        s = str(s).strip().upper()
        if s in {"U","URBAN","1","YES","Y","TRUE"}: return "Urban"
        if s in {"R","RURAL","0","NO","N","FALSE","2"}: return "Rural"
        return None
    raw["urban_rural"] = raw["URBAN"].map(_urban)
    raw["is_chain"]    = (raw["HOME_OFF"].fillna(0).astype(float)!=0.0).astype("Int8")

    # expand to monthly
    def month_span(b, e):
        if pd.isna(b) or pd.isna(e): return []
        s = pd.Period(b, "M").to_timestamp("s"); t = pd.Period(e, "M").to_timestamp("s")
        if t<s: s,t=t,s
        return list(pd.period_range(s, t, freq="M").to_timestamp("s"))

    rows = []
    for r in raw.dropna(subset=["cms_certification_number","FY_BGN_DT","FY_END_DT"]).itertuples(index=False):
        for m in month_span(r.FY_BGN_DT, r.FY_END_DT):
            rows.append(dict(cms_certification_number=r.cms_certification_number, month=m,
                             ownership_type=r.ownership_type, pct_medicare=r.pct_medicare,
                             pct_medicaid=r.pct_medicaid, num_beds=r.num_beds,
                             occupancy_rate=r.occupancy_rate, state=r.STATE, 
                             urban_rural=r.urban_rural, is_chain=r.is_chain))
    if not rows:
        return pd.DataFrame(columns=["cms_certification_number","month"])
    monthly = pd.DataFrame(rows)
    monthly = (monthly.sort_values(["cms_certification_number","month"])
                      .groupby(["cms_certification_number","month"], as_index=False)
                      .agg({"ownership_type":lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
                            "state":lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
                            "urban_rural":lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
                            "is_chain":"max",
                            "pct_medicare":"mean","pct_medicaid":"mean",
                            "num_beds":"mean","occupancy_rate":"mean"}))
    for c in ["pct_medicare","pct_medicaid","occupancy_rate"]:
        monthly[c] = pd.to_numeric(monthly[c], errors="coerce").clip(0,100)
    monthly["num_beds"] = pd.to_numeric(monthly["num_beds"], errors="coerce")
    return monthly

def build_final_panels() -> None:
    # load
    pbj  = pd.read_csv(PBJ_MONTHLY, dtype={"cms_certification_number":"string"}, low_memory=False)
    lite = pd.read_csv(CHOW_LITE, dtype={"cms_certification_number":"string"}, low_memory=False)
    mcr  = pd.read_csv(MCR_WIDE, dtype={"cms_certification_number":"string"}, low_memory=False)

    pbj["cms_certification_number"] = normalize_ccn_any(pbj["cms_certification_number"])
    lite["cms_certification_number"] = normalize_ccn_any(lite["cms_certification_number"])
    mcr["cms_certification_number"]  = normalize_ccn_any(mcr["cms_certification_number"])

    # month
    if "year_month" in pbj.columns:
        pbj["month"] = pd.PeriodIndex(pbj["year_month"].astype(str), freq="M").to_timestamp("s")
    else:
        pbj["month"] = to_monthstart(pbj["month"])

    # first-event months
    lite["first_event_month_lite"] = _first_event_month(lite, [r"^chow_date_\d+$", r"^chow_\d+_date$"])
    mcr["first_event_month_mcr"]   = _first_event_month(mcr,  [r"^chow_\d+_date$"])

    merged_overlap = lite.merge(mcr, on="cms_certification_number", how="inner")

    def _agree(r):
        a, b = r["first_event_month_lite"], r["first_event_month_mcr"]
        own_n = r.get("num_chows",0); mcr_n = r.get("n_chow",0)
        if int(own_n)==0 and int(mcr_n)==0:
            return pd.Series({"agreement":"match_0","change_month":pd.NaT})
        if int(own_n)==1 and int(mcr_n)==1 and _within_k_months(a,b,6):
            return pd.Series({"agreement":"match_1_within_6m","change_month": a if pd.notna(a) else b})
        return pd.Series({"agreement":"mismatch","change_month":pd.NaT})

    ag = merged_overlap.apply(_agree, axis=1)
    agree = pd.concat([merged_overlap[["cms_certification_number","num_chows","n_chow",
                                       "first_event_month_lite","first_event_month_mcr"]], ag], axis=1)

    panel = pbj.merge(agree[["cms_certification_number","agreement","change_month"]],
                      on="cms_certification_number", how="inner")
    panel["treat_post"] = 0
    msk = panel["agreement"].eq("match_1_within_6m") & panel["change_month"].notna()
    panel.loc[msk, "treat_post"] = (panel.loc[msk,"month"] >= panel.loc[msk,"change_month"]).astype(int)
    panel["event_time"] = np.nan
    panel.loc[msk, "event_time"] = (
        (panel.loc[msk,"month"].values.astype("datetime64[M]") - 
         panel.loc[msk,"change_month"].values.astype("datetime64[M]")).astype(int)
    )
    ever = (panel.groupby("cms_certification_number", as_index=False)["treat_post"].max()
                 .rename(columns={"treat_post":"ever_treated"}))
    panel = panel.merge(ever, on="cms_certification_number", how="left")

    # controls (MCR monthly)
    controls = _build_mcr_controls_monthly()
    if not controls.empty:
        panel = panel.merge(controls, on=["cms_certification_number","month"], how="left")
        # ffill/bfill within CCN for controls
        panel = panel.sort_values(["cms_certification_number","month"]).reset_index(drop=True)
        for cat in ["ownership_type","state","urban_rural","is_chain"]:
            if cat in panel.columns:
                panel[cat] = panel.groupby("cms_certification_number")[cat].transform(lambda s: s.ffill().bfill())
        for num in ["pct_medicare","pct_medicaid","num_beds","occupancy_rate"]:
            if num in panel.columns:
                panel[num] = panel.groupby("cms_certification_number")[num].transform(lambda s: s.ffill().bfill())

        # dummies (keep labels too in full file)
        if "ownership_type" in panel.columns:
            up = panel["ownership_type"].astype("string").str.upper()
            panel["for_profit"] = (up=="FOR-PROFIT").astype("Int8")
            panel["non_profit"] = (up=="NONPROFIT").astype("Int8")
        if "urban_rural" in panel.columns:
            panel["urban"] = (panel["urban_rural"].astype("string").str.upper()=="URBAN").astype("Int8")
        if "is_chain" in panel.columns:
            panel["is_chain"] = panel["is_chain"].fillna(0).astype("Int8")

    # provider-info vars
    prov = _load_provider_info_monthly()
    if not prov.empty:
        panel = panel.merge(prov, on=["cms_certification_number","month"], how="left")
        # sff dummies
        if "sff_class" in panel.columns:
            s = panel["sff_class"].astype("string").str.lower()
            panel["sff_current"]   = (s=="current").astype("Int8")
            panel["sff_candidate"] = (s=="candidate").astype("Int8")
            panel["sff_former"]    = (s=="former").astype("Int8")
        if "ccrc_facility" in panel.columns:
            panel["ccrc_facility"] = pd.to_numeric(panel["ccrc_facility"], errors="coerce").fillna(0).astype("Int8")

    # save full
    panel.to_csv(PANEL_FULL, index=False)
    print(f"[panel] full → {PANEL_FULL} (rows={len(panel):,})")

    # analytic subset (keep match_0 + match_1_within_6m; drop raw cats)
    analytic = panel[panel["agreement"].isin(["match_0","match_1_within_6m"])].copy()
    drop_raw = [c for c in ["ownership_type","urban_rural","sff_class"] if c in analytic.columns]
    analytic = analytic.drop(columns=drop_raw)
    analytic.to_csv(PANEL_ANALYTIC, index=False)
    print(f"[panel] analytic → {PANEL_ANALYTIC} (rows={len(analytic):,})")