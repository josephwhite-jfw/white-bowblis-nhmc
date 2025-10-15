# 02_mcr_controls_pipeline.py
import re
from pathlib import Path
import pandas as pd, numpy as np
from config import MCR_DIR, CLN, MCR_CONTROLS_CSV

def normalize_ccn_any(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("").str.strip().str.upper()
    s = s.str.replace(r"[ \-\/\.]", "", regex=True)
    is_digits = s.str.fullmatch(r"\d+")
    s = s.mask(is_digits, s.str.zfill(6))
    s = s.replace({"": pd.NA})
    return s

def build_mcr_controls():
    sas = sorted(MCR_DIR.glob("mcr_flatfile_20??.sas7bdat"))
    csv = sorted(MCR_DIR.glob("mcr_flatfile_20??.csv"))
    use_sas = bool(sas)
    frames=[]
    cand = dict(
        PRVDR_NUM=["PRVDR_NUM","provnum","prvdr_num","Provider Number"],
        FY_BGN_DT=["FY_BGN_DT","fy_bgn_dt","Cost Report Fiscal Year beginning date"],
        FY_END_DT=["FY_END_DT","fy_end_dt","Cost Report Fiscal Year ending date"],
        MRC_OWNERSHIP=["MRC_OWNERSHIP","MRC_ownership","mrc_ownership","MRC_ownership_code"],
        PAT_DAYS_TOT=["S3_1_PATDAYS_TOTAL","PATDAYS_TOTAL","PATIENT_DAYS_TOTAL"],
        PAT_DAYS_MCR=["S3_1_PATDAYS_MEDICARE","PATDAYS_MEDICARE"],
        PAT_DAYS_MCD=["S3_1_PATDAYS_MEDICAID","PATDAYS_MEDICAID"],
        BEDDAYS_AVAIL=["S3_1_BEDDAYS_AVAL","BEDDAYS_AVAL","BED_DAYS_AVAIL","S3_1_BED_DAYS_AVAIL"],
        TOT_BEDS=["S3_1_TOTALBEDS","TOTAL_BEDS","TOT_BEDS","BEDS","S3_1_BEDS"],
        AVG_BEDS=["AVG_BEDS","AVERAGE_BEDS","AVG_INPT_BEDS","S3_1_AVG_BEDS"],
        STATE=["MCR_STATE","STATE","PROV_STATE","STATE_CD","PROV_STATE_CD"],
        URBAN=["MCR_URBAN","URBAN_RURAL","URBAN_RURAL_INDICATOR","URBAN","URBRUR","URBAN_IND"],
        HOME_OFFICE=["MCR_HOME_OFFICE","HOME_OFFICE","HOME_OFFICE_IND","HOME_OFFICE_FLAG"]
    )
    def _pick(cols, keys):
        for k in keys:
            if k in cols: return k
        return None

    if use_sas:
        import pyreadstat
        files = sas
        print("[MCR] using SAS files.")
        for fp in files:
            df,_ = pyreadstat.read_sas7bdat(str(fp), disable_datetime_conversion=0)
            df.columns = [c.upper().strip() for c in df.columns]
            keep = {k:_pick(df.columns,v) for k,v in cand.items()}
            for k,v in keep.items():
                if v is None:
                    df[k]=pd.NA; keep[k]=k
            sub = df[[keep[k] for k in keep]].copy(); sub.columns=list(keep.keys())
            sub["file_year"] = int(re.search(r"(\d{4})", fp.name).group(1))
            frames.append(sub)
    else:
        print("[MCR] using CSV files.")
        for fp in csv:
            df = pd.read_csv(fp, low_memory=False)
            df.columns = [c.upper().strip() for c in df.columns]
            keep = {k:_pick(df.columns,v) for k,v in cand.items()}
            for k,v in keep.items():
                if v is None:
                    df[k]=pd.NA; keep[k]=k
            sub = df[[keep[k] for k in keep]].copy(); sub.columns=list(keep.keys())
            sub["file_year"] = int(re.search(r"(\d{4})", fp.name).group(1))
            frames.append(sub)

    raw = pd.concat(frames, ignore_index=True)
    raw["cms_certification_number"] = normalize_ccn_any(raw["PRVDR_NUM"])
    for c in ["FY_BGN_DT","FY_END_DT"]: raw[c]=pd.to_datetime(raw[c], errors="coerce")
    for c in ["PAT_DAYS_TOT","PAT_DAYS_MCR","PAT_DAYS_MCD","BEDDAYS_AVAIL","AVG_BEDS","TOT_BEDS"]:
        raw[c]=pd.to_numeric(raw[c], errors="coerce")

    def own_bucket(code):
        s = str(code).strip().upper() if pd.notna(code) else ""
        if s in {"1","2"}: return "Nonprofit"
        if s in {"3","4","5","6"}: return "For-profit"
        if s in {"7","8","9","10","11","12","13"}: return "Government"
        return None
    raw["ownership_type"] = raw["MRC_OWNERSHIP"].map(own_bucket)
    raw["state"] = raw["STATE"].astype("string").str.strip().str.upper().replace({"":pd.NA})
    def _urb(x):
        if pd.isna(x): return pd.NA
        s=str(x).strip().upper()
        if s in {"U","URBAN","1","Y","YES","TRUE"}: return "Urban"
        if s in {"R","RURAL","0","N","NO","FALSE","2"}: return "Rural"
        return pd.NA
    raw["urban_rural"]=raw["URBAN"].map(_urb)
    def _chain(v):
        if pd.isna(v): return 0
        try:
            f=float(str(v).strip())
            return int(f!=0.0)
        except:
            s=str(v).strip().upper()
            return 0 if s in {"","0","N","NO","FALSE"} else 1
    raw["is_chain"]=raw["HOME_OFFICE"].apply(_chain).astype("Int8")

    # beds & occupancy
    period_days = (raw["FY_END_DT"] - raw["FY_BGN_DT"]).dt.days.add(1).where(lambda s: s>0)
    raw["num_beds"] = np.select(
        [raw["TOT_BEDS"].notna(), raw["AVG_BEDS"].notna(), raw["BEDDAYS_AVAIL"].notna() & period_days.notna() & (period_days>0)],
        [raw["TOT_BEDS"], raw["AVG_BEDS"], raw["BEDDAYS_AVAIL"]/period_days],
        default=np.nan
    )
    den1=raw["BEDDAYS_AVAIL"]; den2=raw["num_beds"]*period_days
    occ = np.where(raw["PAT_DAYS_TOT"].notna() & den1.notna() & (den1>0),
                   100*raw["PAT_DAYS_TOT"]/den1,
                   np.where(raw["PAT_DAYS_TOT"].notna() & den2.notna() & (den2>0),
                            100*raw["PAT_DAYS_TOT"]/den2, np.nan))
    raw["occupancy_rate"]=pd.to_numeric(occ, errors="coerce").clip(0,100)
    raw["pct_medicare"]=pd.to_numeric(100*raw["PAT_DAYS_MCR"]/raw["PAT_DAYS_TOT"], errors="coerce").clip(0,100)
    raw["pct_medicaid"]=pd.to_numeric(100*raw["PAT_DAYS_MCD"]/raw["PAT_DAYS_TOT"], errors="coerce").clip(0,100)

    # monthly expand
    def month_range(s,e):
        if pd.isna(s) or pd.isna(e): return []
        if e<s: s,e=e,s
        return pd.period_range(s, e, freq="M").to_timestamp("s")
    rows=[]
    it = raw.dropna(subset=["cms_certification_number","FY_BGN_DT","FY_END_DT"]).itertuples(index=False)
    for r in it:
        months = month_range(r.FY_BGN_DT, r.FY_END_DT)
        if len(months)==0: continue
        block = pd.DataFrame({"month": months})
        block["cms_certification_number"] = r.cms_certification_number
        block["ownership_type"] = getattr(r,"ownership_type",pd.NA)
        block["pct_medicare"]  = getattr(r,"pct_medicare",np.nan)
        block["pct_medicaid"]  = getattr(r,"pct_medicaid",np.nan)
        block["num_beds"]      = getattr(r,"num_beds",np.nan)
        block["occupancy_rate"]= getattr(r,"occupancy_rate",np.nan)
        block["state"]         = getattr(r,"state",pd.NA)
        block["urban_rural"]   = getattr(r,"urban_rural",pd.NA)
        block["is_chain"]      = getattr(r,"is_chain",pd.NA)
        rows.append(block)
    monthly = (pd.concat(rows, ignore_index=True)
               if rows else pd.DataFrame(columns=[
                   "cms_certification_number","month","ownership_type","pct_medicare","pct_medicaid",
                   "num_beds","occupancy_rate","state","urban_rural","is_chain"
               ]))
    # dedupe overlaps
    monthly = (monthly.sort_values(["cms_certification_number","month"])
                      .groupby(["cms_certification_number","month"], as_index=False)
                      .agg({
                          "ownership_type": lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
                          "state":         lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
                          "urban_rural":   lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
                          "is_chain":      "max",
                          "pct_medicare":  "mean",
                          "pct_medicaid":  "mean",
                          "num_beds":      "mean",
                          "occupancy_rate":"mean",
                      }))
    for c in ["pct_medicare","pct_medicaid","occupancy_rate"]:
        monthly[c]=pd.to_numeric(monthly[c], errors="coerce").clip(0,100)
    monthly["num_beds"]=pd.to_numeric(monthly["num_beds"], errors="coerce")

    monthly.to_csv(MCR_CONTROLS_CSV, index=False)
    print(f"[MCR] monthly controls rows={len(monthly):,} CCNs={monthly['cms_certification_number'].nunique():,} → {MCR_CONTROLS_CSV}")

if __name__ == "__main__":
    build_mcr_controls()