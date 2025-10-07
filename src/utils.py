import re, numpy as np, pandas as pd
from .paths import HOSP_PANEL, HOSP_BY_CCN

# === Normalizers ===
def normalize_ccn_any(s: pd.Series) -> pd.Series:
    s = s.astype("string").fillna("").str.strip().str.upper()
    s = s.str.replace(r"[ \-\/\.]", "", regex=True)
    s = s.mask(s.str.fullmatch(r"\d+"), s.str.zfill(6))
    return s.replace({"": pd.NA})

def to_monthstart(x) -> pd.Series:
    s = pd.to_datetime(x, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp("s")

def bool_from_any(x: pd.Series) -> pd.Series:
    s = x.astype("string").str.strip().str.lower()
    return s.map({"1":True,"y":True,"yes":True,"true":True,"t":True,
                  "0":False,"n":False,"no":False,"false":False,"f":False}).astype("boolean")

# === Hospital filters (single source of truth) ===
def load_hosp_panel_true():
    if not HOSP_PANEL.exists():
        return pd.DataFrame(columns=["cms_certification_number","date","provider_resides_in_hospital"])
    df = pd.read_csv(HOSP_PANEL, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    df["cms_certification_number"] = normalize_ccn_any(df["cms_certification_number"])
    df["provider_resides_in_hospital"] = bool_from_any(df["provider_resides_in_hospital"]).fillna(False)
    df["date"] = to_monthstart(df["date"])
    return df[df["provider_resides_in_hospital"] == True][["cms_certification_number","date"]].drop_duplicates()

def load_hosp_ccn_dropset() -> set:
    if not HOSP_BY_CCN.exists():
        return set()
    df = pd.read_csv(HOSP_BY_CCN, dtype=str, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    ccn = next((c for c in ["cms_certification_number","ccn","provnum","prvdr_num"] if c in df.columns), None)
    if ccn is None or "provider_resides_in_hospital" not in df.columns:
        return set()
    df["cms_certification_number"] = normalize_ccn_any(df[ccn])
    df["provider_resides_in_hospital"] = bool_from_any(df["provider_resides_in_hospital"]).fillna(False)
    return set(df.loc[df["provider_resides_in_hospital"]==True, "cms_certification_number"].dropna().unique())

def drop_hospital_ccns(df: pd.DataFrame, ccn_col="cms_certification_number") -> pd.DataFrame:
    drops = load_hosp_ccn_dropset()
    if not drops: return df
    return df[~df[ccn_col].isin(drops)].copy()

def drop_hospital_months(df: pd.DataFrame, ccn_col="cms_certification_number", month_col="month") -> pd.DataFrame:
    true_panel = load_hosp_panel_true()
    if true_panel.empty: return df
    key = true_panel.rename(columns={"date": month_col})
    m = df.merge(key.assign(_drop=1), how="left", on=[ccn_col, month_col])
    return m[m["_drop"].isna()].drop(columns="_drop")