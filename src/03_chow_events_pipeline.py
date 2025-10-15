# 03_chow_events_pipeline.py
import pandas as pd, numpy as np, re
from pathlib import Path
from config import PROV_COMBINED, MCR_CONTROLS_CSV, CHOW_EVENTS_MCR, CHOW_EVENTS_OWNERSHIP, CHOW_OVERLAP_AGREE

# CHOW from ownership/provider-info (flag transitions in ownership_type, chain, name; conservative)
def ownership_chow_from_controls_and_provider():
    prov = pd.read_csv(PROV_COMBINED, parse_dates=["date"], dtype={"cms_certification_number":"string"})
    # monthly state exists in MCR; we’ll align on month
    prov["month"] = prov["date"].dt.to_period("M").dt.to_timestamp("s")
    prov = prov[["cms_certification_number","month","provider_name","sff_class","ccrc_facility"]] \
             .drop_duplicates(["cms_certification_number","month"]).sort_values(["cms_certification_number","month"])
    mcr = pd.read_csv(MCR_CONTROLS_CSV, parse_dates=["month"], dtype={"cms_certification_number":"string"})
    base = mcr.merge(prov, on=["cms_certification_number","month"], how="left")

    # Detect potential CHOW: changes in ownership_type or is_chain or large name change
    base["ownership_type_prev"] = base.groupby("cms_certification_number")["ownership_type"].shift(1)
    base["is_chain_prev"]       = base.groupby("cms_certification_number")["is_chain"].shift(1)
    base["provider_name_prev"]  = base.groupby("cms_certification_number")["provider_name"].shift(1)

    def big_name_change(a,b):
        if pd.isna(a) or pd.isna(b): return False
        a,b=str(a).lower(), str(b).lower()
        if a==b: return False
        # crude: share of common tokens < 0.5
        ta, tb = set(re.split(r"[^a-z0-9]+", a)), set(re.split(r"[^a-z0-9]+", b))
        ta.discard(""); tb.discard("")
        inter = len(ta & tb); denom = max(1,len(ta|tb))
        return (inter/denom) < 0.5

    base["flag_own_change"] = (base["ownership_type"].ne(base["ownership_type_prev"]) & base["ownership_type"].notna() & base["ownership_type_prev"].notna())
    base["flag_chain_change"]= (base["is_chain"].ne(base["is_chain_prev"]) & base["is_chain"].notna() & base["is_chain_prev"].notna())
    base["flag_name_change"] = base.apply(lambda r: big_name_change(r["provider_name"], r["provider_name_prev"]), axis=1)

    events = base.loc[(base["flag_own_change"] | base["flag_chain_change"] | base["flag_name_change"]),
                      ["cms_certification_number","month"]].drop_duplicates()
    events = events.rename(columns={"month":"event_month"})
    events["source"]="ownership_controls"
    events.to_csv(CHOW_EVENTS_OWNERSHIP, index=False)
    print(f"[CHOW/OWN] events={len(events):,} → {CHOW_EVENTS_OWNERSHIP}")
    return events

# CHOW from MCR events: first month of recorded change counts (you may refine if you have detailed MCR events)
def chow_from_mcr_changes():
    m = pd.read_csv(MCR_CONTROLS_CSV, parse_dates=["month"], dtype={"cms_certification_number":"string"})
    m = m.sort_values(["cms_certification_number","month"])
    # define an “ever changed” month = first month ownership_type differs from the initial spell
    m["own_prev"]  = m.groupby("cms_certification_number")["ownership_type"].shift(1)
    m["chain_prev"]= m.groupby("cms_certification_number")["is_chain"].shift(1)
    m["chg"] = (m["ownership_type"].ne(m["own_prev"]) & m["ownership_type"].notna() & m["own_prev"].notna()) | \
               (m["is_chain"].ne(m["chain_prev"]) & m["is_chain"].notna() & m["chain_prev"].notna())
    first = (m.loc[m["chg"], ["cms_certification_number","month"]]
               .groupby("cms_certification_number", as_index=False)
               .first().rename(columns={"month":"event_month"}))
    first["source"]="mcr_controls"
    first.to_csv(CHOW_EVENTS_MCR, index=False)
    print(f"[CHOW/MCR] events={len(first):,} → {CHOW_EVENTS_MCR}")
    return first

# Overlap + agreement label and chosen change_month
def reconcile_and_save():
    own = ownership_chow_from_controls_and_provider()
    mcr = chow_from_mcr_changes()
    merged = own.merge(mcr, on="cms_certification_number", how="outer", suffixes=("_own","_mcr"))
    def pick(row):
        a, b = row.get("event_month_own"), row.get("event_month_mcr")
        label, change = "mismatch", pd.NaT
        if pd.isna(a) and pd.isna(b):
            label, change = "match_0", pd.NaT
        elif pd.notna(a) and pd.notna(b):
            pa, pb = pd.Period(a, "M"), pd.Period(b, "M")
            if abs((pa - pb).n) <= 6:
                label, change = "match_1_within_6m", (pa if pd.notna(a) else pb).to_timestamp("s")
            else:
                label, change = "match_1_diff_month", pd.NaT
        else:
            # one source has event; we treat as mismatch for audit; final panel keeps only match_0 & match_1_within_6m
            if pd.notna(a) or pd.notna(b):
                label="mismatch"
        return pd.Series({"agreement":label,"change_month":change})
    out = pd.concat([merged, merged.apply(pick, axis=1)], axis=1)
    out.to_csv(CHOW_OVERLAP_AGREE, index=False)
    print(f"[CHOW] overlap rows={len(out):,} → {CHOW_OVERLAP_AGREE}")

if __name__ == "__main__":
    reconcile_and_save()
