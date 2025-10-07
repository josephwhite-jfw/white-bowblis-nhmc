# nhmc/chow.py
# Ownership CHOW (signatures→transitions→lite) + MCR CHOW + agreement xlsx

from __future__ import annotations
import re, json
from pathlib import Path
import numpy as np
import pandas as pd

from .paths import OWN_COMBINED, INTERIM, SIG_LONG, CHOW_LITE, MCR_WIDE, CHOW_XLSX, RAW_DIR
from .utils import normalize_ccn_any, to_monthstart, drop_hospital_ccns

INTERIM.mkdir(parents=True, exist_ok=True)
CUTOFF_DATE = pd.Timestamp("2017-01-01")
LEVEL_PRIORITY = ["indirect","direct","partnership"]

def _clean_owner_name(s: str) -> str:
    if pd.isna(s) or not str(s).strip(): return ""
    x = str(s).upper()
    x = re.sub(r"[.,&/()\-']", " ", x)
    x = re.sub(r"\b(INC|CORP|LLC|L\.L\.C\.|LP|LLP|CO|COMPANY|HOLDINGS?|PARTNERS?|CAPITAL|INVESTMENTS?|TRUST|GROUP)\b", "", x)
    x = re.sub(r"\s+"," ",x).strip()
    return x

def _level_bucket(role: str) -> str:
    s = str(role).lower()
    if "indirect" in s: return "indirect"
    if "direct"   in s: return "direct"
    if "partner"  in s: return "partnership"
    return ""

def _normalize_weights(df_block: pd.DataFrame) -> pd.DataFrame:
    g = df_block.copy()
    g["ownership_percentage"] = pd.to_numeric(g["ownership_percentage"], errors="coerce")
    agg = g.groupby("owner_name_norm", as_index=False)["ownership_percentage"].sum(min_count=1)
    if agg["ownership_percentage"].notna().any() and agg["ownership_percentage"].fillna(0).sum()>0:
        vec = agg[agg["ownership_percentage"].notna()].copy()
        vec["ownership_percentage"] = vec["ownership_percentage"] * (100.0/vec["ownership_percentage"].sum())
    else:
        owners = agg["owner_name_norm"].tolist()
        if not owners: return pd.DataFrame({"owner_name_norm":[],"ownership_percentage":[]})
        eq = 100.0/len(owners)
        vec = pd.DataFrame({"owner_name_norm":owners,"ownership_percentage":[eq]*len(owners)})
    vec["ownership_percentage"] = vec["ownership_percentage"].round(1)
    tot = vec["ownership_percentage"].sum()
    if tot>0:
        vec["ownership_percentage"] = (vec["ownership_percentage"]*(100.0/tot)).round(1)
    return vec[vec["ownership_percentage"]>0].reset_index(drop=True)

def _pct_overlap(prev: dict, curr: dict) -> float:
    names = set(prev)|set(curr)
    ov = sum(min(prev.get(n,0.0), curr.get(n,0.0)) for n in names)
    return max(0.0, min(ov/100.0, 1.0))  # 0..1

def build_ownership_chow() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(OWN_COMBINED, low_memory=False)
    for c in ["cms_certification_number","owner_name","role","association_date"]:
        if c not in df.columns:
            raise ValueError(f"[chow] OWN combined missing column: {c}")
    df["cms_certification_number"] = normalize_ccn_any(df["cms_certification_number"])
    df["association_date"] = pd.to_datetime(df["association_date"], errors="coerce")
    df = df.dropna(subset=["association_date"]).copy()
    df["owner_name_norm"] = df["owner_name"].map(_clean_owner_name)
    df["level"] = df["role"].map(_level_bucket)

    # snapshots
    snaps = []
    for (ccn, adate), g in df.groupby(["cms_certification_number","association_date"], sort=True):
        chosen = None
        for lvl in LEVEL_PRIORITY:
            gl = g[g["level"]==lvl]
            if len(gl):
                chosen = (lvl, gl); break
        if chosen is None: 
            continue
        lvl, gl = chosen
        vec = _normalize_weights(gl[["owner_name_norm","ownership_percentage"]])
        wm = dict(zip(vec["owner_name_norm"], vec["ownership_percentage"].astype(float)))
        snaps.append({"cms_certification_number":ccn,"association_date":adate,"source_level":lvl,"weights":wm})
    snaps_df = pd.DataFrame(snaps).sort_values(["cms_certification_number","association_date"]).reset_index(drop=True)

    # regimes → transitions
    rows, long_rows = [], []
    for ccn, g in snaps_df.groupby("cms_certification_number", sort=True):
        g = g.sort_values("association_date").reset_index(drop=True)
        if g.empty: continue
        prev = g.loc[0,"weights"]
        start = g.loc[0,"association_date"]; group_n = 1
        long_rows.append({"cms_certification_number": ccn, "group_n":group_n, "start":start, "end":pd.NaT,
                          "source_level":g.loc[0,"source_level"],
                          "names_list":json.dumps(list(prev.keys())), "pcts_list":json.dumps(list(prev.values()))})
        for i in range(1, len(g)):
            ov = _pct_overlap(prev, g.loc[i,"weights"])
            turnover = 1.0 - ov
            if turnover >= 0.50:
                long_rows[-1]["end"] = g.loc[i-1,"association_date"]
                group_n += 1
                start = g.loc[i,"association_date"]
                long_rows.append({"cms_certification_number": ccn, "group_n":group_n, "start":start, "end":pd.NaT,
                                  "source_level":g.loc[i,"source_level"],
                                  "names_list":json.dumps(list(g.loc[i,"weights"].keys())),
                                  "pcts_list": json.dumps(list(g.loc[i,"weights"].values()))})
            else:
                # continue current regime
                pass
            prev = g.loc[i,"weights"]
        long_rows[-1]["end"] = g.loc[len(g)-1,"association_date"]

        # transitions
        if len(g)>=2:
            for i in range(1, len(g)):
                prev_w = g.loc[i-1,"weights"]; curr_w = g.loc[i,"weights"]
                ov = _pct_overlap(prev_w, curr_w)
                turn_pct = round(100.0*(1.0-ov), 1)
                is_chow = (g.loc[i,"association_date"] >= CUTOFF_DATE) and (turn_pct >= 50.0)
                rows.append({"cms_certification_number":ccn,
                             "to_start":g.loc[i,"association_date"],
                             "turnover_pct":turn_pct, "is_chow":bool(is_chow)})
    long_df = pd.DataFrame(long_rows).sort_values(["cms_certification_number","group_n"]).reset_index(drop=True)

    # lite summary
    trans = pd.DataFrame(rows).sort_values(["cms_certification_number","to_start"]).reset_index(drop=True)
    summary = (trans.loc[trans["is_chow"]]
                    .groupby("cms_certification_number", as_index=False)["to_start"]
                    .agg(list)
                    .rename(columns={"to_start":"chow_dates"}))
    all_ccns = long_df["cms_certification_number"].drop_duplicates().to_frame()
    lite = all_ccns.merge(summary, on="cms_certification_number", how="left")
    lite["num_chows"] = lite["chow_dates"].apply(lambda x: 0 if not isinstance(x, list) else len(x)).astype(int)
    # expand a few columns for convenience
    for i in range(1, 6):
        lite[f"chow_date_{i}"] = lite["chow_dates"].apply(lambda L: pd.to_datetime(L[i-1]).strftime("%Y-%m-%d") if isinstance(L, list) and len(L)>=i else np.nan)
    lite["is_chow"] = (lite["num_chows"]>0).astype(int)
    lite = lite.drop(columns=["chow_dates"])

    long_df.to_csv(SIG_LONG, index=False)
    lite.to_csv(CHOW_LITE, index=False)
    print(f"[chow:own] signatures → {SIG_LONG} (rows={len(long_df):,})")
    print(f"[chow:own] lite       → {CHOW_LITE} (rows={len(lite):,})")
    return long_df, lite

def build_mcr_chow() -> pd.DataFrame:
    MCR_DIR = RAW_DIR / "medicare-cost-reports"
    patt = ["mcr_flatfile_20??.csv","mcr_flatfile_20??.sas7bdat","mcr_flatfile_20??.xpt","mcr_flatfile_20??.XPT"]
    files = []
    for p in patt: files += list(MCR_DIR.glob(p))
    if not files:
        raise FileNotFoundError(f"No MCR flatfiles in {MCR_DIR}")
    frames = []
    # read minimally (3 columns)
    TARGET = {"PRVDR_NUM","S2_2_CHOW","S2_2_CHOWDATE"}
    for fp in sorted(files):
        try:
            if fp.suffix.lower()==".csv":
                df = pd.read_csv(fp, dtype=str, low_memory=False)
            elif fp.suffix.lower()==".sas7bdat":
                df = pd.read_sas(fp, format="sas7bdat", encoding="latin1")
            else:
                df = pd.read_sas(fp, format="xport", encoding="latin1")
            up = {c: c.upper().strip() for c in df.columns}
            rev = {v:k for k,v in up.items()}
            want = [rev[c] for c in TARGET if c in rev]
            sub = df[want].copy()
            sub.columns = [c.upper().strip() for c in sub.columns]
            for t in TARGET:
                if t not in sub.columns: sub[t] = pd.NA
            sub = sub[["PRVDR_NUM","S2_2_CHOW","S2_2_CHOWDATE"]]
            frames.append(sub)
            print(f"[mcr] read {fp.name} rows={len(sub):,}")
        except Exception as e:
            print(f"[mcr] warn {fp.name}: {e}")

    mcr = pd.concat(frames, ignore_index=True)
    mcr["cms_certification_number"] = normalize_ccn_any(mcr["PRVDR_NUM"])
    # SAS numeric date fallback
    num = pd.to_numeric(mcr["S2_2_CHOWDATE"], errors="coerce")
    base = pd.Timestamp("1960-01-01")
    dt = base + pd.to_timedelta(num, unit="D")
    dt2 = pd.to_datetime(mcr["S2_2_CHOWDATE"], errors="coerce")
    mcr["event_date"] = dt.where(num.notna(), dt2)
    mcr = mcr.dropna(subset=["cms_certification_number"]).copy()
    mcr = drop_hospital_ccns(mcr, ccn_col="cms_certification_number")

    events = (mcr.loc[mcr["event_date"].notna(), ["cms_certification_number","event_date"]]
                 .drop_duplicates()
                 .assign(event_month=lambda d: to_monthstart(d["event_date"])))
    post = events.loc[events["event_month"] >= CUTOFF_DATE].copy()

    if post.empty:
        wide = pd.DataFrame(columns=["cms_certification_number","n_chow","is_chow"])
    else:
        post["ord"] = post.groupby("cms_certification_number")["event_month"].rank(method="first").astype(int)
        wide = post.pivot(index="cms_certification_number", columns="ord", values="event_month").reset_index()
        if wide.shape[1]>1:
            wide.columns = ["cms_certification_number"] + [f"chow_{i}_date" for i in wide.columns[1:]]
        else:
            wide.columns = ["cms_certification_number"]
        wide["n_chow"] = post.groupby("cms_certification_number")["event_month"].size().values
        wide["is_chow"] = (wide["n_chow"]>0).astype("int8")

    # stringify dates for CSV
    for c in [x for x in wide.columns if re.fullmatch(r"chow_\d+_date", x)]:
        wide[c] = pd.to_datetime(wide[c], errors="coerce").dt.strftime("%Y-%m-%d")
    wide = wide.sort_values("cms_certification_number").reset_index(drop=True)
    wide.to_csv(MCR_WIDE, index=False)
    print(f"[chow:mcr] provider-wide → {MCR_WIDE} (rows={len(wide):,})")
    return wide

def build_chow_everything() -> None:
    _, lite = build_ownership_chow()
    mcrw = build_mcr_chow()

    # quick agreement table → Excel
    merged = (lite[["cms_certification_number","num_chows"]]
              .rename(columns={"num_chows":"own_num"})
              .merge(mcrw[["cms_certification_number","n_chow"]], on="cms_certification_number", how="outer"))
    merged["own_cat"] = merged["own_num"].fillna(0).astype(int).clip(0, 2).astype(str).replace({"2":"2+"})
    merged["mcr_cat"] = merged["n_chow"].fillna(0).astype(int).clip(0, 2).astype(str).replace({"2":"2+"})
    ctab = pd.crosstab(merged["own_cat"], merged["mcr_cat"]).reindex(index=["0","1","2+"], columns=["0","1","2+"], fill_value=0)
    with pd.ExcelWriter(CHOW_XLSX) as xw:
        ctab.to_excel(xw, sheet_name="Crosstab_0_1_2plus")
        merged.to_excel(xw, sheet_name="Merge", index=False)
    print(f"[chow] agreement xlsx → {CHOW_XLSX}")
