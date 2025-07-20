#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations
import argparse
import logging
import re
import sys
from io import TextIOWrapper
from pathlib import Path
from typing import Dict, Iterable, List
import pandas as pd

try:
    from tqdm.autonotebook import tqdm as _tqdm

    def tqdm(it, **kw):
        return _tqdm(it, **kw)
except ModuleNotFoundError:
    tqdm = lambda it, **kw: it  # type: ignore[misc]

# ------------------------------------------------------------------ #
# Repo paths
# ------------------------------------------------------------------ #
try:
    from src.utils.paths import RAW_DIR, INTERIM_DIR
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[2]
    RAW_DIR = ROOT / "data" / "raw"
    INTERIM_DIR = ROOT / "data" / "interim"

ZIP_DIR = RAW_DIR / "nh-compare"
OUT_PARQUET = INTERIM_DIR / "ownership_combined.parquet"
OUT_CSV = OUT_PARQUET.with_suffix(".csv")

# ------------------------------------------------------------------ #
# Column-standardisation map (same as your notebook)
# ------------------------------------------------------------------ #
COLUMN_MAP: Dict[str, str] = {
    # provider id
    "cms certification number (ccn)": "cms_certification_number",
    "federal provider number": "cms_certification_number",
    "provnum": "cms_certification_number",
    # provider name
    "provider name": "provider_name",
    "provname": "provider_name",
    # role
    "role played by owner or manager in facility": "role",
    "role played by owner in facility": "role",
    "role of owner or manager": "role",
    "owner role": "role",
    "role desc": "role",
    # ownership meta
    "owner type": "owner_type",
    "owner name": "owner_name",
    "ownership percentage": "ownership_percentage",
    "owner percentage": "ownership_percentage",
    "association date": "association_date",
    # processing date
    "processing date": "processing_date",
    "processingdate": "processing_date",
    "process date": "processing_date",
    "processdate": "processing_date",
    "filedate": "processing_date",
}
TARGET_COLS = list(set(COLUMN_MAP.values()))

# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _clean_col(raw: str) -> str | None:
    raw = re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()
    return COLUMN_MAP.get(raw)


def _iter_inner_csvs(zip_files: Iterable[Path], chunksize: int | None) -> Iterable[pd.DataFrame]:
    import zipfile

    pat_old = re.compile(r"ownership_download", re.I)
    pat_new = re.compile(r"NH_Ownership_", re.I)

    for outer in tqdm(zip_files, desc="ZIP archives"):
        with zipfile.ZipFile(outer) as z_outer:
            for inner_name in z_outer.namelist():
                # choose correct owner file inside the inner zip
                mm_yyyy = re.search(r"_(\d{2})_(\d{4})\.zip$", inner_name)
                if not mm_yyyy:
                    continue  # ignore scores/inspection zips

                month, year = map(int, mm_yyyy.groups())
                is_new_fmt = (year > 2020) or (year == 2020 and month >= 8)

                if (is_new_fmt and not pat_new.search(inner_name)) or (
                    not is_new_fmt and not pat_old.search(inner_name)
                ):
                    continue  # wrong file inside this inner zip

                with z_outer.open(inner_name) as inner_bytes:
                    with zipfile.ZipFile(inner_bytes) as z_inner:
                        for member in z_inner.namelist():
                            if not member.lower().endswith(".csv"):
                                continue
                            logging.info("Reading %s in %s", member, outer.name)
                            with z_inner.open(member) as fbytes:
                                ftxt: TextIOWrapper = TextIOWrapper(fbytes, encoding="utf-8")
                                if chunksize:
                                    for chunk in pd.read_csv(
                                        ftxt, dtype=str, chunksize=chunksize
                                    ):
                                        yield chunk, month, year, member
                                else:
                                    yield pd.read_csv(ftxt, dtype=str), month, year, member


def _standardise(df_raw: pd.DataFrame) -> pd.DataFrame:
    # map / filter columns
    new_cols = {c: _clean_col(c) for c in df_raw.columns}
    df = pd.DataFrame()
    for old, new in new_cols.items():
        if new in TARGET_COLS:
            df[new] = df_raw[old]
    # add missing empties
    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _fill_ccn(df: pd.DataFrame) -> pd.DataFrame:
    ccn_col, name_col = "cms_certification_number", "provider_name"
    mapping = (
        df.dropna(subset=[ccn_col])
        .groupby(name_col, observed=True)[ccn_col]
        .agg(lambda s: set(s))
    )
    unamb = mapping[mapping.str.len() == 1].apply(lambda s: next(iter(s)))
    mask = df[ccn_col].isna()
    before = mask.sum()
    df.loc[mask, ccn_col] = df.loc[mask, name_col].map(unamb)
    after = df[ccn_col].isna().sum()
    logging.info("CCN fill: %s → %s missing (filled %s)", before, after, before - after)
    return df


# ------------------------------------------------------------------ #
# Main builders
# ------------------------------------------------------------------ #
def build_df(chunksize: int | None = None) -> pd.DataFrame:
    zip_files = sorted(ZIP_DIR.glob("nh_archive_*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No ZIPs found under {ZIP_DIR}")

    frames: List[pd.DataFrame] = []
    for raw_df, mm, yyyy, src in _iter_inner_csvs(zip_files, chunksize):
        df = _standardise(raw_df)
        df["source_file"] = Path(src).stem
        df["month"] = mm
        df["year"] = yyyy
        df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    combined = _fill_ccn(combined)
    logging.info("Combined shape %s", combined.shape)
    return combined


def write_df(df: pd.DataFrame, to_parquet: bool = True) -> Path:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_PARQUET if to_parquet else OUT_CSV
    if to_parquet:
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    logging.info("Wrote %s (%.2f MB)", out, out.stat().st_size / 1_048_576)
    return out


def build_and_write_ownership(
    chunksize: int | None = None, to_parquet: bool = True
) -> pd.DataFrame:
    df = build_df(chunksize=chunksize)
    write_df(df, to_parquet=to_parquet)
    return df


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ownership_combined table.")
    p.add_argument("--csv", action="store_true", help="write CSV instead of Parquet")
    p.add_argument("--chunksize", type=int, default=None, help="rows per chunk")
    return p.parse_args()


def main() -> None:
    a = _args()
    build_and_write_ownership(chunksize=a.chunksize, to_parquet=not a.csv)


if __name__ == "__main__":
    main()

