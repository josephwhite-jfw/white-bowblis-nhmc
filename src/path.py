from pathlib import Path
import os

def repo_root() -> Path:
    p = Path.cwd()
    while not (p / "data").exists() and p != p.parent:
        p = p.parent
    return p

REPO = repo_root()
RAW_DIR = Path(os.getenv("NH_DATA_DIR", REPO / "data" / "raw")).resolve()
INTERIM = REPO / "data" / "interim"; INTERIM.mkdir(parents=True, exist_ok=True)
CLEAN   = REPO / "data" / "clean";   CLEAN.mkdir(parents=True, exist_ok=True)

PROV_DIR = RAW_DIR / "provider-info-files"; PROV_DIR.mkdir(parents=True, exist_ok=True)
OWN_DIR  = RAW_DIR / "ownership-files";     OWN_DIR.mkdir(parents=True, exist_ok=True)

# Canonical outputs
PROV_COMBINED = PROV_DIR / "provider_info_combined.csv"
HOSP_PANEL    = PROV_DIR / "provider_resides_in_hospital_panel.csv"
HOSP_BY_CCN   = PROV_DIR / "provider_resides_in_hospital_by_ccn.csv"

OWN_COMBINED  = OWN_DIR  / "ownership_combined.csv"

PBJ_MONTHLY   = INTERIM / "pbj_monthly_panel.csv"
PBJ_COVERAGE  = INTERIM / "pbj_monthly_coverage.csv"

SIG_LONG      = INTERIM / "facility_signatures_long.csv"
CHOW_LITE     = INTERIM / "ccn_chow_lite.csv"

MCR_WIDE      = INTERIM / "mcr_chow_provider_events_all.csv"
CHOW_XLSX     = INTERIM / "chow_agreement_tables.xlsx"

PANEL_FULL    = CLEAN   / "pbj_panel_with_chow_dummies.csv"
PANEL_ANALYTIC= CLEAN   / "pbj_panel_analytic.csv"