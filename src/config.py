# 00_config.py
from pathlib import Path
import os

# Project roots
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").is_dir() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

REPO = PROJECT_ROOT
DATA = REPO / "data"
RAW  = Path(os.getenv("NH_DATA_DIR", DATA / "raw")).resolve()
INT  = DATA / "interim"
CLN  = DATA / "clean"
for d in (INT, CLN): d.mkdir(parents=True, exist_ok=True)

# Raw locations (self-contained workflow expects only these)
NH_COMPARE_DIR = RAW / "nh-compare"                 # contains nh_archive_*.zip (yearly) with monthly inner zips
MCR_DIR        = RAW / "medicare-cost-reports"      # mcr_flatfile_20??.(sas7bdat|csv)
PBJ_RAW_DIR    = RAW / "pbj-nurse"

# Provider-info outputs (freshly built here)
PROV_DIR       = RAW / "provider-info-files"
PROV_DIR.mkdir(parents=True, exist_ok=True)

PROV_COMBINED  = PROV_DIR / "provider_info_combined.csv"
HOSP_PANEL_CSV = PROV_DIR / "provider_resides_in_hospital_panel.csv"
HOSP_LATEST_CSV= PROV_DIR / "provider_resides_in_hospital_by_ccn.csv"

# MCR monthly controls
MCR_CONTROLS_CSV = CLN / "mcr_controls_monthly.csv"

# CHOW artifacts (self-contained; we’ll compute both sources where possible)
CHOW_EVENTS_MCR      = CLN / "mcr_chow_events.csv"
CHOW_EVENTS_OWNERSHIP= CLN / "ownership_chow_events.csv"
CHOW_OVERLAP_AGREE   = CLN / "chow_overlap_agreement.csv"

# PBJ monthly (built or reused)
PBJ_MONTHLY_CSV = CLN / "pbj_monthly_panel.csv"

# Final panel
FINAL_PANEL_CSV = CLN / "pbj_panel_with_chow_dummies.csv"

print("[config] RAW =", RAW)