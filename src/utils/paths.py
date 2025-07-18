#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parents[2]  # → repo root
RAW_DIR        = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR    = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR  = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR     = PROJECT_ROOT / "outputs"

