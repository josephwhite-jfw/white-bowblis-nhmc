#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR      = Path(os.getenv("NH_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
INTERIM_DIR  = PROJECT_ROOT / "data" / "interim"
CLEAN_DIR    = PROJECT_ROOT / "data" / "cleaned"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"

