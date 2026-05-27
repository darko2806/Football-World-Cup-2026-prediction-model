"""
Pomocne funkcije za ucitavanje podataka i opste operacije.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
