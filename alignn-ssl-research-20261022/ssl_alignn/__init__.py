"""Meeting implementation; upstream ALIGNN files are imported, never edited."""
from pathlib import Path
import sys

UPSTREAM = Path(__file__).resolve().parents[1] / "vendor" / "alignn"
if not (UPSTREAM / "alignn" / "models" / "alignn_atomwise.py").is_file():
    raise RuntimeError(f"Original ALIGNN source missing: {UPSTREAM}")
sys.path.insert(0, str(UPSTREAM))
