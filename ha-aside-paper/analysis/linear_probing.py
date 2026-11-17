"""Compatibility entrypoint for real-checkpoint instruction/data probing."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.probe_real_models import main

if __name__ == "__main__":
    main()
