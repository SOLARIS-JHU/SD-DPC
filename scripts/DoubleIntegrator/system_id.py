#!/usr/bin/env python
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from _stages import run_system_id  # noqa: E402

if __name__ == "__main__":
    run_system_id("double_integrator", SCRIPTS.parent / "case_studies" / "DoubleIntegrator")
