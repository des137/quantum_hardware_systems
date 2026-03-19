#!/usr/bin/env python3
"""
Regenerate README.md from data/hardware_data.json.

This is a convenience wrapper around:
    python scripts/update_hardware_data.py --readme-only

Use this after manually editing hardware_data.json.
"""

import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.exit(
    subprocess.call(
        [sys.executable, str(repo_root / "scripts" / "update_hardware_data.py"), "--readme-only"],
        cwd=str(repo_root),
    )
)
