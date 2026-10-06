"""
================================================================================
MASTER ORCHESTRATION PIPELINE: FITZHUGH-NAGUMO SciML EXPERIMENTS
================================================================================
Runs:
  1. Multi-Seed Replication & Attractor Basin Retention (fhn_multiseed_replication.py)
  2. Dense Bifurcation Horizon & Curvature Tracking (fhn_bifurcation_analysis.py)
  3. Synchronizes all code, checkpoints, JSONs, and figures to:
     c:\\Users\\satya\\Downloads\\FitzHugh Nagumo\\
================================================================================
"""

import os
import sys
import subprocess
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEST_DIR = r"c:\Users\satya\Downloads\FitzHugh Nagumo"

def run_cmd(cmd):
    print(f"\n>>> Running: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=CURRENT_DIR)
    if res.returncode != 0:
        print(f"Error executing command: {cmd}", file=sys.stderr)
        sys.exit(res.returncode)

def sync_to_destination():
    print(f"\n[Mirroring] Syncing {CURRENT_DIR} -> {DEST_DIR}...", flush=True)
    os.makedirs(DEST_DIR, exist_ok=True)
    
    # PowerShell copy
    ps_cmd = [
        "powershell", "-NoProfile", "-Command",
        f"Copy-Item -Path '{CURRENT_DIR}\\*' -Destination '{DEST_DIR}\\' -Recurse -Force"
    ]
    subprocess.run(ps_cmd, check=True)
    print("[Mirroring] Sync completed successfully!\n", flush=True)

def main():
    py_exe = sys.executable
    print("=" * 80)
    print("STARTING FITZHUGH-NAGUMO FULL EXPERIMENTAL SUITE")
    print("=" * 80)

    # 1. Multi-Seed Replication
    run_cmd([py_exe, os.path.join(CURRENT_DIR, "fhn_multiseed_replication.py")])

    # 2. Bifurcation Analysis & Curvature Tracking
    run_cmd([py_exe, os.path.join(CURRENT_DIR, "fhn_bifurcation_analysis.py")])

    # 3. Lorenz-Style Climate & Ergodic Validation (4 Pillars)
    run_cmd([py_exe, os.path.join(CURRENT_DIR, "fhn_climate_and_ergodic_validation.py")])

    # 4. Synchronize
    sync_to_destination()


    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETED AND SYNCHRONIZED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
