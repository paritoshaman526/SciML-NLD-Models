"""
======================================================================================
MASTER EXPERIMENTAL RUNNER: VAN DER POL OSCILLATOR SciML PIPELINE
======================================================================================
Executes:
1. Step 1: 7-Seed Multi-Seed Training & Publication Table 1 Benchmark
2. Step 2: Dense-Grid Bifurcation Horizon & Curvature Tracking (15 Points)
3. Step 3: Tier-2 4-Pillar Climate & Ergodic Invariance Validation Suite
======================================================================================
"""

import os
import sys
import subprocess
import time

VDP_DIR = r"c:\Users\satya\Downloads\Van Der Pol\vanderpol"
PYTHON_EXE = r"c:\Users\satya\Downloads\Van Der Pol\.venv\Scripts\python.exe"

def run_script(script_name):
    script_path = os.path.join(VDP_DIR, script_name)
    print(f"\n{'='*90}\n>>> EXECUTING: {script_name}\n{'='*90}", flush=True)
    t0 = time.time()
    res = subprocess.run([PYTHON_EXE, script_path], cwd=VDP_DIR, capture_output=False)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"ERROR: {script_name} failed with exit code {res.returncode}", flush=True)
        sys.exit(res.returncode)
    print(f"\n>>> FINISHED: {script_name} in {elapsed:.1f} s", flush=True)

def main():
    print("=" * 100)
    print("STARTING FULL VAN DER POL EXPERIMENTAL SUITE (7-SEED CANONICAL PIPELINE)")
    print("=" * 100)
    
    t_start = time.time()
    run_script("vdp_train_and_table1.py")
    run_script("vdp_bifurcation_and_climate.py")
    total_time = time.time() - t_start

    print("\n" + "=" * 100)
    print(f"ALL VAN DER POL EXPERIMENTS COMPLETED SUCCESSFULLY IN {total_time:.1f} s ({total_time/60:.2f} min)!")
    print("=" * 100)

if __name__ == "__main__":
    main()
