#!/usr/bin/env python3
"""
Master Benchmark Runner and Verification Suite
==============================================
Reproduces and verifies all 8 stages of the Scientific Machine Learning benchmark:
  Stage 1: Ground Truth Solver Verification (DOP853 O(dt^4) Convergence)
  Stage 2: Trajectory & Physical Validity (Parity Asymmetry, Mass Drift, Bounds)
  Stage 3: Local Stability & Jacobian Eigenvalues (Hopf Bifurcation Roots)
  Stage 4: Asymptotic Continuation & Relaxation Folds (Limit Cycles, Periods)
  Stage 5: Ergodic Invariants & Attractors (W_1 Transport, Lyapunov Spectra, D_KY)
  Stage 6: Capacity Scaling & Ablations (10^3 -> 10^6 Overfitting Analysis)
  Stage 7: Out-of-Distribution Bifurcation Generalization & 12-Regime Transfer
  Stage 8: Uncertainty Quantification & Statistical Robustness (Bootstrap CIs, Exact Tests)

Usage:
  python run_master_benchmark.py
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRATCH_DIR = os.path.join(CURRENT_DIR, "scratch")

def main():
    print("=" * 80)
    print("  SCIENTIFIC MACHINE LEARNING (SciML) BENCHMARK REPRODUCIBILITY SUITE")
    print("  'Beyond Trajectory Fitting: Preserving Bifurcation Manifolds,'")
    print("  'Ergodic Invariants, and Stoichiometric Conservation'")
    print("=" * 80)
    
    # Run the comprehensive master verification script
    audit_script = os.path.join(SCRATCH_DIR, "verify_all_master_metrics.py")
    if not os.path.exists(audit_script):
        print(f"Error: Could not find audit script at {audit_script}")
        sys.exit(1)
        
    print("\nExecuting Predefined 8-Stage Computational Audits...\n")
    import subprocess
    result = subprocess.run([sys.executable, audit_script], capture_output=False)
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("  REPRODUCIBILITY CHECK PASSED.")
        print("  Benchmark State: FROZEN across all 7 canonical seeds.")
        print("  Full scientific report: FINAL_SCIENTIFIC_BENCHMARK_REPORT.md")
        print("=" * 80)
    else:
        print("\nVerification failed with exit code:", result.returncode)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
