import sys
import time
import json
import numpy as np
import torch

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
from scratch.test_x7_fix import (
    UnifiedLorenzStyleSciML_Fixed, generate_extended_training_data,
    train_seed, evaluate_x7, SEEDS
)
from train_and_evaluate_lorenz_style_sciml import generate_training_data as generate_original_data

# The 3 critical seeds that previously failed catastrophically
FAILING_SEEDS = [101, 777, 1234]

def run_controlled_comparison():
    print("=" * 80)
    print("CONTROLLED ABLATION: ISOLATING HORIZON (2.5s vs 1.5s) AND M0 FLOOR (1e-4 vs 0.5)")
    print("=" * 80)

    # 1. Dataset with short horizon (1.5s)
    ds_short = generate_original_data()
    # 2. Dataset with extended horizon (2.5s)
    ds_long = generate_extended_training_data(horizon=2.5)

    conditions = [
        {"name": "Condition A (Short 1.5s, Floor 0.5)", "ds": ds_short, "floor": 0.5},
        {"name": "Condition B (Long 2.5s, Floor 0.5)",  "ds": ds_long,  "floor": 0.5},
        {"name": "Condition C (Short 1.5s, Floor 1e-4)", "ds": ds_short, "floor": 1e-4},
        {"name": "Condition D (Long 2.5s, Floor 1e-4)",  "ds": ds_long,  "floor": 1e-4},
    ]

    all_results = {}

    for cond in conditions:
        c_name = cond["name"]
        print(f"\n--- Running {c_name} on critical seeds {FAILING_SEEDS} ---")
        cond_res = {}
        for s in FAILING_SEEDS:
            t0 = time.time()
            m = train_seed(s, cond["ds"], m0_floor=cond["floor"], total_steps=400)
            res = evaluate_x7(m)
            cond_res[str(s)] = res
            print(f"  Seed {s:4d} | Time: {time.time()-t0:.1f}s | Amp: {res['pred_amp']:7.1f} mM | Err: {res['err_pct']:6.2f}% | Status: {res['status']}")
        all_results[c_name] = cond_res

    print("\n" + "=" * 80)
    print("CONTROLLED ABLATION SUMMARY")
    print("=" * 80)
    print(f"{'Condition':36s} | {'Seed 101':18s} | {'Seed 777':18s} | {'Seed 1234':18s} | {'Failures'}")
    print("-" * 100)
    for c_name, res in all_results.items():
        s101 = f"{res['101']['status']} ({res['101']['err_pct']:.1f}%)"
        s777 = f"{res['777']['status']} ({res['777']['err_pct']:.1f}%)"
        s1234 = f"{res['1234']['status']} ({res['1234']['err_pct']:.1f}%)"
        n_fail = sum(1 for s in ['101', '777', '1234'] if res[s]['status'] != 'PASSED')
        print(f"{c_name:36s} | {s101:18s} | {s777:18s} | {s1234:18s} | {n_fail}/3 failed")

    with open("scratch/floor_vs_horizon_ablation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved scratch/floor_vs_horizon_ablation.json successfully!")

if __name__ == "__main__":
    run_controlled_comparison()
