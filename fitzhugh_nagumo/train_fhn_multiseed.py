"""
==============================================================================
MULTI-SEED TRAINING ENGINE: FITZHUGH-NAGUMO SCIENTIFIC ML
==============================================================================
Trains Proposed FiLM SciML models across 7 canonical independent random seeds:
  Seeds: [42, 101, 777, 999, 1234, 2024, 5555]
Also trains an Unconstrained Baseline model (Seed 42) for ablation comparison.

Checkpoints are saved to:
  fitzhugh_nagumo/scratch/fhn_sciml_s{seed}.pt
  fitzhugh_nagumo/scratch/fhn_baseline_s42.pt
==============================================================================
"""

import os
import sys

# Prevent OpenBLAS/OMP thread exhaustion and memory errors on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import random
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from fhn_global_sciml import (
    ProposedFiLMFHNMLP,
    UnconstrainedBaselineFHNMLP,
    build_fhn_dataset,
    train_model,
    set_seed,
    DT
)

SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]


def main():
    print("=" * 90, flush=True)
    print("FITZHUGH-NAGUMO SCIENTIFIC MACHINE LEARNING: 7-SEED TRAINING ENGINE", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print("=" * 90, flush=True)

    scratch_dir = os.path.join(CURRENT_DIR, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    # 1. Define Training Parameter Grid (with Hopf Boundary Corridor Focus)
    train_I = [0.2, 0.28, 0.32, 0.35, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    train_a = [0.5, 0.7, 0.9]
    train_b = [0.6, 0.8, 1.0]
    eps = 0.08

    train_params = []
    for I in train_I:
        for a in train_a:
            for b in train_b:
                train_params.append([I, a, b, eps])

    # 2. Define Validation Parameter Grid (Interpolated / Held-out)
    val_I = [0.29, 0.33, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    val_a = [0.6, 0.8]
    val_b = [0.7, 0.9]
    val_params = []
    for I in val_I:
        for a in val_a:
            for b in val_b:
                val_params.append([I, a, b, eps])

    print(f"Generating datasets: {len(train_params)} train parameter sets, {len(val_params)} val sets...", flush=True)
    t0 = time.time()
    train_ds = build_fhn_dataset(train_params, traj_duration=100.0, dt=DT, num_ics_per_param=2)
    val_ds = build_fhn_dataset(val_params, traj_duration=100.0, dt=DT, num_ics_per_param=2)
    print(f"Dataset generated in {time.time() - t0:.2f} s.\n", flush=True)

    # 3. Train Proposed FiLM SciML Across All 7 Seeds (Enhanced Physics Residual)
    for seed in SEEDS:
        ckpt_path = os.path.join(scratch_dir, f"fhn_sciml_s{seed}.pt")
        print("-" * 75, flush=True)
        print(f"TRAINING ENHANCED PROPOSED FiLM SciML [SEED {seed}] -> {os.path.basename(ckpt_path)}", flush=True)
        print("-" * 75, flush=True)
        model = ProposedFiLMFHNMLP(hidden_dim=128, dt=DT)
        model = train_model(model, train_ds, val_ds, total_steps=1000, batch_size=64, lr=1.5e-3, seed=seed)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}\n", flush=True)

    # 4. Train Unconstrained Baseline SciML (Seed 42)
    base_ckpt = os.path.join(scratch_dir, "fhn_baseline_s42.pt")
    print("-" * 75, flush=True)
    print(f"TRAINING UNCONSTRAINED BASELINE SciML [SEED 42] -> {os.path.basename(base_ckpt)}", flush=True)
    print("-" * 75, flush=True)
    base_model = UnconstrainedBaselineFHNMLP(hidden_dim=128, dt=DT)
    base_model = train_model(base_model, train_ds, val_ds, total_steps=1000, batch_size=64, lr=1.5e-3, seed=42)
    torch.save(base_model.state_dict(), base_ckpt)
    print(f"Saved baseline checkpoint: {base_ckpt}\n", flush=True)


    print("=" * 90, flush=True)
    print("ALL 7 SEEDS + BASELINE TRAINED AND SAVED SUCCESSFULLY!", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
