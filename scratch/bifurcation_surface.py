"""
Phase 3: Bifurcation Surface Reconstruction
RK4 full grid (39x35x5), neural coarse sub-grid (15x15x5).
kd2: 100-6000, kd3: 0.1-2.2, ka: [1.0, 2.5, 5.0, 8.5, 12.0]
"""
import sys, os, json, time
import numpy as np
import torch

PROJ    = r"c:\Users\satya\Downloads\Van Der Pol"
SCRATCH = os.path.join(PROJ, "scratch")
for p in [PROJ, SCRATCH]:
    if p not in sys.path:
        sys.path.insert(0, p)

from biochemical_global_sciml import rk4_step, DEFAULT_GUESS, DT, set_seed
from test_x7_fix import UnifiedLorenzStyleSciML_Fixed, generate_extended_training_data, sample_batch
from train_and_evaluate_lorenz_style_sciml import PHYSICAL_CTOT

# Grid
KD2_VALS = np.concatenate([np.linspace(100, 800, 12), np.linspace(800, 6000, 28)[1:]])  # 39 pts
KD3_VALS = np.linspace(0.1, 2.2, 35)
KA_SLICES = [1.0, 2.5, 5.0, 8.5, 12.0]
KP = 1.0

# RK4 timing
WARMUP_STEPS  = int(5.0 / DT)
MEASURE_STEPS = int(5.0 / DT)
EARLY_STEPS   = int(2.0 / DT)
AMP_THRESHOLD = 20.0
DIV_GUARD     = 1e5

# Neural coarse grid
NN_KD2   = np.linspace(100, 6000, 15)
NN_KD3   = np.linspace(0.1, 2.2, 15)
NN_STEPS = 2000

def measure_rk4_amplitude(params):
    p = np.array(params, dtype=np.float64)
    y = DEFAULT_GUESS.copy()
    for _ in range(WARMUP_STEPS):
        y = rk4_step(y, DT, p)
        if not np.all(np.isfinite(y)) or np.any(np.abs(y) > DIV_GUARD):
            return -1.0, "DIVERGED"
    sig = np.zeros(MEASURE_STEPS)
    for i in range(MEASURE_STEPS):
        y = rk4_step(y, DT, p)
        if not np.all(np.isfinite(y)) or np.any(np.abs(y) > DIV_GUARD):
            return -1.0, "DIVERGED"
        sig[i] = y[2]
        if i == EARLY_STEPS - 1:
            early_amp = (sig[:i+1].max() - sig[:i+1].min()) / 2.0
            if early_amp < 1.0:
                return float(early_amp), "SS"
    last_half = sig[MEASURE_STEPS // 2:]
    amp = (last_half.max() - last_half.min()) / 2.0
    return float(amp), "LC" if amp > AMP_THRESHOLD else "SS"

def load_best_model():
    model = UnifiedLorenzStyleSciML_Fixed(hidden_dim=128, dt=DT, m0_floor=1e-4)
    ckpt  = os.path.join(PROJ, "unified_sciml_fixed_s42.pt")
    if os.path.exists(ckpt):
        try:
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print("  Loaded checkpoint: unified_sciml_fixed_s42.pt", flush=True)
            model.eval(); return model
        except RuntimeError:
            print("  Mismatch — training fresh...", flush=True)
    else:
        print("  No checkpoint — training fresh...", flush=True)
    dataset = generate_extended_training_data(horizon=2.5)
    set_seed(42)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400, eta_min=1e-5)
    w = torch.tensor([1.0, 1.0, 2.0, 5.0]) / model.state_scales
    for step in range(1, 401):
        model.train()
        K = int(np.interp(step, [1, 200, 400], [2, 6, 8]))
        s0_b, p_b, future_b, _, _ = sample_batch(dataset, K=K, batch_size=64)
        c0 = s0_b[:, 0:1] + s0_b[:, 1:2] + s0_b[:, 2:3]
        curr = s0_b; loss = 0.0
        for k in range(K):
            curr = model(curr, p_b, target_ctot=c0)
            loss += torch.mean(((curr - future_b[:, k]) * w) ** 2)
        loss = loss / K
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step(); sched.step()
        if step % 100 == 0:
            print(f"    Step {step}/400  loss={loss.item():.4e}", flush=True)
    model.eval()
    torch.save(model.state_dict(), ckpt)
    print("  Saved checkpoint.", flush=True)
    return model

def measure_neural_fast(model, params):
    p = np.array(params, dtype=np.float64)
    p_t = torch.tensor(params[:4], dtype=torch.float32).unsqueeze(0)
    y = DEFAULT_GUESS.copy()
    for _ in range(int(2.5 / DT)):
        y = rk4_step(y, DT, p)
        if not np.all(np.isfinite(y)) or np.any(np.abs(y) > DIV_GUARD):
            return -1.0, "DIVERGED"
    curr = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    c_tot = curr[:, 0] + curr[:, 1] + curr[:, 2]
    sig = []
    with torch.no_grad():
        for _ in range(NN_STEPS):
            curr = model(curr, p_t, target_ctot=c_tot)
            if not torch.all(torch.isfinite(curr)) or curr.abs().max() > DIV_GUARD:
                return -1.0, "DIVERGED"
            sig.append(float(curr[0, 2]))
    sig = np.array(sig)
    amp = (sig[NN_STEPS//2:].max() - sig[NN_STEPS//2:].min()) / 2.0
    return float(amp), "LC" if amp > AMP_THRESHOLD else "SS"

def main():
    print("=" * 70, flush=True)
    print("PHASE 3: BIFURCATION SURFACE RECONSTRUCTION", flush=True)
    print(f"RK4  grid: {len(KD2_VALS)} x {len(KD3_VALS)} x {len(KA_SLICES)} = {len(KD2_VALS)*len(KD3_VALS)*len(KA_SLICES)} pts", flush=True)
    print(f"NN   grid: {len(NN_KD2)} x {len(NN_KD3)} x {len(KA_SLICES)} = {len(NN_KD2)*len(NN_KD3)*len(KA_SLICES)} pts", flush=True)
    print("=" * 70, flush=True)

    print("\nLoading neural model...", flush=True)
    model = load_best_model()

    results = {
        "grid_kd2": KD2_VALS.tolist(), "grid_kd3": KD3_VALS.tolist(),
        "nn_kd2": NN_KD2.tolist(),     "nn_kd3": NN_KD3.tolist(),
        "ka_slices": KA_SLICES, "kp": KP, "amp_threshold_mM": AMP_THRESHOLD,
        "slices": {}
    }
    t_total = time.time()

    for ka in KA_SLICES:
        print(f"\n{'='*50}", flush=True)
        print(f"Slice: ka = {ka:.1f}", flush=True)

        # RK4 full grid
        rk4_amp = np.full((len(KD3_VALS), len(KD2_VALS)), np.nan)
        rk4_cls = np.full((len(KD3_VALS), len(KD2_VALS)), "", dtype=object)
        lc = ss = div = 0
        t0 = time.time()
        for j, kd3 in enumerate(KD3_VALS):
            for i, kd2 in enumerate(KD2_VALS):
                amp, cls = measure_rk4_amplitude([ka, kd2, kd3, KP])
                rk4_amp[j, i] = amp; rk4_cls[j, i] = cls
                if cls == "LC": lc += 1
                elif cls == "SS": ss += 1
                else: div += 1
            done = (j+1)*len(KD2_VALS); tot = len(KD3_VALS)*len(KD2_VALS)
            eta = (tot-done)/max(done/max(time.time()-t0,1e-3),1e-3)
            print(f"  RK4 row {j+1:2d}/{len(KD3_VALS)} kd3={kd3:.2f} | LC={lc} SS={ss} DIV={div} | ETA {eta:.0f}s", flush=True)
        print(f"  RK4 done: {time.time()-t0:.1f}s", flush=True)

        # Neural coarse grid
        print(f"  NN {len(NN_KD2)*len(NN_KD3)} pts...", flush=True)
        nn_amp = np.full((len(NN_KD3), len(NN_KD2)), np.nan)
        nn_cls = np.full((len(NN_KD3), len(NN_KD2)), "", dtype=object)
        nlc = nss = ndiv = 0
        t0 = time.time()
        for j, kd3 in enumerate(NN_KD3):
            for i, kd2 in enumerate(NN_KD2):
                amp, cls = measure_neural_fast(model, [ka, kd2, kd3, KP])
                nn_amp[j, i] = amp; nn_cls[j, i] = cls
                if cls == "LC": nlc += 1
                elif cls == "SS": nss += 1
                else: ndiv += 1
        print(f"  NN  done: {time.time()-t0:.1f}s  LC={nlc} SS={nss} DIV={ndiv}", flush=True)

        results["slices"][str(ka)] = {
            "rk4_amp": rk4_amp.tolist(), "rk4_cls": rk4_cls.tolist(),
            "nn_amp":  nn_amp.tolist(),  "nn_cls":  nn_cls.tolist(),
            "rk4_lc": lc, "rk4_ss": ss, "rk4_div": div,
            "nn_lc": nlc, "nn_ss": nss, "nn_div":  ndiv,
        }

    outpath = os.path.join(SCRATCH, "bifurcation_surface_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {outpath} ({os.path.getsize(outpath)//1024} KB)", flush=True)
    print(f"Total time: {(time.time()-t_total)/60:.1f} min", flush=True)
    print("\n=== HOPF BOUNDARY SUMMARY ===", flush=True)
    for ka in KA_SLICES:
        d = results["slices"][str(ka)]
        tot = d["rk4_lc"] + d["rk4_ss"]
        print(f"  ka={ka:.1f}: RK4 {d['rk4_lc']}/{tot} LC ({100*d['rk4_lc']/max(tot,1):.1f}%)  "
              f"NN {d['nn_lc']}/{len(NN_KD2)*len(NN_KD3)} LC ({100*d['nn_lc']/max(len(NN_KD2)*len(NN_KD3),1):.1f}%)", flush=True)

if __name__ == "__main__":
    main()
