import os, sys, time
import numpy as np

PROJ = r"c:\Users\satya\Downloads\Van Der Pol"
sys.path.insert(0, PROJ)
from biochemical_global_sciml import system, rk4_step, DEFAULT_GUESS

print("=" * 80)
print("GROUND TRUTH RK4 LIMIT CYCLE EXTINCTION ANALYSIS")
print("=" * 80)

# Full grid definition as used in benchmark:
KD2_VALS = np.concatenate([np.linspace(100, 800, 12), np.linspace(800, 6000, 28)[1:]])  # 39 pts
KD3_VALS = np.linspace(0.1, 2.2, 35)
dt = 0.01
WARMUP_STEPS = 1200   # 12.0s warmup
MEASURE_STEPS = 500   # 5.0s measurement
AMP_THRESH = 8.0      # mM

# Let's test a fine array of ka values around the transition
ka_values = [8.5, 9.0, 9.2, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.95, 9.975, 10.0, 10.2]

print(f"Testing {len(ka_values)} ka slices across {len(KD2_VALS)} kd2 x {len(KD3_VALS)} kd3...")
print(f"Timing: {WARMUP_STEPS*dt:.1f}s warmup + {MEASURE_STEPS*dt:.1f}s measurement, threshold = {AMP_THRESH} mM\n")

results = {}

for ka in ka_values:
    t0 = time.time()
    # To be fast and accurate, for each kd3 (from highest 2.2 down to 0.1), check if ANY kd2 sustains LC
    # Since kd3_max is monotonic or bounded, scanning from top kd3 down finds kd3_max quickly!
    found_kd3_max = 0.0
    lc_count = 0
    max_amp = 0.0
    best_pt = None
    
    # Check all (kd3, kd2)
    for kd3 in reversed(KD3_VALS):
        has_lc_at_this_kd3 = False
        for kd2 in KD2_VALS:
            p = np.array([ka, kd2, kd3, 1.0], dtype=np.float64)
            y = DEFAULT_GUESS.copy()
            for _ in range(WARMUP_STEPS):
                y = rk4_step(y, dt, p)
            sig_min = y[2]
            sig_max = y[2]
            for _ in range(MEASURE_STEPS):
                y = rk4_step(y, dt, p)
                if y[2] < sig_min: sig_min = y[2]
                if y[2] > sig_max: sig_max = y[2]
            amp = (sig_max - sig_min) / 2.0
            if amp > AMP_THRESH:
                lc_count += 1
                has_lc_at_this_kd3 = True
                if amp > max_amp:
                    max_amp = amp
                    best_pt = (kd2, kd3)
        if has_lc_at_this_kd3 and found_kd3_max == 0.0:
            found_kd3_max = kd3
            
    results[ka] = {
        "kd3_max": found_kd3_max,
        "lc_count": lc_count,
        "max_amp": max_amp,
        "best_pt": best_pt,
        "elapsed": time.time() - t0
    }
    print(f"ka = {ka:6.3f} | kd3_max = {found_kd3_max:6.3f} | LC count = {lc_count:4d} | Max Amp = {max_amp:7.2f} mM | Time = {results[ka]['elapsed']:.1f}s", flush=True)

print("\n" + "=" * 80)
print("FINAL SUMMARY OF RK4 GROUND TRUTH EXTINCTION:")
for ka, r in results.items():
    print(f"ka = {ka:6.3f} : kd3_max = {r['kd3_max']:.3f} s^-1, total LC points = {r['lc_count']}")
