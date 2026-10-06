import os, sys, time
import numpy as np
from numba import njit, prange

PROJ = r"c:\Users\satya\Downloads\Van Der Pol"
sys.path.insert(0, PROJ)
from biochemical_global_sciml import system, rk4_step, DEFAULT_GUESS

KD2_VALS = np.concatenate([np.linspace(100, 800, 12), np.linspace(800, 6000, 28)[1:]])  # 39 pts
KD3_VALS = np.linspace(0.1, 2.2, 35)

@njit(parallel=True, fastmath=True)
def scan_slice_rk4(ka, kd2_arr, kd3_arr, warmup_steps, measure_steps, dt, amp_thresh):
    n3 = len(kd3_arr)
    n2 = len(kd2_arr)
    amps = np.zeros((n3, n2))
    
    for i in prange(n3):
        kd3 = kd3_arr[i]
        for j in range(n2):
            kd2 = kd2_arr[j]
            p = np.array([ka, kd2, kd3, 1.0])
            y = np.array([6657.918, 3267.0, 77.082, 2.038])
            for _ in range(warmup_steps):
                y = rk4_step(y, dt, p)
            s_min = y[2]
            s_max = y[2]
            for _ in range(measure_steps):
                y = rk4_step(y, dt, p)
                if y[2] < s_min:
                    s_min = y[2]
                if y[2] > s_max:
                    s_max = y[2]
            amps[i, j] = (s_max - s_min) / 2.0
    return amps

if __name__ == "__main__":
    dt = 0.01
    warmup = 1200
    measure = 500
    thresh = 8.0
    
    # Warmup jit compilation
    print("Compiling Numba JIT parallel scanner...")
    _ = scan_slice_rk4(8.5, KD2_VALS[:2], KD3_VALS[:2], 10, 10, dt, thresh)
    print("Compilation done!\n")
    
    ka_test_list = [1.0, 2.5, 5.0, 7.0, 8.0, 8.5, 9.0, 9.2, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.95, 9.975, 10.0, 10.2, 10.5, 11.0, 12.0]
    
    print(f"{'ka':6s} | {'kd3_max (s^-1)':14s} | {'LC Count':9s} | {'Max Amp (mM)':12s} | {'Elapsed'}")
    print("-" * 65)
    
    for ka in ka_test_list:
        t0 = time.time()
        amps = scan_slice_rk4(ka, KD2_VALS, KD3_VALS, warmup, measure, dt, thresh)
        is_lc = amps > thresh
        lc_count = np.sum(is_lc)
        max_amp = np.max(amps)
        
        kd3_max = 0.0
        for i in range(len(KD3_VALS) - 1, -1, -1):
            if np.any(is_lc[i, :]):
                kd3_max = KD3_VALS[i]
                break
        t_el = time.time() - t0
        print(f"{ka:6.3f} | {kd3_max:14.3f} | {lc_count:9d} | {max_amp:12.2f} | {t_el:6.2f}s", flush=True)
