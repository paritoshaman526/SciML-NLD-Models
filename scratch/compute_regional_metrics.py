import json, numpy as np
from scipy.interpolate import UnivariateSpline

with open('scratch/dense_grid_bifurcation_deviation.json') as f:
    d = json.load(f)

ka = np.array(d['ka_grid'])
rk4 = np.array([r['rk4_kd3_max'] for r in d['rows']])
dev = np.array([r['deviation'] for r in d['rows']])

# Fit spline
spl = UnivariateSpline(ka, rk4, s=0.005)
dense_k = np.linspace(1.0, 12.0, 1000)
y = spl(dense_k)
y_prime = spl.derivative(1)(dense_k)
y_double_prime = spl.derivative(2)(dense_k)
signed_curv = y_double_prime / (1.0 + y_prime**2)**1.5

intervals = [
    ("Regime 1: Ceiling Plateau (1.0 <= ka <= 2.0)",  1.0, 2.0),
    ("Regime 2: Transition Drop (2.0 < ka <= 3.5)",   2.0001, 3.5),
    ("Regime 3: Persistent Plateau (3.5 < ka <= 6.5)", 3.5001, 6.5),
    ("Regime 4: Manifold Deceleration (6.5 < ka <= 9.0)", 6.5001, 9.0),
    ("Regime 5: Benchmark Spine (9.0 < ka <= 10.0)",  9.0001, 10.0),
    ("Regime 6: Extinction Tail (10.0 < ka <= 12.0)", 10.0001, 12.0),
]

print("=== REGIONAL METRICS (DISJOINT INTERVALS) ===")
for name, k_min, k_max in intervals:
    mask = (dense_k >= k_min) & (dense_k <= k_max)
    yp = y_prime[mask]
    ypp = y_double_prime[mask]
    k_curv = signed_curv[mask]
    
    disc_mask = (ka >= k_min - 0.01) & (ka <= k_max + 0.01)
    dev_sub = dev[disc_mask]
    
    print(f"\n{name}:")
    print(f"   Descent Rate |y'| : {np.min(np.abs(yp)):.3f} to {np.max(np.abs(yp)):.3f} (mean {np.mean(np.abs(yp)):.3f})")
    print(f"   y'' (2nd deriv)   : {np.min(ypp):+.3f} to {np.max(ypp):+.3f}")
    print(f"   Signed Curv k     : {np.min(k_curv):+.3f} to {np.max(k_curv):+.3f}")
    print(f"   Observed Dev      : {np.min(dev_sub):+.3f} to {np.max(dev_sub):+.3f} s^-1 (mean {np.mean(dev_sub):+.3f})")
