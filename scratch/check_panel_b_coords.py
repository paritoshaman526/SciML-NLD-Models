import numpy as np

data = np.load(r"c:\Users\satya\Downloads\Van Der Pol\vanderpol\scratch\vdp_settled_bifurcation_data.npz")
mu = data["mu_sweep"]
gt_p = data["gt_period"]
a_p = data["a_p_mean"]
mask = (mu > 0.05) & (gt_p > 0)

mu_p = mu[mask]
gt_p_vals = gt_p[mask]
a_p_vals = a_p[mask]
delta_T = a_p_vals - gt_p_vals

for m, g, a, d in zip(mu_p, gt_p_vals, a_p_vals, delta_T):
    print(f"mu={m:+.2f} | T_GT={g:.3f} s | T_ModelA={a:.3f} s | Delta_T={d:+.4f} s")
