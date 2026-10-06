import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

PROJ = r"c:\Users\satya\Downloads\Van Der Pol"
SCRATCH = os.path.join(PROJ, "scratch")

json_path = os.path.join(SCRATCH, "dense_grid_bifurcation_deviation.json")
with open(json_path) as f:
    data = json.load(f)

ka_arr = np.array(data["ka_grid"])
rk4_arr = np.array([r["rk4_kd3_max"] for r in data["rows"]])
ens_mean = np.array([r["ensemble_mean_kd3_max"] for r in data["rows"]])
ens_std = np.array([r["ensemble_std_kd3_max"] for r in data["rows"]])
dev_arr = np.array([r["deviation"] for r in data["rows"]])
delta = data["kd3_step"] # 0.06176 s^-1

seeds = [42, 101, 777, 999, 1234, 2024, 5555]
seed_curves = {s: np.array([r["per_seed_kd3_max"][str(s)] for r in data["rows"]]) for s in seeds}

# Smooth spline derivatives of the true RK4 boundary
spl = UnivariateSpline(ka_arr, rk4_arr, s=0.005)
ka_dense = np.linspace(1.0, 12.0, 500)
rk4_dense = spl(ka_dense)
d1_dense = spl.derivative(1)(ka_dense)      # y' = d(kd3_max)/d(ka)
d2_dense = spl.derivative(2)(ka_dense)      # y'' = d^2(kd3_max)/d(ka)^2
# Curvature definition: kappa(x) = |y''| / (1 + y'^2)^(3/2)
curv_dense = np.abs(d2_dense) / (1.0 + d1_dense**2)**1.5

plt.style.use("default")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6.4), facecolor="white")

# ==============================================================================
# PANEL 1: High-Resolution Bifurcation Boundary Comparison
# ==============================================================================
ax1.plot(ka_arr, rk4_arr, "r-o", linewidth=2.6, markersize=5.5, label=r"True Ground-Truth RK4 ($k_{d3,\max}$)", zorder=4)

seed_colors = ["#2b6cb0", "#319795", "#805ad5", "#d69e2e", "#dd6b20", "#e53e3e", "#3182ce"]
for i, s in enumerate(seeds):
    ax1.plot(ka_arr, seed_curves[s], "--", color=seed_colors[i], alpha=0.35, linewidth=1.1, label=f"Seed {s}" if i < 2 else None)

ax1.plot(ka_arr, ens_mean, "-", color="#1a365d", linewidth=2.6, label=f"7-Seed Ensemble Mean (RMSE={data['rmse']:.3f} " + r"$\mathrm{s}^{-1}$)", zorder=5)
ax1.fill_between(ka_arr, ens_mean - ens_std, ens_mean + ens_std, color="#2b6cb0", alpha=0.20, label="Inter-Seed Std Dev")

# Highlight domain boundary at 9.975 - 10.0
idx_9975 = np.argmin(np.abs(ka_arr - 9.975))
ax1.axvline(9.975, color="#718096", linestyle=":", linewidth=1.4, alpha=0.8)
ax1.scatter([9.975], [rk4_arr[idx_9975]], color="red", s=70, zorder=6)
ax1.scatter([9.975], [ens_mean[idx_9975]], color="#1a365d", s=70, zorder=6)
ax1.annotate(r"Benchmark Edge $k_a = 9.975$" + f"\nRK4: {rk4_arr[idx_9975]:.3f}\nSciML: {ens_mean[idx_9975]:.3f}",
             xy=(9.975, rk4_arr[idx_9975]), xytext=(7.4, 0.25),
             arrowprops=dict(arrowstyle="->", color="#2d3748", lw=1.2),
             fontsize=9.5, bbox=dict(boxstyle="round,pad=0.3", fc="#edf2f7", ec="#cbd5e0"))

ax1.set_xlabel(r"Bifurcation Parameter $k_a$", fontsize=11.5)
ax1.set_ylabel(r"Limit-Cycle Boundary $k_{d3,\max}$ ($\mathrm{s}^{-1}$)", fontsize=11.5)
ax1.set_title("(a) Full-Resolution Bifurcation Boundary\n(18 Dense Slices, 35 Vertical $k_{d3}$ Bins)", fontsize=11.5, fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.5)
ax1.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5)
ax1.set_ylim(-0.05, 2.35)
ax1.set_xlim(0.8, 12.2)

# ==============================================================================
# PANEL 2: Pointwise Deviation vs. Underlying Sweep Grid Resolution
# ==============================================================================
ax2.plot(ka_arr, dev_arr, "s-", color="#2c5282", linewidth=2.4, markersize=5.5, label=r"Ensemble Deviation $\Delta k_{d3,\max}$", zorder=4)
ax2.axhline(0, color="black", linestyle="-", linewidth=1.1, alpha=0.7)

# Explicitly labeled grid resolution reference lines
ax2.axhline(delta, color="#38a169", linestyle="--", linewidth=1.3, alpha=0.85, label=f"±1 Grid Cell Spacing (Δ = {delta:.3f} " + r"$\mathrm{s}^{-1}$)")
ax2.axhline(-delta, color="#38a169", linestyle="--", linewidth=1.3, alpha=0.85)
ax2.axhline(2*delta, color="#d69e2e", linestyle=":", linewidth=1.4, alpha=0.9, label=f"±2 Grid Cell Spacing (2Δ = {2*delta:.3f} " + r"$\mathrm{s}^{-1}$)")
ax2.axhline(-2*delta, color="#d69e2e", linestyle=":", linewidth=1.4, alpha=0.9)
ax2.fill_between(ka_arr, -delta, delta, color="#38a169", alpha=0.10)
ax2.fill_between(ka_arr, -2*delta, 2*delta, color="#d69e2e", alpha=0.08)

# Shaded region highlighting that deviation exceeds grid resolution
ax2.axvspan(2.5, 8.5, color="#fed7d7", alpha=0.25, zorder=1)
ax2.text(5.5, 0.285, "Deviation Exceeds Grid Resolution\n(Genuine Tracking Lag, Not Discretization)", 
         ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#9b2c2c",
         bbox=dict(boxstyle="round,pad=0.25", fc="#fff5f5", ec="#feb2b2", alpha=0.9))

ax2.set_xlabel(r"Bifurcation Parameter $k_a$", fontsize=11.5)
ax2.set_ylabel(r"Deviation $\Delta k_{d3,\max}$ ($\mathrm{s}^{-1}$)", fontsize=11.5)
ax2.set_title("(b) Pointwise Ensemble Deviation\n(Deviation Exceeds Grid Resolution Across Mid-Range)", fontsize=11.5, fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.5)
ax2.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9.5)
ax2.set_ylim(-0.10, 0.36)
ax2.set_xlim(0.8, 12.2)

# ==============================================================================
# PANEL 3: Disentangling Curvature vs. Descent Rate vs. Persistent Lag
# ==============================================================================
ax3_twin = ax3.twinx()

# Plot deviation profile on left axis
l1 = ax3.plot(ka_arr, dev_arr, "s-", color="#2c5282", linewidth=2.4, markersize=5.5, label=r"Ensemble Deviation $\Delta k_{d3,\max}$ (Left Axis)")
ax3.set_ylabel(r"Deviation $\Delta k_{d3,\max}$ ($\mathrm{s}^{-1}$)", color="#2c5282", fontsize=11.5)
ax3.tick_params(axis='y', labelcolor="#2c5282")
ax3.set_ylim(-0.10, 0.38)

# Plot True RK4 absolute slope |y'| and Curvature kappa on twin axis
l2 = ax3_twin.plot(ka_dense, np.abs(d1_dense), "-", color="#c53030", linewidth=2.2, label=r"Descent Rate $|y'| = |dk_{d3}/dk_a|$ (Right Axis)")
l3 = ax3_twin.plot(ka_dense, curv_dense, "--", color="#dd6b20", linewidth=2.0, label=r"Boundary Curvature $\kappa(k_a) = |y''|/(1+y'^2)^{3/2}$ (Right Axis)")
ax3_twin.set_ylabel(r"Descent Rate $|y'|$ & Curvature $\kappa(k_a)$", color="#c53030", fontsize=11.5)
ax3_twin.tick_params(axis='y', labelcolor="#c53030")
ax3_twin.set_ylim(0.0, 0.42)

# Highlight inflection onset (ka=1.5-3.0) and persistent plateau (ka=3.0-6.5)
ax3.axvspan(1.5, 3.0, color="#feebc8", alpha=0.35, zorder=1)
ax3.text(2.25, 0.32, "Curvature Peak\n& Onset of Drop", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#7b341e")

ax3.axvspan(3.0, 6.5, color="#fed7d7", alpha=0.30, zorder=1)
ax3.text(4.75, 0.02, "Persistent Plateau:\nCorrection Stays Saturated Past\nWhere Curvature Eases", 
         ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#9b2c2c",
         bbox=dict(boxstyle="round,pad=0.25", fc="#fff5f5", ec="#feb2b2", alpha=0.9))

ax3.set_xlabel(r"Bifurcation Parameter $k_a$", fontsize=11.5)
ax3.set_title("(c) Mechanism: Curvature vs. Descent Rate vs. Error\n(Onset Saturation & Persistent Plateau-Lag)", fontsize=11.5, fontweight="bold")
ax3.grid(True, linestyle=":", alpha=0.5)

# Combined legend
lines = l1 + l2 + l3
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc="upper right", frameon=True, framealpha=0.9, fontsize=8.5)
ax3.set_xlim(0.8, 12.2)

plt.tight_layout()
out_png = os.path.join(SCRATCH, "dense_grid_deviation_mechanism.png")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Saved refined 3-panel publication figure to: {out_png}")
