import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
delta = data["kd3_step"]

seeds = [42, 101, 777, 999, 1234, 2024, 5555]
seed_curves = {s: np.array([r["per_seed_kd3_max"][str(s)] for r in data["rows"]]) for s in seeds}

plt.style.use("default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor="white")

# Panel 1: Dense Boundary Comparison
# True RK4
ax1.plot(ka_arr, rk4_arr, "r-o", linewidth=2.8, markersize=6, label=r"True Ground-Truth RK4 ($k_{d3,\max}$)", zorder=4)

# Individual seeds
seed_colors = ["#2b6cb0", "#319795", "#805ad5", "#d69e2e", "#dd6b20", "#e53e3e", "#3182ce"]
for i, s in enumerate(seeds):
    ax1.plot(ka_arr, seed_curves[s], "--", color=seed_colors[i], alpha=0.35, linewidth=1.2, label=f"Seed {s}" if i < 2 else None)

# Ensemble Mean
ax1.plot(ka_arr, ens_mean, "-", color="#1a365d", linewidth=2.6, label=f"7-Seed Ensemble Mean (RMSE={data['rmse']:.3f} " + r"$s^{-1}$)", zorder=5)
ax1.fill_between(ka_arr, ens_mean - ens_std, ens_mean + ens_std, color="#2b6cb0", alpha=0.18, label="Inter-Seed Std Dev")

# Highlight 9.975
idx_9975 = np.argmin(np.abs(ka_arr - 9.975))
ax1.axvline(9.975, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
ax1.scatter([9.975], [rk4_arr[idx_9975]], color="red", s=80, zorder=6)
ax1.scatter([9.975], [ens_mean[idx_9975]], color="#1a365d", s=80, zorder=6)
ax1.annotate(r"$k_a = 9.975$" + f"\nRK4: {rk4_arr[idx_9975]:.3f}\nSciML: {ens_mean[idx_9975]:.3f}",
             xy=(9.975, rk4_arr[idx_9975]), xytext=(7.8, 0.25),
             arrowprops=dict(arrowstyle="->", color="#2d3748", lw=1.2),
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#edf2f7", ec="#cbd5e0"))

ax1.set_xlabel(r"Bifurcation Parameter $k_a$", fontsize=12)
ax1.set_ylabel(r"Maximum Limit-Cycle Boundary $k_{d3,\max}$ ($\mathrm{s}^{-1}$)", fontsize=12)
ax1.set_title("Full-Resolution Bifurcation Boundary: True RK4 vs. 7-Seed SciML Ensemble\n(18 Dense Slices across Active and Extinction Regimes)", fontsize=12, fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.5)
ax1.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
ax1.set_ylim(-0.05, 2.35)
ax1.set_xlim(0.8, 12.2)

# Panel 2: Pointwise Deviation Analysis
ax2.plot(ka_arr, dev_arr, "s-", color="#2c5282", linewidth=2.4, markersize=6, label=r"Ensemble Deviation $\Delta k_{d3,\max}$")
ax2.axhline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.7)
ax2.axhline(delta, color="#48bb78", linestyle="--", linewidth=1.2, alpha=0.8, label=r"$\pm 1$ Grid Bin ($\Delta = 0.062\ \mathrm{s}^{-1}$)")
ax2.axhline(-delta, color="#48bb78", linestyle="--", linewidth=1.2, alpha=0.8)
ax2.axhline(2*delta, color="#ecc94b", linestyle=":", linewidth=1.2, alpha=0.8, label=r"$\pm 2$ Grid Bins ($\Delta = 0.124\ \mathrm{s}^{-1}$)")
ax2.axhline(-2*delta, color="#ecc94b", linestyle=":", linewidth=1.2, alpha=0.8)
ax2.fill_between(ka_arr, -delta, delta, color="#48bb78", alpha=0.10)
ax2.fill_between(ka_arr, -2*delta, 2*delta, color="#ecc94b", alpha=0.08)

ax2.axvline(9.975, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
ax2.annotate(f"Dev at 9.975: {dev_arr[idx_9975]:+.3f}" + r" $\mathrm{s}^{-1}$",
             xy=(9.975, dev_arr[idx_9975]), xytext=(9.2, 0.02),
             arrowprops=dict(arrowstyle="->", color="#2d3748", lw=1.2),
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#edf2f7", ec="#cbd5e0"))

ax2.set_xlabel(r"Bifurcation Parameter $k_a$", fontsize=12)
ax2.set_ylabel(r"Deviation $\Delta k_{d3,\max} = \mathrm{SciML} - \mathrm{RK4}$ ($\mathrm{s}^{-1}$)", fontsize=12)
ax2.set_title("Pointwise Deviation along Bifurcation Horizon\n(Calibrated against Discrete Grid Quantization)", fontsize=12, fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.5)
ax2.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=10)
ax2.set_ylim(-0.15, 0.35)
ax2.set_xlim(0.8, 12.2)

plt.tight_layout()
out_png = os.path.join(SCRATCH, "dense_grid_deviation_analysis.png")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Saved publication figure to: {out_png}")
