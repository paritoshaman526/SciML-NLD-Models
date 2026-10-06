"""
Lorenz Attractor Dynamical System as a Time Series
==================================================
Simulates the canonical 3D chaotic Lorenz system:
    dx/dt = s * (y - x)
    dy/dt = r * x - y - x * z
    dz/dt = x * y - b * z

Features:
- Accurate 4th-Order Runge-Kutta (RK4) and Forward Euler integrators.
- Multivariable time series extraction for x(t), y(t), and z(t).
- Chaotic sensitivity analysis (Butterfly Effect / exponential divergence).
- Comprehensive multi-panel visualization:
    1. 3D phase space attractor trajectory (colored by time progression).
    2. Synchronized x(t), y(t), z(t) component time series.
    3. Sensitive dependence on initial conditions (trajectory divergence).
- CSV export capability for SciML / machine learning time series workflows.
"""

import os
import sys

# Prevent OpenBLAS thread allocation issues on Windows by ensuring env variables are active at process spawn
if sys.platform.startswith("win") and "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import subprocess
    result = subprocess.run([sys.executable] + sys.argv, env=os.environ)
    sys.exit(result.returncode)

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# =====================================================================
# 1. System Equations and Numerical Integrators
# =====================================================================
def lorenz_deriv(
    xyz: np.ndarray,
    s: float = 10.0,
    r: float = 28.0,
    b: float = 8.0 / 3.0
) -> np.ndarray:
    """
    Computes partial derivatives of the Lorenz system at point xyz.

    Parameters
    ----------
    xyz : array-like, shape (3,)
        State vector [x, y, z].
    s : float, default=10.0
        Prandtl number (sigma).
    r : float, default=28.0
        Rayleigh number (rho).
    b : float, default=8/3 (~2.6667)
        Geometric factor (beta).

    Returns
    -------
    xyz_dot : np.ndarray, shape (3,)
        Time derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)


def rk4_step(
    xyz: np.ndarray,
    dt: float,
    s: float = 10.0,
    r: float = 28.0,
    b: float = 8.0 / 3.0
) -> np.ndarray:
    """Performs a single 4th-Order Runge-Kutta integration step."""
    k1 = lorenz_deriv(xyz, s=s, r=r, b=b)
    k2 = lorenz_deriv(xyz + 0.5 * dt * k1, s=s, r=r, b=b)
    k3 = lorenz_deriv(xyz + 0.5 * dt * k2, s=s, r=r, b=b)
    k4 = lorenz_deriv(xyz + dt * k3, s=s, r=r, b=b)
    return xyz + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler_step(
    xyz: np.ndarray,
    dt: float,
    s: float = 10.0,
    r: float = 28.0,
    b: float = 8.0 / 3.0
) -> np.ndarray:
    """Performs a single Forward Euler integration step."""
    return xyz + dt * lorenz_deriv(xyz, s=s, r=r, b=b)


def simulate_lorenz(
    y0: Tuple[float, float, float] = (0.0, 1.0, 1.05),
    t_max: float = 50.0,
    dt: float = 0.01,
    s: float = 10.0,
    r: float = 28.0,
    b: float = 8.0 / 3.0,
    method: str = "rk4"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates the Lorenz system over time.

    Parameters
    ----------
    y0 : tuple of float, default=(0.0, 1.0, 1.05)
        Initial state [x0, y0, z0].
    t_max : float, default=50.0
        Total simulation time duration.
    dt : float, default=0.01
        Time step size.
    s, r, b : float
        Lorenz parameters.
    method : str, default='rk4'
        Integration method: 'rk4' (recommended) or 'euler'.

    Returns
    -------
    t : np.ndarray, shape (num_steps + 1,)
        Time coordinates.
    states : np.ndarray, shape (num_steps + 1, 3)
        Array of state trajectories where columns correspond to x, y, and z.
    """
    num_steps = int(round(t_max / dt))
    t = np.linspace(0.0, t_max, num_steps + 1)
    states = np.empty((num_steps + 1, 3), dtype=np.float64)
    states[0] = y0

    step_func = rk4_step if method.lower() == "rk4" else euler_step

    for i in range(num_steps):
        states[i + 1] = step_func(states[i], dt=dt, s=s, r=r, b=b)

    return t, states


# =====================================================================
# 2. Data Export Utility
# =====================================================================
def save_time_series_to_csv(
    t: np.ndarray,
    states: np.ndarray,
    filename: str = "lorenz_time_series.csv"
) -> None:
    """Saves generated time series (t, x, y, z) into a CSV file."""
    data = np.column_stack((t, states))
    header = "time,x,y,z"
    np.savetxt(filename, data, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"Time series saved to {filename} ({len(t):,d} records)")


# =====================================================================
# 3. Comprehensive Time Series & Attractor Visualization
# =====================================================================
def plot_lorenz_time_series(
    t: np.ndarray,
    states: np.ndarray,
    perturbed_states: Optional[np.ndarray] = None,
    s: float = 10.0,
    r: float = 28.0,
    b: float = 8.0 / 3.0,
    save_fig: Optional[str] = "lorenz_time_series.png",
    show_plot: bool = True
) -> plt.Figure:
    """
    Generates a publication-grade multi-panel figure displaying:
      - 3D Lorenz Attractor Phase Portrait
      - Synchronized x(t), y(t), z(t) component time series
      - Butterfly effect / Chaotic divergence over time (if perturbed_states provided)
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 1.4], hspace=0.3, wspace=0.22)

    # -------------------------------------------------------------
    # 1. 3D Phase Space Attractor (Left Column: Span 2 Rows)
    # -------------------------------------------------------------
    ax_3d = fig.add_subplot(gs[0:2, 0], projection="3d")
    
    # Plot trajectory with gradient coloring by time
    points = states.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(t.min(), t.max())
    lc = Line3DCollection(segments, cmap="plasma", norm=norm, linewidth=0.75, alpha=0.85)
    lc.set_array(t[:-1])
    ax_3d.add_collection3d(lc)

    # Auto-scale 3D limits
    ax_3d.set_xlim(states[:, 0].min() - 1, states[:, 0].max() + 1)
    ax_3d.set_ylim(states[:, 1].min() - 1, states[:, 1].max() + 1)
    ax_3d.set_zlim(states[:, 2].min() - 1, states[:, 2].max() + 1)

    # Initial condition marker
    ax_3d.scatter(
        [states[0, 0]], [states[0, 1]], [states[0, 2]],
        color="crimson", s=40, zorder=10, label=f"IC ({states[0,0]:.1f}, {states[0,1]:.1f}, {states[0,2]:.2f})"
    )

    ax_3d.set_xlabel("X (Convective Intensity)", fontsize=10, labelpad=8)
    ax_3d.set_ylabel("Y (Temp Difference)", fontsize=10, labelpad=8)
    ax_3d.set_zlabel("Z (Heat Conduction)", fontsize=10, labelpad=8)
    ax_3d.set_title(
        rf"Lorenz Attractor 3D Phase Portrait ($\sigma={s},\ \rho={r},\ \beta={b:.2f}$)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax_3d.view_init(elev=24, azim=42)
    cbar = fig.colorbar(lc, ax=ax_3d, shrink=0.55, aspect=15, pad=0.1)
    cbar.set_label("Time ($t$)", fontsize=9)
    ax_3d.legend(loc="upper left", fontsize=9)

    # -------------------------------------------------------------
    # 2. Synchronized Component Time Series (Right Column: 3 Subplots)
    # -------------------------------------------------------------
    labels = ["x(t)  [Convective Flow]", "y(t)  [Horizontal Temp Gradient]", "z(t)  [Vertical Temp Gradient]"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    for i in range(3):
        ax_ts = fig.add_subplot(gs[i, 1])
        ax_ts.plot(t, states[:, i], color=colors[i], lw=1.2, label=f"Primary Trajectory {labels[i].split()[0]}")
        
        # If perturbed trajectory exists, show divergence
        if perturbed_states is not None:
            ax_ts.plot(
                t, perturbed_states[:, i],
                color="red", ls="--", lw=1.0, alpha=0.75,
                label=r"Perturbed Trajectory ($\Delta y_0 = 10^{-5}$)"
            )

        ax_ts.set_ylabel(labels[i], fontsize=10, fontweight="bold")
        ax_ts.grid(True, linestyle=":", alpha=0.6)
        ax_ts.set_xlim(t[0], t[-1])

        if i == 0:
            ax_ts.set_title("Synchronized State Variable Time Series", fontsize=12, fontweight="bold")
            ax_ts.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
        if i == 2:
            ax_ts.set_xlabel("Time ($t$)", fontsize=11, fontweight="bold")
        else:
            ax_ts.set_xticklabels([])

    # -------------------------------------------------------------
    # 3. Butterfly Effect / Error Divergence (Bottom-Left Subplot)
    # -------------------------------------------------------------
    ax_err = fig.add_subplot(gs[2, 0])
    if perturbed_states is not None:
        separation = np.linalg.norm(states - perturbed_states, axis=1)
        ax_err.semilogy(t, separation, color="#9467bd", lw=1.5, label=r"Separation $||\mathbf{x}_1(t) - \mathbf{x}_2(t)||$")
        ax_err.set_title("Chaotic Divergence: The Butterfly Effect", fontsize=11, fontweight="bold")
        ax_err.set_xlabel("Time ($t$)", fontsize=10)
        ax_err.set_ylabel(r"Log Distance $||\Delta \mathbf{x}(t)||$", fontsize=10)
        ax_err.grid(True, which="both", linestyle=":", alpha=0.6)
        ax_err.legend(loc="lower right", fontsize=8.5)
        ax_err.set_xlim(t[0], t[-1])
    else:
        # If no perturbation, display 2D projection (X vs Z - classic butterfly lobes)
        ax_err.plot(states[:, 0], states[:, 2], color="#333333", lw=0.6, alpha=0.8)
        ax_err.set_title("2D Projection: X vs Z (Butterfly Profile)", fontsize=11, fontweight="bold")
        ax_err.set_xlabel("X", fontsize=10)
        ax_err.set_ylabel("Z", fontsize=10)
        ax_err.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Lorenz Dynamical System: Time Series and Phase Dynamics", fontsize=14, fontweight="bold", y=0.98)

    if save_fig:
        plt.savefig(save_fig, dpi=300, bbox_inches="tight")
        print(f"Figure saved successfully to: {save_fig}")

    if show_plot:
        plt.show()

    return fig


# =====================================================================
# 4. Main Execution Pipeline
# =====================================================================
def main():
    # Simulation Parameters
    dt = 0.01
    t_max = 40.0
    s = 10.0
    r = 28.0
    b = 8.0 / 3.0  # ~2.6667

    # Initial Condition 1 (Canonical baseline)
    y0_base = (0.0, 1.0, 1.05)

    # Initial Condition 2 (Tiny perturbation to demonstrate butterfly effect)
    delta = 1e-5
    y0_perturbed = (0.0, 1.0 + delta, 1.05)

    print("=" * 70)
    print("Lorenz Model Time Series Simulation")
    print(f"Parameters : sigma (s)={s}, rho (r)={r}, beta (b)={b:.4f}")
    print(f"Time Grid  : dt={dt}, t_max={t_max} ({int(t_max/dt):,d} steps)")
    print(f"Base IC    : {y0_base}")
    print(f"Perturb IC : {y0_perturbed} (difference = {delta})")
    print("=" * 70)

    # Simulate base trajectory with high-accuracy RK4
    t, states = simulate_lorenz(
        y0=y0_base, t_max=t_max, dt=dt, s=s, r=r, b=b, method="rk4"
    )

    # Simulate perturbed trajectory to demonstrate chaotic sensitivity
    _, perturbed_states = simulate_lorenz(
        y0=y0_perturbed, t_max=t_max, dt=dt, s=s, r=r, b=b, method="rk4"
    )

    # Summary Statistics
    print("\nState Variables Summary:")
    names = ["x(t)", "y(t)", "z(t)"]
    for i, name in enumerate(names):
        print(
            f"  {name:5s} -> Min: {states[:, i].min():8.3f}, "
            f"Max: {states[:, i].max():8.3f}, "
            f"Mean: {states[:, i].mean():8.3f}, "
            f"Std: {states[:, i].std():8.3f}"
        )

    # Save time series to CSV
    save_time_series_to_csv(t, states, filename="lorenz_time_series.csv")

    # Plot time series and chaotic divergence
    plot_lorenz_time_series(
        t=t,
        states=states,
        perturbed_states=perturbed_states,
        s=s,
        r=r,
        b=b,
        save_fig="lorenz_time_series.png",
        show_plot=False  # Headless-safe, can be set to True for interactive windows
    )


if __name__ == "__main__":
    main()
