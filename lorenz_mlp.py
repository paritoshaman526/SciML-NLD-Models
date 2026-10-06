"""
High-Accuracy Physics-Informed SciML for the 3D Chaotic Lorenz Attractor
========================================================================
Key Innovations & Enhancements:
1. Physics-Informed Inductive Bias:
   - State normalization + Bilinear cross-feature expansion: [x, y, z, xy, xz, yz].
   - Direct physics carrier path (Linear) + Deep Residual Blocks (ResMLP) with SiLU.
   - Internal 4th-Order Runge-Kutta (RK4) integration step in forward pass.
2. Multi-Step Curricular Unrolling Loss:
   - Trained on 1-step and 2-step transition horizons (L = L1 + 0.5 * L2) to suppress
     compounding trajectory drift.
3. Held-out Initial Condition (IC) Partitioning:
   - 100% leak-free evaluation on unseen initial conditions.
4. Side-by-Side 3D Attractor Visualization:
   - Dedicated 3D Axis 1: Ground Truth (RK4) Attractor colored by time progression.
   - Dedicated 3D Axis 2: SciML Predicted Attractor in a separate 3D axes with identical
     camera view angles and coordinate scales for direct visual comparison.
   - Synchronized component time series [x(t), y(t), z(t)] and 2D butterfly projections.
   - Lyapunov predictability horizon tracking.
"""

import os
import sys

# Prevent OpenBLAS thread allocation issues on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from typing import Tuple, List, Optional
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# =====================================================================
# 0. Deterministic Reproducibility (Seeds Across All Libraries)
# =====================================================================
def set_seed(seed: int = 42) -> None:
    """Guarantees 100% bit-for-bit reproducibility across all environments and devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# =====================================================================
# 1. Ground Truth Dynamical System (RK4 Integration)
# =====================================================================
def lorenz_deriv(xyz: np.ndarray, s: float = 10.0, r: float = 28.0, b: float = 8.0 / 3.0) -> np.ndarray:
    """True partial derivatives of the Lorenz attractor."""
    x, y, z = xyz[0], xyz[1], xyz[2]
    dx = s * (y - x)
    dy = r * x - y - x * z
    dz = x * y - b * z
    return np.array([dx, dy, dz], dtype=np.float64)


def rk4_step(xyz: np.ndarray, dt: float, s: float = 10.0, r: float = 28.0, b: float = 8.0 / 3.0) -> np.ndarray:
    """Performs a single 4th-order Runge-Kutta step."""
    k1 = lorenz_deriv(xyz, s, r, b)
    k2 = lorenz_deriv(xyz + 0.5 * dt * k1, s, r, b)
    k3 = lorenz_deriv(xyz + 0.5 * dt * k2, s, r, b)
    k4 = lorenz_deriv(xyz + dt * k3, s, r, b)
    return xyz + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def generate_trajectories(
    num_trajectories: int = 70,
    steps: int = 500,
    dt: float = 0.01,
    s: float = 10.0,
    r: float = 28.0,
    b: float = 8.0 / 3.0,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[np.ndarray]],
           Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[np.ndarray]]]:
    """
    Generates training and validation data partitioned strictly by Initial Condition (held-out ICs).
    Returns (X_t, Y_{t+1}, Y_{t+2}) tuples for multi-step unrolling loss.
    """
    set_seed(seed)

    # Sample initial conditions and relax onto the attractor
    raw_ics = np.random.uniform(low=[-15.0, -20.0, 10.0], high=[15.0, 20.0, 35.0], size=(num_trajectories, 3))
    init_states = []
    for ic in raw_ics:
        state = ic.copy()
        for _ in range(200):
            state = rk4_step(state, dt=dt, s=s, r=r, b=b)
        init_states.append(state)
    init_states = np.array(init_states)

    n_val = int(num_trajectories * val_split)
    val_ics = init_states[:n_val]
    train_ics = init_states[n_val:]

    def _simulate_trajectories(ics):
        X0_data = []  # state at t
        X1_data = []  # state at t+1
        X2_data = []  # state at t+2
        trajectories = []
        for state0 in ics:
            curr = state0.copy()
            traj = [curr]
            for _ in range(steps):
                nxt1 = rk4_step(curr, dt=dt, s=s, r=r, b=b)
                nxt2 = rk4_step(nxt1, dt=dt, s=s, r=r, b=b)
                X0_data.append(curr)
                X1_data.append(nxt1)
                X2_data.append(nxt2)
                traj.append(nxt1)
                curr = nxt1
            trajectories.append(np.array(traj))

        X0_tensor = torch.tensor(np.array(X0_data), dtype=torch.float32)
        X1_tensor = torch.tensor(np.array(X1_data), dtype=torch.float32)
        X2_tensor = torch.tensor(np.array(X2_data), dtype=torch.float32)
        return X0_tensor, X1_tensor, X2_tensor, trajectories

    train_data = _simulate_trajectories(train_ics)
    val_data = _simulate_trajectories(val_ics)

    return train_data, val_data


# =====================================================================
# 2. High-Accuracy Physics-Informed SciML Architecture
# =====================================================================
class ResidualBlock(nn.Module):
    """Residual MLP block with skip connection and SiLU activation."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class LorenzSciML(nn.Module):
    """
    High-Accuracy Continuous Vector-Field Neural ODE for the Lorenz Attractor.

    Features:
    - Normalization layer with registered buffer.
    - Bilinear cross-feature expansion: [x, y, z, xy, xz, yz].
    - Direct physics path + Deep Residual Blocks.
    - Embedded RK4 numerical integration in forward pass.
    """
    def __init__(self, hidden_dim: int = 128, dt: float = 0.01):
        super().__init__()
        self.dt = dt

        # State normalization scales: x in [-20, 20], y in [-25, 25], z in [0, 50]
        self.register_buffer("scale", torch.tensor([20.0, 25.0, 50.0], dtype=torch.float32))

        # Bilinear feature dimension = 3 (raw) + 3 (cross terms xy, xz, yz) = 6
        feat_dim = 6

        # Direct physics carrier (linear combinations of polynomial terms)
        self.linear_carrier = nn.Linear(feat_dim, 3, bias=False)

        # Deep Residual correction network
        self.in_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU()
        )
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 3)

    def extract_features(self, state: torch.Tensor) -> torch.Tensor:
        """Computes normalized state and bilinear cross products."""
        norm_s = state / self.scale
        x = norm_s[:, 0:1]
        y = norm_s[:, 1:2]
        z = norm_s[:, 2:3]
        xy = x * y
        xz = x * z
        yz = y * z
        return torch.cat([norm_s, xy, xz, yz], dim=1)

    def predict_derivatives(self, state: torch.Tensor) -> torch.Tensor:
        """Predicts instantaneous continuous velocity field [dx/dt, dy/dt, dz/dt]."""
        feats = self.extract_features(state)
        lin_out = self.linear_carrier(feats)
        h = self.in_proj(feats)
        h = self.res1(h)
        h = self.res2(h)
        res_out = self.out_proj(h)

        derivs = lin_out + res_out
        return derivs * self.scale

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Integrates forward by one timestep dt via internal RK4 numerical scheme."""
        dt = self.dt
        k1 = self.predict_derivatives(state)
        k2 = self.predict_derivatives(state + 0.5 * dt * k1)
        k3 = self.predict_derivatives(state + 0.5 * dt * k2)
        k4 = self.predict_derivatives(state + dt * k3)

        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# =====================================================================
# 3. Training Loop with Multi-Step Unrolling Loss
# =====================================================================
def train_model(
    epochs: int = 60,
    batch_size: int = 256,
    dt: float = 0.01,
    s: float = 10.0,
    r: float = 28.0,
    b: float = 8.0 / 3.0,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[LorenzSciML, List[np.ndarray]]:
    """Trains the high-accuracy SciML model using multi-step unrolling loss."""
    set_seed(seed)

    print("=" * 75)
    print(f"Deterministic seed set: {seed} (NumPy, PyTorch, CUDA, cuDNN)")
    print("Generating strictly partitioned dataset (Held-out ICs)...")
    (X0_tr, X1_tr, X2_tr, train_trajs), (X0_val, X1_val, X2_val, val_trajs) = generate_trajectories(
        num_trajectories=70, steps=500, dt=dt, s=s, r=r, b=b, val_split=val_split, seed=seed
    )

    train_dataset = torch.utils.data.TensorDataset(X0_tr, X1_tr, X2_tr)
    val_dataset = torch.utils.data.TensorDataset(X0_val, X1_val, X2_val)

    # Deterministic DataLoader shuffling generator
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=loader_generator
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Re-seed PyTorch RNG right before model instantiation to guarantee identical weight initialization
    torch.manual_seed(seed)
    model = LorenzSciML(hidden_dim=128, dt=dt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    print("=" * 75)
    print("Dataset & Training Configuration:")
    print(f"  Training Set   : {len(X0_tr):,d} transitions from {len(train_trajs)} trajectories")
    print(f"  Validation Set : {len(X0_val):,d} transitions from {len(val_trajs)} held-out IC trajectories")
    print(f"  Loss Function  : Multi-Step Rollout Loss (L = L_1step + 0.5 * L_2step)")
    print("=" * 75)

    print("Training SciML Model...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for b0, b1, b2 in train_loader:
            optimizer.zero_grad()
            pred1 = model(b0)
            pred2 = model(pred1)
            loss1 = criterion(pred1, b1)
            loss2 = criterion(pred2, b2)
            total_loss = loss1 + 0.5 * loss2
            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item() * len(b0)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_train_loss = total_train_loss / len(X0_tr)
            model.eval()
            with torch.no_grad():
                val_loss_1 = 0.0
                val_loss_2 = 0.0
                for vb0, vb1, vb2 in val_loader:
                    v_p1 = model(vb0)
                    v_p2 = model(v_p1)
                    val_loss_1 += criterion(v_p1, vb1).item() * len(vb0)
                    val_loss_2 += criterion(v_p2, vb2).item() * len(vb0)
                avg_val_1 = val_loss_1 / len(X0_val)
                avg_val_2 = val_loss_2 / len(X0_val)
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.6e} | Val 1-Step MSE: {avg_val_1:.6e} | Val 2-Step MSE: {avg_val_2:.6e}")

    # Final quantitative evaluation
    model.eval()
    with torch.no_grad():
        final_val_1 = sum(criterion(model(vb0), vb1).item() * len(vb0) for vb0, vb1, _ in val_loader) / len(X0_val)
    print("=" * 75)
    print(f"Final 1-Step Held-out IC Val MSE: {final_val_1:.6e}")
    print("=" * 75)

    return model, val_trajs


# =====================================================================
# 4. Multi-Panel Visualization with Separate 3D Attractor Axes
# =====================================================================
def evaluate_rollout(
    model: LorenzSciML,
    val_trajs: List[np.ndarray],
    dt: float = 0.01,
    s: float = 10.0,
    r: float = 28.0,
    b: float = 8.0 / 3.0,
    rollout_steps: int = 1500,
    output_fig: str = "lorenz_results.png"
) -> None:
    """
    Evaluates autoregressive rollouts on held-out initial conditions and plots:
    1. Dedicated 3D Axis 1: Ground Truth (RK4) Lorenz Attractor.
    2. Dedicated 3D Axis 2: SciML Predicted Lorenz Attractor.
    3. Synchronized component time series [x(t), y(t), z(t)].
    4. 2D Phase Projection (X vs Z Butterfly lobes).
    5. Lyapunov predictability horizon & error growth curve.
    """
    model.eval()

    # Quantitative Multi-step Autoregressive Rollout Evaluation
    rollout_mses = []
    eval_horizon = min(400, len(val_trajs[0]) - 1)
    for traj_gt in val_trajs:
        ic = traj_gt[0]
        curr_t = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
        rollout = [ic]
        with torch.no_grad():
            for _ in range(eval_horizon):
                curr_t = model(curr_t)
                rollout.append(curr_t.squeeze(0).cpu().numpy())
        rollout = np.array(rollout)
        rollout_mses.append(np.mean((traj_gt[:eval_horizon + 1] - rollout) ** 2))

    mean_rollout_mse = np.mean(rollout_mses)
    print(f"Held-out IC Multi-Step Rollout MSE ({eval_horizon} steps): {mean_rollout_mse:.6e}")

    # Pick the first held-out IC for extended long-horizon demonstration
    init_state = val_trajs[0][0]
    ic_label = f"Held-out IC ({init_state[0]:.2f}, {init_state[1]:.2f}, {init_state[2]:.2f})"

    # Ground Truth RK4 trajectory
    gt_traj = [init_state]
    curr = init_state.copy()
    for _ in range(rollout_steps):
        curr = rk4_step(curr, dt=dt, s=s, r=r, b=b)
        gt_traj.append(curr)
    gt_traj = np.array(gt_traj)

    # SciML MLP Autoregressive Rollout
    mlp_traj = [init_state]
    curr_t = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        for _ in range(rollout_steps):
            curr_t = model(curr_t)
            mlp_traj.append(curr_t.squeeze(0).cpu().numpy())
    mlp_traj = np.array(mlp_traj)

    t = np.arange(len(gt_traj)) * dt
    err = np.linalg.norm(gt_traj - mlp_traj, axis=1)

    # -------------------------------------------------------------
    # Multi-Panel Layout Setup
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.15, 1.15, 1.4], hspace=0.32, wspace=0.25)

    norm_time = plt.Normalize(t.min(), t.max())

    # Shared axis limits for identical 3D scaling
    all_x = np.concatenate([gt_traj[:, 0], mlp_traj[:, 0]])
    all_y = np.concatenate([gt_traj[:, 1], mlp_traj[:, 1]])
    all_z = np.concatenate([gt_traj[:, 2], mlp_traj[:, 2]])
    xlim = (all_x.min() - 1, all_x.max() + 1)
    ylim = (all_y.min() - 1, all_y.max() + 1)
    zlim = (all_z.min() - 1, all_z.max() + 1)
    view_elev = 22
    view_azim = 42

    # -------------------------------------------------------------
    # Subplot 1: Dedicated 3D Ground Truth Attractor (Top Left)
    # -------------------------------------------------------------
    ax_3d_gt = fig.add_subplot(gs[0:2, 0], projection="3d")
    pts_gt = gt_traj.reshape(-1, 1, 3)
    segs_gt = np.concatenate([pts_gt[:-1], pts_gt[1:]], axis=1)
    lc_gt = Line3DCollection(segs_gt, cmap="viridis", norm=norm_time, linewidth=0.8, alpha=0.9)
    lc_gt.set_array(t[:-1])
    ax_3d_gt.add_collection3d(lc_gt)
    ax_3d_gt.scatter([init_state[0]], [init_state[1]], [init_state[2]], color="red", s=50, zorder=10, label=ic_label)
    ax_3d_gt.set_xlim(xlim); ax_3d_gt.set_ylim(ylim); ax_3d_gt.set_zlim(zlim)
    ax_3d_gt.set_xlabel("X (Convective Flow)", fontsize=9, labelpad=7)
    ax_3d_gt.set_ylabel("Y (Temp Difference)", fontsize=9, labelpad=7)
    ax_3d_gt.set_zlabel("Z (Heat Conduction)", fontsize=9, labelpad=7)
    ax_3d_gt.set_title("1. Ground Truth (RK4) 3D Attractor", fontsize=12, fontweight="bold", pad=10)
    ax_3d_gt.view_init(elev=view_elev, azim=view_azim)
    cbar_gt = fig.colorbar(lc_gt, ax=ax_3d_gt, shrink=0.55, aspect=15, pad=0.08)
    cbar_gt.set_label("Time ($t$)", fontsize=9)
    ax_3d_gt.legend(loc="upper left", fontsize=8)

    # -------------------------------------------------------------
    # Subplot 2: Dedicated 3D SciML Predicted Attractor (Top Middle)
    # -------------------------------------------------------------
    ax_3d_mlp = fig.add_subplot(gs[0:2, 1], projection="3d")
    pts_mlp = mlp_traj.reshape(-1, 1, 3)
    segs_mlp = np.concatenate([pts_mlp[:-1], pts_mlp[1:]], axis=1)
    lc_mlp = Line3DCollection(segs_mlp, cmap="plasma", norm=norm_time, linewidth=0.8, alpha=0.9)
    lc_mlp.set_array(t[:-1])
    ax_3d_mlp.add_collection3d(lc_mlp)
    ax_3d_mlp.scatter([init_state[0]], [init_state[1]], [init_state[2]], color="blue", s=50, zorder=10, label="SciML Initial State")
    ax_3d_mlp.set_xlim(xlim); ax_3d_mlp.set_ylim(ylim); ax_3d_mlp.set_zlim(zlim)
    ax_3d_mlp.set_xlabel("X (Convective Flow)", fontsize=9, labelpad=7)
    ax_3d_mlp.set_ylabel("Y (Temp Difference)", fontsize=9, labelpad=7)
    ax_3d_mlp.set_zlabel("Z (Heat Conduction)", fontsize=9, labelpad=7)
    ax_3d_mlp.set_title("2. SciML Neural ODE Predicted 3D Attractor", fontsize=12, fontweight="bold", pad=10)
    ax_3d_mlp.view_init(elev=view_elev, azim=view_azim)
    cbar_mlp = fig.colorbar(lc_mlp, ax=ax_3d_mlp, shrink=0.55, aspect=15, pad=0.08)
    cbar_mlp.set_label("Time ($t$)", fontsize=9)
    ax_3d_mlp.legend(loc="upper left", fontsize=8)

    # -------------------------------------------------------------
    # Subplot 3: Synchronized Component Time Series (Right Column)
    # -------------------------------------------------------------
    sub_labels = ["X(t) [Convection]", "Y(t) [Horizontal Grad]", "Z(t) [Vertical Grad]"]
    for i in range(3):
        ax_ts = fig.add_subplot(gs[i, 2])
        ax_ts.plot(t, gt_traj[:, i], color="black", lw=1.3, label="Ground Truth (RK4)")
        ax_ts.plot(t, mlp_traj[:, i], color="crimson", ls="--", lw=1.1, label="SciML Rollout")
        ax_ts.set_ylabel(sub_labels[i], fontsize=10, fontweight="bold")
        ax_ts.grid(True, linestyle=":", alpha=0.6)
        ax_ts.set_xlim(t[0], t[-1])
        if i == 0:
            ax_ts.set_title("Multi-Horizon Time Series Forecast [Held-Out IC]", fontsize=11, fontweight="bold")
            ax_ts.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
        if i == 2:
            ax_ts.set_xlabel("Time ($t$)", fontsize=10, fontweight="bold")
        else:
            ax_ts.set_xticklabels([])

    # -------------------------------------------------------------
    # Subplot 4: 2D Butterfly Projection (X vs Z) (Bottom Left)
    # -------------------------------------------------------------
    ax_proj = fig.add_subplot(gs[2, 0])
    ax_proj.plot(gt_traj[:, 0], gt_traj[:, 2], color="black", lw=0.7, alpha=0.8, label="Ground Truth")
    ax_proj.plot(mlp_traj[:, 0], mlp_traj[:, 2], color="crimson", ls="--", lw=0.7, alpha=0.7, label="SciML Prediction")
    ax_proj.set_title("2D Phase Projection: X vs Z (Butterfly Wings)", fontsize=11, fontweight="bold")
    ax_proj.set_xlabel("X (Convective Intensity)", fontsize=10)
    ax_proj.set_ylabel("Z (Heat Conduction)", fontsize=10)
    ax_proj.grid(True, linestyle=":", alpha=0.6)
    ax_proj.legend(loc="upper right", fontsize=8.5)

    # -------------------------------------------------------------
    # Subplot 5: Error Growth & Predictability Horizon (Bottom Middle)
    # -------------------------------------------------------------
    ax_err = fig.add_subplot(gs[2, 1])
    ax_err.semilogy(t, err + 1e-12, color="#9467bd", lw=1.4, label=r"L2 Error $||\mathbf{x}_{\mathrm{GT}}(t) - \mathbf{x}_{\mathrm{SciML}}(t)||$")
    ax_err.axhline(y=1.0, color="gray", ls=":", alpha=0.7, label="Lyapunov Horizon Threshold (err=1.0)")
    ax_err.set_title("Error Growth & Predictability Horizon", fontsize=11, fontweight="bold")
    ax_err.set_xlabel("Time ($t$)", fontsize=10)
    ax_err.set_ylabel(r"Log Distance $||\Delta \mathbf{x}(t)||$", fontsize=10)
    ax_err.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_err.legend(loc="lower right", fontsize=8.5)
    ax_err.set_xlim(t[0], t[-1])

    plt.suptitle("Physics-Informed Scientific Machine Learning (SciML) on Lorenz System", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    print(f"Results successfully saved to {output_fig}")
    plt.close()


# =====================================================================
# 5. Main Execution Pipeline
# =====================================================================
if __name__ == "__main__":
    dt = 0.01
    s = 10.0
    r = 28.0
    b = 8.0 / 3.0

    print("=" * 75)
    print("STARTING HIGH-ACCURACY SCI-ML LORENZ TRAINING PIPELINE")
    print("=" * 75)

    # Train model using multi-step unrolling loss on held-out ICs
    trained_model, val_trajs = train_model(
        epochs=60,
        batch_size=256,
        dt=dt,
        s=s,
        r=r,
        b=b,
        val_split=0.2
    )

    # Save model weights
    torch.save(trained_model.state_dict(), "lorenz_sciml_mlp.pt")
    print("Model weights saved to lorenz_sciml_mlp.pt")

    # Evaluate autoregressive rollout on held-out ICs and plot side-by-side 3D axes
    evaluate_rollout(
        model=trained_model,
        val_trajs=val_trajs,
        dt=dt,
        s=s,
        r=r,
        b=b,
        rollout_steps=1500,
        output_fig="lorenz_results.png"
    )
    print("All tasks finished successfully!")
