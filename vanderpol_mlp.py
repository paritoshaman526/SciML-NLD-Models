import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 0. Deterministic Reproducibility
# ==========================================
def set_seed(seed=42):
    """Guarantees 100% bit-for-bit reproducibility across all environments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ==========================================
# 1. Data Generation (Ground Truth via RK4)
# ==========================================
def vanderpol_deriv(state, mu=1.0):
    x, v = state[0], state[1]
    dx = v
    dv = mu * (1.0 - x**2) * v - x
    return np.array([dx, dv])

def rk4_step(state, dt, mu=1.0):
    k1 = vanderpol_deriv(state, mu)
    k2 = vanderpol_deriv(state + 0.5 * dt * k1, mu)
    k3 = vanderpol_deriv(state + 0.5 * dt * k2, mu)
    k4 = vanderpol_deriv(state + dt * k3, mu)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def generate_trajectories(num_trajectories=60, steps=250, dt=0.05, mu=1.0, val_split=0.2, seed=42):
    """
    Generates training and validation data partitioned strictly by Initial Condition (held-out ICs).
    This prevents temporal data leakage across train and validation splits.
    """
    set_seed(seed)
    # Sample initial conditions in phase space [-2.5, 2.5] x [-3.0, 3.0]
    init_states = np.random.uniform(low=[-2.5, -3.0], high=[2.5, 3.0], size=(num_trajectories, 2))

    n_val = int(num_trajectories * val_split)
    val_ics = init_states[:n_val]
    train_ics = init_states[n_val:]

    def _simulate_trajectories(ics):
        X_data = [] # [x_t, v_t]
        Y_data = [] # [x_{t+1}, v_{t+1}]
        trajectories = [] # full state trajectories [steps + 1, 2]
        for state0 in ics:
            curr = state0
            traj = [curr]
            for _ in range(steps):
                nxt = rk4_step(curr, dt, mu)
                X_data.append(curr)
                Y_data.append(nxt)
                traj.append(nxt)
                curr = nxt
            trajectories.append(np.array(traj))

        X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32)
        Y_tensor = torch.tensor(np.array(Y_data), dtype=torch.float32)
        return X_tensor, Y_tensor, trajectories

    train_data = _simulate_trajectories(train_ics)
    val_data = _simulate_trajectories(val_ics)

    return train_data, val_data


# ==========================================
# 2. Physics-Constrained MLP Architecture
# ==========================================
class VanDerPolMLP(nn.Module):
    def __init__(self, hidden_dim=64, dt=0.05):
        super().__init__()
        self.dt = dt
        
        # Acceleration network: (x, v) -> x_ddot
        self.accel_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """
        Integrates forward by one timestep dt using semi-implicit Euler
        state: [batch_size, 2] -> (x, v)
        """
        x = state[:, 0:1]
        v = state[:, 1:2]

        # Network only predicts acceleration (second derivative)
        accel = self.accel_net(state)

        # Kinematic integration step
        v_next = v + self.dt * accel
        x_next = x + self.dt * v_next # semi-implicit Euler update

        return torch.cat([x_next, v_next], dim=1)

    def predict_derivatives(self, state):
        """Returns [dx/dt, dv/dt] for vector field visualization."""
        x = state[:, 0:1]
        v = state[:, 1:2]
        accel = self.accel_net(state)
        return torch.cat([v, accel], dim=1)

# ==========================================
# 3. Training Loop
# ==========================================
def train_model(epochs=60, batch_size=128, dt=0.05, mu=1.0, val_split=0.2, seed=42):
    set_seed(seed)
    (X_train, Y_train, train_trajs), (X_val, Y_val, val_trajs) = generate_trajectories(
        num_trajectories=60, steps=250, dt=dt, mu=mu, val_split=val_split, seed=seed
    )
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=loader_generator
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    model = VanDerPolMLP(hidden_dim=64, dt=dt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    print("=" * 65)
    print("Held-out Initial Condition (IC) Dataset Partition:")
    print(f"  Training Set   : {len(X_train):,d} transitions from {len(train_trajs)} IC trajectories")
    print(f"  Validation Set : {len(X_val):,d} transitions from {len(val_trajs)} held-out IC trajectories")
    print("=" * 65)

    print("Training MLP...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred_y = model(batch_x)
            loss = criterion(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(batch_x)
        
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_train_loss = total_train_loss / len(X_train)
            model.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                for v_x, v_y in val_loader:
                    v_pred = model(v_x)
                    total_val_loss += criterion(v_pred, v_y).item() * len(v_x)
                avg_val_loss = total_val_loss / len(X_val)
            print(f"Epoch {epoch+1:02d}/{epochs} | Train MSE: {avg_train_loss:.6f} | Val MSE (Held-out ICs): {avg_val_loss:.6f}")

    # Compute final 1-step validation loss on held-out ICs
    model.eval()
    with torch.no_grad():
        final_val_loss = 0.0
        for v_x, v_y in val_loader:
            final_val_loss += criterion(model(v_x), v_y).item() * len(v_x)
        final_val_loss /= len(X_val)
    print("=" * 65)
    print(f"Final 1-Step Held-out IC Val MSE: {final_val_loss:.6e}")
    print("=" * 65)

    return model, val_trajs

# ==========================================
# 4. Long-Horizon Rollout & Evaluation
# ==========================================
def evaluate_rollout(model, val_trajs=None, dt=0.05, mu=1.0, rollout_steps=400, output_fig="vanderpol_results.png"):
    model.eval()

    # 1. Quantitative Multi-step Autoregressive Rollout Evaluation on Held-Out ICs
    if val_trajs is not None and len(val_trajs) > 0:
        rollout_mses = []
        for traj_gt in val_trajs:
            ic = traj_gt[0]
            steps = len(traj_gt) - 1
            curr_t = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
            rollout = [ic]
            with torch.no_grad():
                for _ in range(steps):
                    curr_t = model(curr_t)
                    rollout.append(curr_t.squeeze(0).cpu().numpy())
            rollout = np.array(rollout)
            rollout_mses.append(np.mean((traj_gt - rollout) ** 2))

        mean_val_rollout_mse = np.mean(rollout_mses)
        print(f"Held-out IC Multi-Step Rollout MSE (mean across {len(val_trajs)} trajectories, {len(val_trajs[0])-1} steps): {mean_val_rollout_mse:.6e}")

        # Pick the first held-out IC trajectory for visualization
        init_state = val_trajs[0][0]
        ic_label = f"Held-out IC ({init_state[0]:.2f}, {init_state[1]:.2f})"
        title_suffix = "[Held-out IC]"
    else:
        init_state = np.array([0.1, 0.1])
        ic_label = "Start (0.1, 0.1)"
        title_suffix = ""

    # Ground Truth trajectory
    gt_traj = [init_state]
    curr = init_state.copy()
    for _ in range(rollout_steps):
        curr = rk4_step(curr, dt, mu)
        gt_traj.append(curr)
    gt_traj = np.array(gt_traj)

    # MLP Autoregressive Rollout
    mlp_traj = [init_state]
    curr_t = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        for _ in range(rollout_steps):
            curr_t = model(curr_t)
            mlp_traj.append(curr_t.squeeze(0).cpu().numpy())
    mlp_traj = np.array(mlp_traj)

    # Plot Phase Portrait and Time Series
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Phase Space (x vs v)
    axes[0].plot(gt_traj[:, 0], gt_traj[:, 1], 'k-', label='Ground Truth (RK4)', linewidth=2)
    axes[0].plot(mlp_traj[:, 0], mlp_traj[:, 1], 'r--', label='MLP Rollout', linewidth=1.8)
    axes[0].scatter([init_state[0]], [init_state[1]], color='blue', label=ic_label, zorder=5)
    axes[0].set_title(rf"Van der Pol Phase Portrait ($\mu={mu}$) {title_suffix}".strip())
    axes[0].set_xlabel("Position ($x$)")
    axes[0].set_ylabel("Velocity ($v$)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # 2. Position Time Series
    t = np.arange(len(gt_traj)) * dt
    axes[1].plot(t, gt_traj[:, 0], 'k-', label='Ground Truth $x(t)$', linewidth=2)
    axes[1].plot(t, mlp_traj[:, 0], 'r--', label='MLP $x(t)$', linewidth=1.8)
    axes[1].set_title(f"Long-Horizon State Trajectory {title_suffix}".strip())
    axes[1].set_xlabel("Time ($t$)")
    axes[1].set_ylabel("Position ($x$)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    print(f"Results saved to {output_fig}")
    plt.close()

if __name__ == "__main__":
    dt = 0.05
    mu = 1.0
    trained_model, val_trajs = train_model(epochs=60, dt=dt, mu=mu, val_split=0.2)
    evaluate_rollout(trained_model, val_trajs=val_trajs, dt=dt, mu=mu, output_fig="vanderpol_results.png")

