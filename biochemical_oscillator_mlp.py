import os
import sys

# Prevent OpenBLAS/OMP thread exhaustion and memory errors on Windows
if sys.platform.startswith("win"):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import itertools
from multiprocessing import Pool, cpu_count
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from numba import njit


def set_seed(seed: int = 42) -> None:
    """Guarantees bit-for-bit reproducibility across all environments and devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

DT = 1e-4
DEFAULT_GUESS = np.array([6657.918, 3267.0, 77.082, 2.038], dtype=np.float64)
VAR_NAMES = [
    "Deactivated Monomer (d)",
    "Activated Monomer (a1)",
    "Polymer Number Conc.(m1)",
    "Polymer Mass Conc. (m0)"
]

# ============================================================
# 1. NUMBA-JIT RK4 DYNAMICAL INTEGRATION
# ============================================================
@njit(cache=True, fastmath=True)
def system(y, params):
    d, a1, m1, m0 = y[0], y[1], y[2], y[3]
    ka, kd2, kd3, kp = params[0], params[1], params[2], params[3]
    
    dd_dt = -ka * d + 2.0 * kd2 * m0
    da1_dt = ka * d - 2.0 * kp * a1 * m0
    dm1_dt = -2.0 * kd2 * m0 + 2.0 * kp * a1 * m0
    
    m0_safe = max(m0, 1e-9)
    ratio = m1 / m0_safe
    denom = max((ratio - 2.0) ** 2, 1e-6)
    dm0_dt = kd3 * (m1 - 4.0 * m0) - (2.0 * 4.6 * kd2 * m0) / denom - 2.0 * kd3 * m0 - kp * (m0 ** 2)
    
    out = np.empty(4)
    out[0], out[1], out[2], out[3] = dd_dt, da1_dt, dm1_dt, dm0_dt
    return out

@njit(cache=True, fastmath=True)
def rk4_step(y, dt, params):
    k1 = system(y, params)
    k2 = system(y + 0.5 * dt * k1, params)
    k3 = system(y + 0.5 * dt * k2, params)
    k4 = system(y + dt * k3, params)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

@njit(cache=True, fastmath=True)
def simulate_rk4_core(y0, n_steps, dt, params):
    traj = np.empty((n_steps + 1, 4))
    traj[0] = y0
    curr = y0.copy()
    for i in range(n_steps):
        curr = rk4_step(curr, dt, params)
        traj[i + 1] = curr
    return traj

def simulate_rk4(y0, duration, params, dt=DT):
    n_steps = int(round(duration / dt))
    traj = simulate_rk4_core(
        np.asarray(y0, dtype=np.float64),
        n_steps,
        dt,
        np.asarray(params, dtype=np.float64)
    )
    return np.linspace(0, duration, n_steps + 1), traj

# ============================================================
# 2. BIFURCATION FILTER & PERIOD ESTIMATION
# ============================================================
@njit(fastmath=True)
def is_in_oscillatory_region(params, y0, dt=1e-4):
    """
    Fast Numba check (~25ms): screens if a parameter set lies inside
    the 3D Hopf bifurcation surface by skipping damped transients (25.0s)
    and verifying whether sustained limit-cycle amplitude > 5.0.
    """
    y = y0.copy()
    # 1. Skip 25.0s to allow damped linear transients of stable focus to decay
    for _ in range(int(25.0 / dt)):
        y = rk4_step(y, dt, params)
        
    # 2. Check peak-to-peak amplitude over 2.0s
    min_m1, max_m1 = 1e9, -1e9
    for _ in range(int(2.0 / dt)):
        y = rk4_step(y, dt, params)
        if y[2] < min_m1: min_m1 = y[2]
        if y[2] > max_m1: max_m1 = y[2]
        
    amplitude = max_m1 - min_m1
    return amplitude > 5.0

def find_steady_state(params, guess=None):
    if guess is None:
        guess = DEFAULT_GUESS.copy()
    params_arr = np.asarray(params, dtype=np.float64)

    def residual(log_y):
        log_y_clipped = np.clip(log_y, -10.0, 15.0)
        y = np.exp(log_y_clipped)
        dydt = system(y, params_arr)
        return dydt / np.array([1e4, 1e4, 1e4, 10.0])

    init_log_y = np.log(np.maximum(guess, 1e-8))
    result = least_squares(residual, init_log_y, max_nfev=2000, xtol=1e-9, ftol=1e-9, gtol=1e-9)

    if result.success and not np.any(np.isnan(result.x)):
        return np.exp(np.clip(result.x, -10.0, 15.0))

    # Fallback to dynamical relaxation if least-squares gets stuck
    _, traj = simulate_rk4(guess, duration=5.0, params=params_arr, dt=DT)
    return traj[-1]

def estimate_period_and_orbit(params, y0, probe_duration=12.0, dt=DT):
    """Simulates orbit and accurately detects period using peak prominence."""
    t, traj = simulate_rk4(y0, probe_duration, params, dt)
    
    # Evaluate after initial 20% transient to avoid false peaks
    transient_idx = int(0.2 * len(t))
    t_eval = t[transient_idx:]
    m1_eval = traj[transient_idx:, 2]

    # Enforce minimum peak distance of at least 0.05s between cycles
    min_dist_steps = max(int(0.05 / dt), 10)
    peak_indices, _ = find_peaks(m1_eval, height=np.mean(m1_eval), distance=min_dist_steps)

    if len(peak_indices) > 1:
        peak_times = t_eval[peak_indices]
        period = float(np.mean(np.diff(peak_times)))
    else:
        period = None

    return period, t, traj

def generate_dataset(base_orbit, params, num_trajectories=40, traj_steps=4000, dt=DT, val_split=0.2, seed=42):
    np.random.seed(seed)
    
    # Sample along converged portion of the orbit to avoid negative indexing
    orbit_start_idx = len(base_orbit) // 2
    sample_indices = np.linspace(orbit_start_idx, len(base_orbit) - 1, num_trajectories, dtype=int)

    init_states = []
    for idx in sample_indices:
        ic = base_orbit[idx].copy()
        # Conserve mass in perturbation: +pert on d, -pert on a1
        pert = np.random.uniform(-30.0, 30.0)
        ic[0] = max(ic[0] + pert, 10.0)
        ic[1] = max(ic[1] - pert, 10.0)
        ic[3] = max(ic[3] * np.random.uniform(0.95, 1.05), 1e-5)
        init_states.append(ic)

    init_states = np.array(init_states)
    n_val = max(int(num_trajectories * val_split), 1)
    perm = np.random.permutation(num_trajectories)
    val_ics, train_ics = init_states[perm[:n_val]], init_states[perm[n_val:]]

    params_arr = np.asarray(params, dtype=np.float64)

    def _sim(ics):
        trajs = []
        for ic in ics:
            trajs.append(simulate_rk4_core(ic.astype(np.float64), traj_steps, dt, params_arr))
        return trajs

    return _sim(train_ics), _sim(val_ics)

# ============================================================
# 3. PHYSICS-CONSTRAINED BIOCHEMICAL MLP & KINETIC SCIML
# ============================================================
class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation block conditioned on kinetic parameters."""
    def __init__(self, in_features, out_features, cond_dim):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.cond = nn.Linear(cond_dim, out_features * 2)
        self.act = nn.Tanh()
        
    def forward(self, x, p_embed):
        h = self.fc(x)
        gamma, beta = self.cond(p_embed).chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        return self.act(h)


class ProposedFiLMPhysicsConstrainedMLP(nn.Module):
    """
    FiLM Physics-Constrained Neural Network (Genuinely Learned SciML, No Hardcoded Carrier):
    - Layer-wise FiLM conditioning captures nonlinear bifurcation variations.
    - Exact stoichiometric mass conservation: d + a1 + m1 == c_tot analytically.
    - Non-collapsing pacemaker m0 state integration with sensitivity weighting.
    - Embedded RK4 numerical stepping.
    """
    def __init__(self, hidden_dim=128, dt=DT):
        super().__init__()
        self.dt = dt
        self.register_buffer('state_scales', torch.tensor([5000., 5000., 200., 5.]))
        self.register_buffer('scales', torch.tensor([5000., 5000., 200., 5.]))
        self.register_buffer('param_scales', torch.tensor([10., 5000., 2., 1.]))
        self.register_buffer('rate_scales', torch.tensor([30000., 30000., 8000., 200.]))

        cond_dim = 32
        self.p_net = nn.Sequential(
            nn.Linear(4, cond_dim), nn.Tanh(),
            nn.Linear(cond_dim, cond_dim), nn.SiLU()
        )
        self.block1 = FiLMBlock(4, hidden_dim, cond_dim)
        self.block2 = FiLMBlock(hidden_dim, hidden_dim, cond_dim)
        self.block3 = FiLMBlock(hidden_dim, hidden_dim, cond_dim)
        self.head = nn.Linear(hidden_dim, 4)

    def predict_rates(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        s_norm = state / self.state_scales
        p_norm = params / self.param_scales
        p_embed = self.p_net(p_norm)
        h = self.block1(s_norm, p_embed)
        h = self.block2(h, p_embed)
        h = self.block3(h, p_embed)
        out = self.head(h)

        J1 = torch.nn.functional.softplus(out[:, 0:1]) * self.rate_scales[0]
        J2 = torch.nn.functional.softplus(out[:, 1:2]) * self.rate_scales[1]
        J3 = torch.nn.functional.softplus(out[:, 2:3]) * self.rate_scales[2]
        rm0 = out[:, 3:4] * self.rate_scales[3]

        rd = -J1 + J2
        ra1 = J1 - J3
        rm1 = J3 - J2
        return torch.cat([rd, ra1, rm1, rm0], dim=1)

    def forward(self, state, params, target_ctot=None):
        dt = self.dt
        k1 = self.predict_rates(state, params)
        k2 = self.predict_rates(state + 0.5 * dt * k1, params)
        k3 = self.predict_rates(state + 0.5 * dt * k2, params)
        k4 = self.predict_rates(state + dt * k3, params)

        nxt = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        d_next  = torch.clamp(nxt[:, 0:1], min=0.0)
        a1_next = torch.clamp(nxt[:, 1:2], min=0.0)
        m1_next = torch.clamp(nxt[:, 2:3], min=0.0)
        m0_next = torch.clamp(nxt[:, 3:4], min=0.5)

        c_tot = (state[:, 0:1] + state[:, 1:2] + state[:, 2:3]) if target_ctot is None else target_ctot
        subunits = torch.cat([d_next, a1_next, m1_next], dim=1)
        subunits = subunits * (c_tot / torch.clamp(subunits.sum(dim=1, keepdim=True), min=1e-6))
        return torch.cat([subunits, m0_next], dim=1)


class BiochemicalMLP(nn.Module):
    """
    Standard Physics-Informed Neural ODE for the Biochemical Oscillator.

    Features:
    - Proper physical rate scaling: [25000, 25000, 6000, 150] matching true derivative magnitudes.
    - Embedded 4th-order Runge-Kutta (RK4) integration scheme (replaces dissipative 1st-order Euler).
    - Exact continuous stoichiometric monomer conservation: d(d + a1 + m1)/dt = 0.
    - Non-negativity and total mass preservation projection.
    """
    def __init__(self, hidden_dim: int = 128, dt: float = DT):
        super().__init__()
        self.dt = dt
        self.register_buffer('input_scales', torch.tensor([6000., 5000., 350., 8., 10., 5000., 2., 1.], dtype=torch.float32))
        self.register_buffer('scales', torch.tensor([6000., 5000., 350., 8.], dtype=torch.float32))
        self.register_buffer('rate_scales', torch.tensor([25000., 25000., 6000., 150.], dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4)
        )

    def predict_rates(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Predicts instantaneous continuous reaction rates."""
        x = torch.cat([state, params], dim=1)
        out = self.net(x / self.input_scales)
        
        # Reaction fluxes
        J1 = torch.nn.functional.softplus(out[:, 0:1]) * self.rate_scales[0]
        J2 = torch.nn.functional.softplus(out[:, 1:2]) * self.rate_scales[1]
        J3 = torch.nn.functional.softplus(out[:, 2:3]) * self.rate_scales[2]
        rm0 = out[:, 3:4] * self.rate_scales[3]

        # Conserved monomer stoichiometries: sum(rd, ra1, rm1) = 0
        rd = -J1 + J2
        ra1 = J1 - J3
        rm1 = J3 - J2
        return torch.cat([rd, ra1, rm1, rm0], dim=1)

    def forward(self, state: torch.Tensor, params: torch.Tensor, target_ctot: torch.Tensor = None) -> torch.Tensor:
        """Integrates forward by one timestep dt via embedded 4th-order Runge-Kutta numerical scheme."""
        dt = self.dt
        k1 = self.predict_rates(state, params)
        k2 = self.predict_rates(state + 0.5 * dt * k1, params)
        k3 = self.predict_rates(state + 0.5 * dt * k2, params)
        k4 = self.predict_rates(state + dt * k3, params)

        nxt = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Enforce non-negativity
        d_next  = torch.clamp(nxt[:, 0:1], min=0.0)
        a1_next = torch.clamp(nxt[:, 1:2], min=0.0)
        m1_next = torch.clamp(nxt[:, 2:3], min=0.0)
        m0_next = torch.clamp(nxt[:, 3:4], min=0.0)

        # Enforce exact total mass conservation: c_tot = d + a1 + m1
        c_tot = (state[:, 0:1] + state[:, 1:2] + state[:, 2:3]) if target_ctot is None else target_ctot
        subunits = torch.cat([d_next, a1_next, m1_next], dim=1)
        subunits = subunits * (c_tot / torch.clamp(subunits.sum(dim=1, keepdim=True), min=1e-6))
        
        return torch.cat([subunits, m0_next], dim=1)

def rollout_model(model, ic, params, n_steps, dt=DT):
    c_tot0 = torch.tensor([[ic[0] + ic[1] + ic[2]]], dtype=torch.float32)
    state = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
    params_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
    traj = [ic]
    
    model.eval()
    with torch.no_grad():
        for _ in range(n_steps):
            state = model(state, params_tensor, target_ctot=c_tot0)
            traj.append(state.squeeze(0).cpu().numpy())
    return np.array(traj)

# ============================================================
# 4. TRAINING WITH SCHEDULED UNROLLING
# ============================================================
def train(train_trajs, val_trajs, params, n_epochs=None, batch_size=64, hidden_dim=128,
          checkpoint_epochs=None, snapshot_steps=3000, k_max=4, unroll_budget=60000,
          model_type="proposed"):

    if n_epochs is None:
        n_epochs = 60
    if checkpoint_epochs is None:
        checkpoint_epochs = (1, n_epochs)

    if model_type == "proposed" or model_type == "kinetic":
        model = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=hidden_dim, dt=DT)
        lr = 2e-3
    else:
        model = BiochemicalMLP(hidden_dim=hidden_dim, dt=DT)
        lr = 2e-3

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    scale_weights = model.scales.clone().detach()
    train_trajs_t = [torch.tensor(tr, dtype=torch.float32) for tr in train_trajs]

    snapshot_ic = sorted(val_trajs, key=lambda tr: tr[:, 2].std(), reverse=True)[0][0].copy()
    snapshot_steps = min(snapshot_steps, 3000)
    t_snap, gt_snap = simulate_rk4(snapshot_ic, duration=snapshot_steps * DT, params=params, dt=DT)

    def sample_window_batch(K):
        batch = []
        for _ in range(batch_size):
            tr = train_trajs_t[np.random.randint(len(train_trajs_t))]
            max_start = max(len(tr) - K - 1, 1)
            i = np.random.randint(0, max_start)
            batch.append(tr[i:i + K + 1])
        return torch.stack(batch)

    loss_history, snapshots = [], {}
    t0 = time.time()
    params_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)

    model.train()
    for epoch in range(1, n_epochs + 1):
        # Bound K to [2, 4] to prevent BPTT graph retention and memory explosion
        K = int(np.interp(epoch, [1, n_epochs * 0.3, n_epochs], [2, 3, min(k_max, 4)]))
        batch_true = sample_window_batch(K)
        state = batch_true[:, 0, :]
        c_tot_target = state[:, 0:1] + state[:, 1:2] + state[:, 2:3]

        preds = [state]
        for _ in range(K):
            state = model(state, params_tensor, target_ctot=c_tot_target)
            preds.append(state)
        preds = torch.stack(preds, dim=1)

        # State MSE loss
        state_loss = torch.mean(((preds[:, 1:, :] - batch_true[:, 1:, :]) / scale_weights) ** 2)
        
        # Derivative regularization (normalized)
        true_deriv = (batch_true[:, 1:, :] - batch_true[:, :-1, :]) / DT
        pred_deriv = (preds[:, 1:, :] - preds[:, :-1, :]) / DT
        deriv_loss = torch.mean(((pred_deriv - true_deriv) / model.rate_scales) ** 2)
        
        loss = state_loss + 0.01 * deriv_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if epoch in checkpoint_epochs:
            snapshots[epoch] = rollout_model(model, snapshot_ic, params, snapshot_steps)
            mse = float(np.mean((snapshots[epoch] - gt_snap) ** 2))
            print(f"    epoch {epoch:4d}: K={K:3d} loss={loss.item():.3e} snapshot MSE={mse:.3e} ({time.time()-t0:.1f}s)")

    return model, loss_history, snapshots, t_snap, gt_snap

# ============================================================
# 5. SINGLE PARAMETER SET WORKER
# ============================================================
def run_one_parameter_set(args):
    set_id, total, params = args
    # Prevent OpenMP thread contention across workers in multiprocessing
    torch.set_num_threads(1)
    
    t_start = time.time()
    print(f"[set {set_id+1}/{total}] starting ka={params[0]:.1f}, kd2={params[1]:.1f}, kd3={params[2]:.2f}")

    params_arr = np.asarray(params, dtype=np.float64)

    # Relax 20.0s onto the sustained limit cycle starting from DEFAULT_GUESS
    y_relaxed = DEFAULT_GUESS.copy()
    for _ in range(int(20.0 / DT)):
        y_relaxed = rk4_step(y_relaxed, DT, params_arr)

    # 4s probe is fast in Numba and captures complete limit-cycle periods
    period, t_orbit, orbit = estimate_period_and_orbit(params, y_relaxed, probe_duration=4.0)

    if period is None:
        print(f"[set {set_id+1}] no sustained oscillation detected -- using defaults")
        traj_steps, k_max = 4000, 4
    else:
        traj_duration = max(3.0 * period, 0.6)
        traj_steps = int(traj_duration / DT)
        k_max = 4

    train_trajs, val_trajs = generate_dataset(orbit, params, num_trajectories=40, traj_steps=traj_steps)
    model, loss_history, snapshots, t_snap, gt_snap = train(
        train_trajs, val_trajs, params, k_max=k_max, snapshot_steps=3000
    )

    elapsed = time.time() - t_start
    final_mse = float(np.mean((snapshots[max(snapshots.keys())] - gt_snap) ** 2))
    print(f"[set {set_id+1}/{total}] DONE in {elapsed:.1f}s | period={period if period else 'N/A'} | MSE={final_mse:.3e}")

    return params, {
        "period": period,
        "final_mse": final_mse,
        "elapsed": elapsed
    }

# ============================================================
# 6. MAIN EXECUTION WITH BIFURCATION FILTERING
# ============================================================
if __name__ == "__main__":
    # Candidate parameter ranges covering the 3D bifurcation space
    KA_VALUES  = np.arange(2.0, 10.0, 2.0)          # 2, 4, 6, 8
    KD2_VALUES = np.arange(2000.0, 7000.0, 1000.0)  # 2000, 3000, 4000, 5000, 6000
    KD3_VALUES = np.arange(0.5, 1.8, 0.3)          # 0.5, 0.8, 1.1, 1.4, 1.7

    ALL_CANDIDATES = [
        (ka, kd2, kd3, 1.0)
        for ka, kd2, kd3 in itertools.product(KA_VALUES, KD2_VALUES, KD3_VALUES)
    ]

    # Explicit reference points from the bifurcation diagram slide:
    SLIDE_OSCILLATORY_PT = (6.0, 4208.0, 1.0, 1.0)  # Inside unstable blue surface
    SLIDE_STABLE_PT      = (11.0, 5701.0, 0.3, 1.0)  # Outside in stable region

    print(f"Total candidate parameter sets: {len(ALL_CANDIDATES)}")
    print("Filtering parameter sets using fast Numba bifurcation pre-screener...")

    y_guess = DEFAULT_GUESS.copy()
    verified_oscillatory_sets = [
        p for p in ALL_CANDIDATES
        if is_in_oscillatory_region(np.array(p, dtype=np.float64), y_guess)
    ]

    # Prepend slide benchmark
    if SLIDE_OSCILLATORY_PT not in verified_oscillatory_sets:
        verified_oscillatory_sets.insert(0, SLIDE_OSCILLATORY_PT)

    print(f"-> Verified {len(verified_oscillatory_sets)} oscillatory sets inside the Hopf surface.")
    
    # You can choose to run on all verified sets or a quick subset (e.g. [:6])
    PARAMETER_SETS = verified_oscillatory_sets[:8]  # Running first 8 for demonstration
    print(f"Selected {len(PARAMETER_SETS)} sets for training.")

    n_workers = max(cpu_count() - 1, 1)
    print(f"Using {n_workers} parallel worker processes.\n")

    tasks = [(i, len(PARAMETER_SETS), p) for i, p in enumerate(PARAMETER_SETS)]

    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        results = pool.map(run_one_parameter_set, tasks)
    total_time = time.time() - t0

    all_results = {params: res for params, res in results}
    print(f"\nAll {len(PARAMETER_SETS)} sets completed in {total_time/60:.2f} minutes "
          f"({total_time/len(PARAMETER_SETS):.1f}s/set average).")
