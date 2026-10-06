# Scientific Machine Learning Validation Report: FitzHugh-Nagumo Model
## Controlled Architectural Dichotomy: From-Scratch Vector Field vs. Physics-Residual Embedding

**Repository Locations:**
- Workspace Development: [`fitzhugh_nagumo/`](file:///c:/Users/satya/Downloads/Van%20Der%20Pol/fitzhugh_nagumo)
- Synchronized Target Folder: [`FitzHugh Nagumo/`](file:///c:/Users/satya/Downloads/FitzHugh%20Nagumo)

---

## 1. Executive Summary & Formulation

The FitzHugh-Nagumo (FHN) system provides the canonical model for excitable dynamical systems and neuronal action-potential generation:
$$\frac{dv}{dt} = v - \frac{v^3}{3} - w + I$$
$$\frac{dw}{dt} = \epsilon (v + a - b w)$$

### Analytical Bifurcation Ground Truth
The unique fixed point $(v^*, w^*)$ is obtained from the cubic nullcline intersection $v^3/3 + (1/b - 1)v + (a/b - I) = 0$ with $w^* = (v^* + a)/b$. The Jacobian trace yields the exact analytical Hopf bifurcation condition:
$$\text{Tr}(J) = 1 - (v^*)^2 - \epsilon b = 0 \implies |v^*| = \sqrt{1 - \epsilon b}$$
$$I_{\text{Hopf}}(a, b) = \frac{(v^*)^3}{3} + \left(\frac{1}{b} - 1\right)v^* + \frac{a}{b}$$
For canonical baseline parameters ($a=0.70, b=0.80, \epsilon=0.08$), the lower supercritical Hopf threshold is $I_{\text{H1}} = 0.3313$ (onset of limit-cycle spiking).

---

## 2. The Controlled Architectural Dichotomy

Rather than presenting an unverified hybrid model or reverting to pure data, we present **both architectural paradigms side-by-side** under identical 7-seed evaluation rigor:

1. **Model A: Pure Data / From-Scratch FiLM SciML**:
   - Neural network learns the complete autonomous vector field $\mathbf{f}_\theta(\mathbf{y}, \mathbf{p})$ from trajectory observations without baking in physical terms.
   - **Purpose**: Serves as a genuine test of neural capacity limits, exposing how steep manifold curvature ($\kappa(b)$) induces boundary tracking lag and saturation.
2. **Model B: Hybrid / Physics-Residual FiLM SciML**:
   - Vector field embeds prior physical kinetics ($\dot{v}_{\text{phys}}, \dot{w}_{\text{phys}}$) with bounded parameter-conditioned residual modulation ($\pm 10\%$ on $\dot{v}$, $\pm 5\%$ on $\dot{w}$).
   - **Purpose**: Evaluates grey-box error bounding, matching the formulation of the biochemical oscillator.

---

## 3. Multi-Seed Replication Benchmark (Table 1 Format)

**Benchmark Parameter Set:** $I = 0.80, a = 0.70, b = 0.80, \epsilon = 0.08$  
**Ground Truth RK4 Reference:** Period $T = 36.5167\text{ s}$, Amplitude $= 3.8441$, Self-Decay Retention $= 100.00\%$

| Metric / Dimension | Baseline Generic MLP | **Model A: From-Scratch FiLM** | **Model B: Physics-Residual FiLM** |
| :--- | :---: | :---: | :---: |
| **Physics Terms Baked In?** | No (Black-box) | **No (Pure Data Discovery)** | **Yes (Grey-box Residual)** |
| **Limit-Cycle Period $T$** | $37.3333\text{ s}$ ($2.24\%$ err) | **$37.3238 \pm 0.3048\text{ s}$ ($2.21\%$ err)** | **$36.4667 \pm 0.0436\text{ s}$ ($0.13\%$ err)** |
| **Limit-Cycle Amplitude** | $4.0791$ ($106.11\%$) | **$3.8652 \pm 0.0134$ ($100.55\%$)** | **$3.8396 \pm 0.0029$ ($99.88\%$)** |
| **Amplitude Error vs GT** | $+6.11\%$ | **$+0.55 \pm 0.35\%$** | **$-0.12 \pm 0.07\%$** |
| **Self-Decay Retention** | $100.55\%$ | **$100.02 \pm 0.04\%$** | **$100.00 \pm 0.00\%$** |
| **Ensemble Seed Consistency** | High seed scatter | **Unimodal across all 7 seeds** | **Sub-0.1% variance across seeds** |

### Detailed 7-Seed Breakdown for Model A (From-Scratch):
- Seed 42: $T = 37.4000\text{ s}$, $\text{Amp} = 3.8516$ ($100.19\%$)
- Seed 101: $T = 37.8000\text{ s}$, $\text{Amp} = 3.8752$ ($100.81\%$)
- Seed 777: $T = 37.6333\text{ s}$, $\text{Amp} = 3.8389$ ($99.86\%$)
- Seed 999: $T = 37.0167\text{ s}$, $\text{Amp} = 3.8678$ ($100.61\%$)
- Seed 1234: $T = 37.1500\text{ s}$, $\text{Amp} = 3.8713$ ($100.71\%$)
- Seed 2024: $T = 37.3833\text{ s}$, $\text{Amp} = 3.8753$ ($100.81\%$)
- Seed 5555: $T = 36.8833\text{ s}$, $\text{Amp} = 3.8764$ ($100.84\%$)

### Detailed 7-Seed Breakdown for Model B (Physics-Residual):
- Seed 42: $T = 36.4667\text{ s}$, $\text{Amp} = 3.8427$ ($99.96\%$)
- Seed 101: $T = 36.5333\text{ s}$, $\text{Amp} = 3.8398$ ($99.89\%$)
- Seed 777: $T = 36.4333\text{ s}$, $\text{Amp} = 3.8368$ ($99.81\%$)
- Seed 999: $T = 36.4833\text{ s}$, $\text{Amp} = 3.8398$ ($99.89\%$)
- Seed 1234: $T = 36.5167\text{ s}$, $\text{Amp} = 3.8348$ ($99.76\%$, **exact period match**)
- Seed 2024: $T = 36.4167\text{ s}$, $\text{Amp} = 3.8429$ ($99.97\%$)
- Seed 5555: $T = 36.4333\text{ s}$, $\text{Amp} = 3.8376$ ($99.83\%$)

---

## 4. Dense Grid Bifurcation Horizon & Curvature Tracking

Evaluation across 15 slice points in recovery damping parameter $b \in [0.55, 1.25]$ ($a = 0.70, \epsilon = 0.08$):

| $b$ | Analytical Hopf $I_{\text{H1}}$ | Ground Truth RK4 $I_{\text{RK4}}$ | Model A Deviation $\Delta I_{\text{scratch}}$ | Model B Deviation $\Delta I_{\text{residual}}$ | True Slope $s(b)$ | Curvature $\kappa(b)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.550** | 0.1612 | 0.1757 | **+0.0536** | **-0.0008** | 0.6529 | 0.3287 |
| **0.600** | 0.2066 | 0.2050 | **+0.0399** | **-0.0031** | 0.6642 | 0.0621 |
| **0.650** | 0.2450 | 0.2407 | **+0.0224** | **-0.0035** | 0.6421 | 0.4617 |
| **0.700** | 0.2779 | 0.2724 | **+0.0039** | **-0.0049** | 0.5944 | 0.7204 |
| **0.750** | 0.3064 | 0.2999 | **-0.0079** | **-0.0038** | 0.5287 | **1.0318 (Peak)** |
| **0.800** | 0.3313 | 0.3239 | **-0.0176** | **-0.0024** | 0.4598 | 0.9487 |
| **0.850** | 0.3532 | 0.3456 | **-0.0236** | **-0.0037** | 0.4022 | 0.8275 |
| **0.900** | 0.3728 | 0.3644 | **-0.0295** | **-0.0025** | 0.3561 | 0.6751 |
| **0.950** | 0.3902 | 0.3813 | **-0.0336** | **-0.0036** | 0.3183 | 0.6115 |
| **1.000** | 0.4059 | 0.3960 | **-0.0367** | **-0.0026** | 0.2855 | 0.5388 |
| **1.050** | 0.4200 | 0.4101 | **-0.0379** | **-0.0036** | 0.2577 | 0.4587 |
| **1.100** | 0.4329 | 0.4224 | **-0.0374** | **-0.0044** | 0.2350 | 0.3730 |
| **1.150** | 0.4446 | 0.4341 | **-0.0377** | **-0.0020** | 0.2173 | 0.2832 |
| **1.200** | 0.4553 | 0.4440 | **-0.0376** | **-0.0019** | 0.2046 | 0.1906 |
| **1.250** | 0.4651 | 0.4540 | **-0.0361** | **-0.0018** | 0.1970 | 0.0962 |

### Regional Analysis:
- **Region 1 ($b \in [0.55, 0.85]$ - Steep Slope & Curvature Onset):**
  - Model A (From-Scratch): Mean Dev $= +0.0157$, Regional RMSE $= 0.0299$
  - Model B (Physics-Residual): Mean Dev $= -0.0031$, Regional RMSE $= \mathbf{0.0033}$
- **Region 2 ($b \in [0.90, 1.25]$ - Flattening Slope & Asymptotic Regime):**
  - Model A (From-Scratch): Mean Dev $= -0.0345$, Regional RMSE $= 0.0348$ (**plateau lag at $-0.037$**)
  - Model B (Physics-Residual): Mean Dev $= -0.0029$, Regional RMSE $= \mathbf{0.0030}$
- **Overall Dense Grid RMSE**:
  - Model A (From-Scratch): $\mathbf{0.0248}$ (Peak Error: $+0.0536$)
  - Model B (Physics-Residual): $\mathbf{0.0032}$ (Peak Error: $-0.0049$)

---

## 5. Figures: Side-by-Side Architectural Contrast

1. **Model A (From-Scratch Discovery)**: [`fhn_dense_grid_from_scratch.png`](file:///c:/Users/satya/Downloads/FitzHugh%20Nagumo/scratch/fhn_dense_grid_from_scratch.png)  
   *Captures genuine capacity saturation: deviation $\Delta I$ transitions from positive overshoot (+0.054) to negative plateau lag (-0.037) precisely where the true boundary curvature peaks ($\kappa = 1.0318$).*
2. **Model B (Physics-Residual Embedding)**: [`fhn_dense_grid_physics_residual.png`](file:///c:/Users/satya/Downloads/FitzHugh%20Nagumo/scratch/fhn_dense_grid_physics_residual.png)  
   *Demonstrates residual error bounding: deviations are flattened across the entire parameter span to within $\pm 0.0049$ ($\text{RMSE} = 0.0032$).*
3. **4-Pillar Climate & Ergodic Attractor Validation**: [`fhn_climate_ergodic_validation.png`](file:///c:/Users/satya/Downloads/FitzHugh%20Nagumo/scratch/fhn_climate_ergodic_validation.png)  
   *Validates invariant ergodic measure ($W_1 = 0.0034$), exact harmonic Fourier comb ($f_0 = 0.02930\text{ Hz}$), Poincaré return map fixed point ($V_{\max}^* = 1.9172$), and neural Jacobian continuation ($I_{\text{Hopf}} = 0.3315$).*

---

## 6. Synthesis Matrix Across All Four Systems

| System | Dynamical Phenomenon | Architectural Formulation | Key Scientific Insight | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Lorenz 1963** | Chaotic Strange Attractor | **From-Scratch (Pure Data)** | Lyapunov spectrum, $D_{\text{KY}}$, Liouville dissipation, and lobe-switching preserved without baked equations. | **Complete** |
| **FitzHugh-Nagumo (Model A)** | Excitable Limit Cycle / Hopf | **From-Scratch (Pure Data)** | Shows architectural capacity limits: curvature $\kappa(b)$ induces boundary lag and saturation plateau. | **Complete** |
| **FitzHugh-Nagumo (Model B)** | Excitable Limit Cycle / Hopf | **Physics-Residual (Grey-Box)** | Shows residual bounding: bounds error to $\text{RMSE} = 0.0032$ and reproduces analytical Jacobian crossing. | **Complete** |
| **Van der Pol** | Relaxation Oscillator / Hopf | **From-Scratch (Model A) & Physics-Residual (Model B)** | 7-seed replication complete: Model A period $6.6533\text{ s}$ ($0.15\%$ err), continuous Hopf $\mu_c = +0.0367 \pm 0.0206$; Model B period $6.6487\text{ s}$ ($0.22\%$ err). | **Complete** |
