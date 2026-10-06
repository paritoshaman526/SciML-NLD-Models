# Comprehensive Parameter Generalization Study Across Diverse $k_a, k_{d2}, k_{d3}$

> **Experimental Overview**:
> - **Deep Extrapolation ($k_a > 10$) [Stable Region: Pure Steady-State Flat Line]**: Evaluates system behavior well past the nominal training interval ($k_a \in [2, 8]$) and past the supercritical Hopf bifurcation boundary ($k_a \approx 7.1 - 8.2$). As proven by Jacobian eigenvalue analysis ($\mathrm{Re}(\lambda) < 0$), no sustained limit cycles exist in this regime; on the attractor, the system resides in a stationary fixed point where the ground truth is literally a flat horizontal straight line ($T=0, \mathrm{Amp}=0$). Models are evaluated on their ability to maintain the flat line versus hallucinating spurious oscillations.
> - **Diverse Biochemical Parameter Configurations**: Explores extreme corners of parameter space including ultra-low $k_a$ ($k_a=1.0$), low $k_{d2}$ with high $k_{d3}$ ($k_{d2}=2600, k_{d3}=1.1$), elevated $k_{d3}$ ($1.3$), high $k_{d2}$ with low $k_{d3}$ ($k_{d2}=5200, k_{d3}=0.35$), and near-critical Hopf boundaries ($k_a=8.5$).
> - **Multi-Seed Statistical Replication**: All metrics computed across all 7 canonical seeds (`[42, 101, 2024, 777, 999, 1234, 5555]`). Formatted as `Mean ± Std [Median, Min–Max]`.


## ka > 10 Deep Extrapolation (Stable Region: Steady State)

### X1: ka=11.0, kd2=6000, kd3=0.15 (Steady State Flat Line, ka=11) — [Steady State Flat Line]
- **Kinetic Parameters**: $k_a = 11.0$, $k_{d2} = 6000.0$, $k_{d3} = 0.15$, $k_p = 1.0$
- **Ground Truth Attractor**: Fixed Point ($m_1^* = 523.2$ mM, $T = 0.0000$ s, Amplitude $= 0.00$ mM, Pure Flat Straight Line)

| Model | Hallucinated Amp (mM) [GT=0.0] | Mass Drift (mM) |
| :--- | :---: | :---: |
| **FiLM Bifurcation-Aware** | 12.66 ± 8.95 mM [7.76, 2.05–26.86] | 0.0026 ± 0.0000 |
| **FiLM Pure** | 910.63 ± 803.19 mM [429.85, 306.53–2368.13] | 0.0035 ± 0.0002 |
| **FiLM FT (alpha=0)** | 915.53 ± 814.69 mM [442.56, 290.53–2399.20] | 0.0036 ± 0.0002 |
| **FiLM+Div** | 742.60 ± 559.86 mM [418.72, 364.03–1967.53] | 0.0036 ± 0.0002 |
| Projected Baseline | 466.85 ± 236.60 mM [357.00, 241.89–990.72] | 0.0039 ± 0.0002 |
| Baseline MLP (Unconstrained) | 566.85 ± 273.96 mM [458.04, 234.88–1027.68] | 1066.0 ± 792.3 |

---

### X2: ka=12.0, kd2=6000, kd3=0.15 (Steady State Flat Line, ka=12) — [Steady State Flat Line]
- **Kinetic Parameters**: $k_a = 12.0$, $k_{d2} = 6000.0$, $k_{d3} = 0.15$, $k_p = 1.0$
- **Ground Truth Attractor**: Fixed Point ($m_1^* = 558.1$ mM, $T = 0.0000$ s, Amplitude $= 0.00$ mM, Pure Flat Straight Line)

| Model | Hallucinated Amp (mM) [GT=0.0] | Mass Drift (mM) |
| :--- | :---: | :---: |
| **FiLM Bifurcation-Aware** | 454.68 ± 990.52 mM [60.01, 16.33–2880.14] | 0.0028 ± 0.0004 |
| **FiLM Pure** | 912.92 ± 753.43 mM [469.60, 309.05–2276.41] | 0.0035 ± 0.0002 |
| **FiLM FT (alpha=0)** | 926.36 ± 762.55 mM [511.66, 304.40–2300.72] | 0.0034 ± 0.0003 |
| **FiLM+Div** | 860.74 ± 525.66 mM [585.78, 382.71–1929.58] | 0.0036 ± 0.0002 |
| Projected Baseline | 466.35 ± 207.97 mM [364.11, 262.21–903.30] | 0.0039 ± 0.0003 |
| Baseline MLP (Unconstrained) | 555.28 ± 237.82 mM [473.86, 254.56–918.10] | 1094.5 ± 834.7 |

---

### X3: ka=13.0, kd2=6000, kd3=0.25 (Steady State Flat Line, ka=13) — [Steady State Flat Line]
- **Kinetic Parameters**: $k_a = 13.0$, $k_{d2} = 6000.0$, $k_{d3} = 0.25$, $k_p = 1.0$
- **Ground Truth Attractor**: Fixed Point ($m_1^* = 432.6$ mM, $T = 0.0000$ s, Amplitude $= 0.00$ mM, Pure Flat Straight Line)

| Model | Hallucinated Amp (mM) [GT=0.0] | Mass Drift (mM) |
| :--- | :---: | :---: |
| **FiLM Bifurcation-Aware** | 407.95 ± 873.16 mM [54.32, 37.28–2546.52] | 0.0025 ± 0.0004 |
| **FiLM Pure** | 326.79 ± 336.75 mM [206.56, 86.50–1139.18] | 0.0029 ± 0.0003 |
| **FiLM FT (alpha=0)** | 327.07 ± 357.40 mM [201.30, 78.39–1192.85] | 0.0029 ± 0.0004 |
| **FiLM+Div** | 322.43 ± 291.60 mM [220.52, 141.65–1029.47] | 0.0029 ± 0.0002 |
| Projected Baseline | 428.45 ± 226.60 mM [399.15, 221.13–938.57] | 0.0039 ± 0.0002 |
| Baseline MLP (Unconstrained) | 510.10 ± 215.03 mM [495.13, 249.88–885.95] | 996.7 ± 729.7 |

---

### X4: ka=10.5, kd2=4600, kd3=0.60 (Steady State Flat Line, ka=10.5) — [Steady State Flat Line]
- **Kinetic Parameters**: $k_a = 10.5$, $k_{d2} = 4600.0$, $k_{d3} = 0.6$, $k_p = 1.0$
- **Ground Truth Attractor**: Fixed Point ($m_1^* = 323.9$ mM, $T = 0.0000$ s, Amplitude $= 0.00$ mM, Pure Flat Straight Line)

| Model | Hallucinated Amp (mM) [GT=0.0] | Mass Drift (mM) |
| :--- | :---: | :---: |
| **FiLM Bifurcation-Aware** | 7.06 ± 1.83 mM [6.12, 4.77–9.96] | 0.0027 ± 0.0002 |
| **FiLM Pure** | 206.27 ± 142.61 mM [117.62, 83.85–471.51] | 0.0032 ± 0.0002 |
| **FiLM FT (alpha=0)** | 233.02 ± 152.43 mM [116.38, 85.21–492.25] | 0.0032 ± 0.0002 |
| **FiLM+Div** | 134.10 ± 80.44 mM [115.16, 69.70–315.37] | 0.0029 ± 0.0002 |
| Projected Baseline | 152.23 ± 86.77 mM [141.76, 48.73–284.22] | 0.0035 ± 0.0003 |
| Baseline MLP (Unconstrained) | 179.73 ± 92.20 mM [128.54, 67.57–320.74] | 874.0 ± 662.3 |

---

### X5: ka=12.0, kd2=4500, kd3=0.50 (Steady State Flat Line, ka=12) — [Steady State Flat Line]
- **Kinetic Parameters**: $k_a = 12.0$, $k_{d2} = 4500.0$, $k_{d3} = 0.5$, $k_p = 1.0$
- **Ground Truth Attractor**: Fixed Point ($m_1^* = 405.3$ mM, $T = 0.0000$ s, Amplitude $= 0.00$ mM, Pure Flat Straight Line)

| Model | Hallucinated Amp (mM) [GT=0.0] | Mass Drift (mM) |
| :--- | :---: | :---: |
| **FiLM Bifurcation-Aware** | 5.32 ± 1.70 mM [4.97, 3.21–7.95] | 0.0026 ± 0.0000 |
| **FiLM Pure** | 364.04 ± 194.77 mM [268.83, 157.69–681.25] | 0.0032 ± 0.0003 |
| **FiLM FT (alpha=0)** | 423.26 ± 211.74 mM [441.36, 161.52–727.46] | 0.0034 ± 0.0001 |
| **FiLM+Div** | 272.64 ± 207.38 mM [188.95, 95.61–738.11] | 0.0030 ± 0.0003 |
| Projected Baseline | 194.93 ± 86.62 mM [184.60, 96.72–311.06] | 0.0036 ± 0.0003 |
| Baseline MLP (Unconstrained) | 225.14 ± 80.53 mM [217.82, 100.75–345.19] | 1007.0 ± 823.5 |

---

### X6: ka=14.0, kd2=5000, kd3=0.30 (Steady State Flat Line, ka=14) — [Steady State Flat Line]
- **Kinetic Parameters**: $k_a = 14.0$, $k_{d2} = 5000.0$, $k_{d3} = 0.3$, $k_p = 1.0$
- **Ground Truth Attractor**: Fixed Point ($m_1^* = 542.7$ mM, $T = 0.0000$ s, Amplitude $= 0.00$ mM, Pure Flat Straight Line)

| Model | Hallucinated Amp (mM) [GT=0.0] | Mass Drift (mM) |
| :--- | :---: | :---: |
| **FiLM Bifurcation-Aware** | 5.56 ± 2.37 mM [4.79, 2.52–9.66] | 0.0021 ± 0.0000 |
| **FiLM Pure** | 400.15 ± 170.88 mM [414.35, 236.94–758.51] | 0.0029 ± 0.0002 |
| **FiLM FT (alpha=0)** | 415.23 ± 180.28 mM [395.56, 245.94–811.59] | 0.0029 ± 0.0002 |
| **FiLM+Div** | 750.49 ± 484.56 mM [566.36, 183.90–1723.92] | 0.0028 ± 0.0003 |
| Projected Baseline | 309.33 ± 108.89 mM [291.04, 190.26–545.21] | 0.0036 ± 0.0002 |
| Baseline MLP (Unconstrained) | 332.95 ± 88.09 mM [344.90, 180.20–456.18] | 1210.4 ± 973.3 |

---


## Diverse ka, kd2, kd3 Regimes (Limit Cycles)

### X7: ka=1.0, kd2=3000, kd3=0.80 (Ultra-Low ka Slow Relaxation) — [Sustained Limit Cycle]
- **Kinetic Parameters**: $k_a = 1.0$, $k_{d2} = 3000.0$, $k_{d3} = 0.8$, $k_p = 1.0$
- **Ground Truth**: Period $T = 0.7047$ s, Amplitude $= 721.2 mM$

| Model | Period Error (%) [Median, Min–Max] | Amp Ratio ($A_{\text{model}}/A_{\text{true}}$) | Mass Drift (mM) |
| :--- | :---: | :---: | :---: |
| **FiLM Bifurcation-Aware** | 30.33 ± 7.27% [30.42%, 16.29–40.80%] | 1.430 ± 0.077 [1.425, 1.283–1.536] | 0.0029 ± 0.0000 |
| **FiLM Pure** | 29.01 ± 12.52% [30.06%, 1.33–41.67%] | 1.418 ± 0.054 [1.427, 1.339–1.503] | 0.0029 ± 0.0001 |
| **FiLM FT (alpha=0)** | 28.15 ± 11.94% [28.78%, 2.04–40.78%] | 1.409 ± 0.050 [1.425, 1.328–1.485] | 0.0029 ± 0.0001 |
| **FiLM+Div** | 18.26 ± 8.49% [20.05%, 2.49–29.31%] | 1.056 ± 0.180 [1.067, 0.847–1.428] | 0.0030 ± 0.0001 |
| Projected Baseline | 42.67 ± 12.68% [47.24%, 22.92–57.64%] | 0.860 ± 0.515 [0.638, 0.286–1.458] | 0.0037 ± 0.0001 |
| Baseline MLP (Unconstrained) | 104.64 ± 11.37% [100.00%, 100.00–132.49%] | 1.055 ± 1.552 [0.539, 0.121–4.835] | 995.4 ± 619.2 |

---

### X8: ka=2.5, kd2=3200, kd3=0.80 (Low ka Relaxation Cycle) — [Sustained Limit Cycle]
- **Kinetic Parameters**: $k_a = 2.5$, $k_{d2} = 3200.0$, $k_{d3} = 0.8$, $k_p = 1.0$
- **Ground Truth**: Period $T = 0.3602$ s, Amplitude $= 886.3 mM$

| Model | Period Error (%) [Median, Min–Max] | Amp Ratio ($A_{\text{model}}/A_{\text{true}}$) | Mass Drift (mM) |
| :--- | :---: | :---: | :---: |
| **FiLM Bifurcation-Aware** | 3.48 ± 1.47% [3.29%, 1.62–6.26%] | 1.081 ± 0.043 [1.086, 0.996–1.149] | 0.0029 ± 0.0001 |
| **FiLM Pure** | 5.17 ± 4.45% [3.19%, 0.51–13.90%] | 1.024 ± 0.037 [1.029, 0.970–1.088] | 0.0029 ± 0.0001 |
| **FiLM FT (alpha=0)** | 4.97 ± 4.17% [3.92%, 0.35–13.74%] | 1.023 ± 0.033 [1.032, 0.972–1.074] | 0.0029 ± 0.0001 |
| **FiLM+Div** | 32.06 ± 13.82% [30.72%, 13.59–53.67%] | 0.661 ± 0.314 [0.787, 0.085–1.044] | 0.0031 ± 0.0001 |
| Projected Baseline | 13.88 ± 7.98% [14.57%, 3.07–28.23%] | 0.993 ± 0.215 [1.070, 0.560–1.193] | 0.0036 ± 0.0001 |
| Baseline MLP (Unconstrained) | 50.13 ± 51.59% [33.02%, 17.36–174.87%] | 1.222 ± 0.433 [1.144, 0.650–2.189] | 380.4 ± 412.5 |

---

### X9: ka=3.5, kd2=2600, kd3=1.10 (Low kd2, High kd3) — [Sustained Limit Cycle]
- **Kinetic Parameters**: $k_a = 3.5$, $k_{d2} = 2600.0$, $k_{d3} = 1.1$, $k_p = 1.0$
- **Ground Truth**: Period $T = 0.1806$ s, Amplitude $= 297.4 mM$

| Model | Period Error (%) [Median, Min–Max] | Amp Ratio ($A_{\text{model}}/A_{\text{true}}$) | Mass Drift (mM) |
| :--- | :---: | :---: | :---: |
| **FiLM Bifurcation-Aware** | 10.10 ± 3.42% [11.66%, 4.01–13.63%] | 1.544 ± 0.114 [1.584, 1.394–1.704] | 0.0032 ± 0.0000 |
| **FiLM Pure** | 13.88 ± 4.30% [12.70%, 7.01–21.58%] | 1.671 ± 0.123 [1.627, 1.485–1.846] | 0.0032 ± 0.0000 |
| **FiLM FT (alpha=0)** | 13.99 ± 4.02% [12.44%, 8.96–21.64%] | 1.677 ± 0.111 [1.624, 1.546–1.867] | 0.0032 ± 0.0000 |
| **FiLM+Div** | 33.62 ± 29.34% [10.63%, 3.89–72.17%] | 1.738 ± 0.651 [1.859, 0.571–2.431] | 0.0032 ± 0.0000 |
| Projected Baseline | 29.80 ± 7.67% [33.36%, 19.68–39.05%] | 1.447 ± 0.797 [1.664, 0.195–2.280] | 0.0037 ± 0.0001 |
| Baseline MLP (Unconstrained) | 23.67 ± 11.55% [23.58%, 1.45–38.68%] | 2.882 ± 2.901 [2.352, 0.165–9.402] | 807.5 ± 555.0 |

---

### X10: ka=4.5, kd2=3500, kd3=1.30 (Elevated kd3 Fast Cycle) — [Sustained Limit Cycle]
- **Kinetic Parameters**: $k_a = 4.5$, $k_{d2} = 3500.0$, $k_{d3} = 1.3$, $k_p = 1.0$
- **Ground Truth**: Period $T = 0.1636$ s, Amplitude $= 137.5 mM$

| Model | Period Error (%) [Median, Min–Max] | Amp Ratio ($A_{\text{model}}/A_{\text{true}}$) | Mass Drift (mM) |
| :--- | :---: | :---: | :---: |
| **FiLM Bifurcation-Aware** | 28.07 ± 8.77% [26.94%, 18.00–41.52%] | 2.641 ± 0.825 [2.467, 1.554–3.942] | 0.0030 ± 0.0001 |
| **FiLM Pure** | 42.99 ± 16.91% [36.53%, 25.85–81.52%] | 2.258 ± 0.574 [2.112, 1.618–3.241] | 0.0030 ± 0.0000 |
| **FiLM FT (alpha=0)** | 44.61 ± 23.36% [34.97%, 25.91–99.91%] | 2.299 ± 0.603 [2.232, 1.521–3.363] | 0.0030 ± 0.0000 |
| **FiLM+Div** | 42.36 ± 31.96% [32.96%, 5.77–97.83%] | 2.707 ± 0.858 [2.473, 1.556–3.974] | 0.0030 ± 0.0001 |
| Projected Baseline | 12.63 ± 9.26% [12.10%, 0.84–29.87%] | 2.862 ± 1.592 [3.317, 0.220–4.509] | 0.0036 ± 0.0001 |
| Baseline MLP (Unconstrained) | 11.05 ± 8.63% [8.67%, 2.39–28.66%] | 4.378 ± 4.125 [3.385, 0.191–12.434] | 502.1 ± 369.0 |

---

### X11: ka=5.0, kd2=5200, kd3=0.35 (High kd2, Low kd3 Massive Cycle) — [Sustained Limit Cycle]
- **Kinetic Parameters**: $k_a = 5.0$, $k_{d2} = 5200.0$, $k_{d3} = 0.35$, $k_p = 1.0$
- **Ground Truth**: Period $T = 0.3673$ s, Amplitude $= 812.7 mM$

| Model | Period Error (%) [Median, Min–Max] | Amp Ratio ($A_{\text{model}}/A_{\text{true}}$) | Mass Drift (mM) |
| :--- | :---: | :---: | :---: |
| **FiLM Bifurcation-Aware** | 11.29 ± 4.02% [11.76%, 4.94–18.23%] | 1.092 ± 0.081 [1.081, 1.008–1.261] | 0.0031 ± 0.0000 |
| **FiLM Pure** | 10.46 ± 5.17% [9.82%, 0.89–18.16%] | 1.148 ± 0.068 [1.125, 1.034–1.247] | 0.0031 ± 0.0001 |
| **FiLM FT (alpha=0)** | 10.81 ± 5.07% [10.21%, 1.08–18.49%] | 1.146 ± 0.059 [1.112, 1.073–1.248] | 0.0031 ± 0.0001 |
| **FiLM+Div** | 33.71 ± 21.00% [46.11%, 1.21–56.86%] | 0.537 ± 0.489 [0.133, 0.105–1.201] | 0.0031 ± 0.0001 |
| Projected Baseline | 29.35 ± 8.52% [26.38%, 16.55–41.86%] | 0.702 ± 0.223 [0.803, 0.291–0.899] | 0.0035 ± 0.0001 |
| Baseline MLP (Unconstrained) | 21.68 ± 12.61% [16.54%, 8.83–41.18%] | 0.943 ± 0.456 [0.976, 0.199–1.500] | 202.0 ± 145.4 |

---

### X12: ka=8.5, kd2=4000, kd3=0.85 (Near-Critical Boundary) — [Sustained Limit Cycle]
- **Kinetic Parameters**: $k_a = 8.5$, $k_{d2} = 4000.0$, $k_{d3} = 0.85$, $k_p = 1.0$
- **Ground Truth**: Period $T = 0.1441$ s, Amplitude $= 16.2 mM$

| Model | Period Error (%) [Median, Min–Max] | Amp Ratio ($A_{\text{model}}/A_{\text{true}}$) | Mass Drift (mM) |
| :--- | :---: | :---: | :---: |
| **FiLM Bifurcation-Aware** | 7.71 ± 2.97% [8.16%, 3.34–13.26%] | 7.097 ± 6.251 [5.382, 0.459–20.910] | 0.0028 ± 0.0002 |
| **FiLM Pure** | 24.84 ± 14.97% [20.34%, 11.84–58.92%] | 13.420 ± 11.986 [8.295, 0.756–32.641] | 0.0028 ± 0.0003 |
| **FiLM FT (alpha=0)** | 24.19 ± 14.14% [18.90%, 11.80–55.84%] | 14.139 ± 12.003 [10.917, 0.884–30.465] | 0.0029 ± 0.0003 |
| **FiLM+Div** | 5.57 ± 5.67% [3.04%, 0.40–17.40%] | 6.271 ± 9.388 [1.322, 0.483–28.397] | 0.0026 ± 0.0002 |
| Projected Baseline | 24.60 ± 4.58% [26.36%, 16.26–30.73%] | 3.955 ± 1.940 [4.152, 0.395–6.432] | 0.0034 ± 0.0002 |
| Baseline MLP (Unconstrained) | 38.83 ± 33.28% [24.65%, 23.12–120.15%] | 3.274 ± 1.416 [3.262, 0.637–4.928] | 607.8 ± 500.0 |

---

