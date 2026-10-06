# Sensitivity Sweep of Pacemaker Floor & Single-Component Ablations

### A. Pacemaker $m_0$ Floor Sweep ($[0.5, 0.2, 0.1, 0.05, 0.01, 10^{-4}]$)

| Clamp Floor | Period $T$ (s) | Period Err (%) | Amplitude | Amp Err (%) | Retention (%) | Minimum $m_0$ in Trajectory | Mass Drift ($\mu\text{M}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  0.5000 | 0.1659 | 0.06% | 223.7 | 6.58% | 94.2% | 1.7618 | 0.0030 |
|  0.2000 | 0.1659 | 0.06% | 223.7 | 6.58% | 94.2% | 1.7618 | 0.0030 |
|  0.1000 | 0.1659 | 0.06% | 223.7 | 6.58% | 94.2% | 1.7618 | 0.0030 |
|  0.0500 | 0.1659 | 0.06% | 223.7 | 6.58% | 94.2% | 1.7618 | 0.0030 |
|  0.0100 | 0.1659 | 0.06% | 223.7 | 6.58% | 94.2% | 1.7618 | 0.0030 |
|  0.0001 | 0.1659 | 0.06% | 223.7 | 6.58% | 94.2% | 1.7618 | 0.0030 |

### B. Single-Component Architectural Ablation Study (Total Pool $C_{\text{tot}} \equiv 10,002.0\,\mu\text{M}$)

| Architecture Variant | Period Err (%) | Amp Err (%) | Retention (%) | Mass Drift ($\mu\text{M}$) | Rel. Mass Drift (%) | Rollout MSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Proposed SciML** | 0.06% | 6.58% | 94.2% | 0.0030 | 0.00003% | 1.99e+03 |
| w/o Exact Mass Conservation | 0.06% | 6.58% | 94.2% | 0.0265 | 0.00026% | 1.99e+03 |
| w/o Positivity Clamp (m0>=0.0) | 0.06% | 6.58% | 94.2% | 0.0030 | 0.00003% | 1.99e+03 |
| Unconstrained Baseline MLP | 11.47% | 62.63% | 51.2% | 94.6 | 0.946% | 1.29e+06 |

