import sys
sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
from train_and_evaluate_lorenz_style_sciml import SWEEP_CASES, PHYSICAL_CTOT
print("PHYSICAL_CTOT =", PHYSICAL_CTOT)
print("12 training regimes:")
for c in SWEEP_CASES:
    p = c["params"]
    t = "SS" if c["is_steady_state"] else "LC"
    print("  {:5s}  ka={:.1f}  kd2={:.0f}  kd3={:.3f}  kp={:.2f}  {}".format(c["id"], p[0], p[1], p[2], p[3], t))
