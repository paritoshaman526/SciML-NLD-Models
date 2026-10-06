import os, sys
import numpy as np

PROJ = r"c:\Users\satya\Downloads\Van Der Pol"
sys.path.insert(0, PROJ)
from biochemical_global_sciml import system, rk4_step, DEFAULT_GUESS

print("=" * 70)
print("TESTING RK4 EXTINCTION ON SLICE A (kd2=4600, kd3=0.60, kp=1.0)")
print("=" * 70)

kd2, kd3, kp = 4600.0, 0.60, 1.0
dt = 0.01

for ka in [8.5, 9.0, 9.2, 9.3, 9.4, 9.5, 9.8, 9.975, 10.0, 10.5, 12.0]:
    p = np.array([ka, kd2, kd3, kp], dtype=np.float64)
    y = DEFAULT_GUESS.copy()
    # 20 seconds warmup (2000 steps)
    for _ in range(2000):
        y = rk4_step(y, dt, p)
    sig = []
    for _ in range(1000): # 10 seconds measurement
        y = rk4_step(y, dt, p)
        sig.append(y[2])
    amp = (max(sig) - min(sig)) / 2.0
    status = "LC" if amp > 8.0 else "SS"
    print(f"ka = {ka:6.3f} | Amp after 20s warmup: {amp:10.4f} mM | Class: {status}")
