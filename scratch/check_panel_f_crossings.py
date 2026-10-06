import torch
import numpy as np
import sys
import os

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
from validate_biochemical_climate_and_ergodic import true_jacobian, compute_neural_jacobian
from train_and_evaluate_lorenz_style_sciml import UnifiedLorenzStyleSciML, find_physical_steady_state, DT

model = UnifiedLorenzStyleSciML(hidden_dim=128, dt=DT)
model.load_state_dict(torch.load("unified_sciml_seed_42.pt", map_location="cpu"))
model.eval()

ka_sweep = np.linspace(6.0, 14.0, 33)
re_lambda_gt = []
re_lambda_sciml = []

for ka_v in ka_sweep:
    p_v = (ka_v, 4600.0, 0.60, 1.0)
    y_star = find_physical_steady_state(p_v)

    J_true = true_jacobian(y_star, p_v)
    eigs_true = np.linalg.eigvals(J_true)
    max_re_true = float(np.max(np.real(eigs_true)))
    re_lambda_gt.append(max_re_true)

    J_nn = compute_neural_jacobian(model, y_star, p_v)
    eigs_nn = np.linalg.eigvals(J_nn)
    max_re_nn = float(np.max(np.real(eigs_nn)))
    re_lambda_sciml.append(max_re_nn)

re_lambda_gt = np.array(re_lambda_gt)
re_lambda_sciml = np.array(re_lambda_sciml)

crossings_gt = np.where(np.diff(np.sign(re_lambda_gt)))[0]
crossings_sciml = np.where(np.diff(np.sign(re_lambda_sciml)))[0]

print("Ground Truth zero crossings at ka indices:", crossings_gt, "ka values:", [ka_sweep[i] for i in crossings_gt])
print("SciML zero crossings at ka indices:", crossings_sciml, "ka values:", [ka_sweep[i] for i in crossings_sciml])
print("SciML max Re(lambda) values across sweep:")
for ka, r_gt, r_nn in zip(ka_sweep, re_lambda_gt, re_lambda_sciml):
    print(f"ka={ka:5.2f}: True Re={r_gt:+8.4f} | SciML Re={r_nn:+8.4f} | sign={'+' if r_nn > 0 else '-'}")
