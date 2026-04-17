import numpy as np
import matplotlib.pyplot as plt
import os
import json
import importlib
import argparse

# ----------------------------
# CLI arguments
# ----------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["selfish", "auxiliary"],
    help="Choose simulation mode"
)

args = parser.parse_args()
MODE = args.mode

# ----------------------------
# Config
# ----------------------------

L = 100
T_low  = np.linspace(0.1, 0.67, 20, endpoint=False)
T_mid  = np.linspace(0.67, 0.9, 50, endpoint=False)  # dense region
T_high = np.linspace(0.9, 1.0, 20)

T_list = np.concatenate([T_low, T_mid, T_high])
cone_list = [306]
N_RUNS = 10

OUTPUT_DIR = f"{MODE}_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Dynamic import
# ----------------------------

sim_module = importlib.import_module(MODE)
run_simulation = sim_module.run_simulation

# ----------------------------
# Run sweep
# ----------------------------

results = {}

for theta in cone_list:
    theta_key = float(theta)
    results[theta_key] = []

    for T in T_list:
        print(f"[{MODE}] Running: theta={theta:.1f}, T={T:.2f}")

        m_vals = []
        epr_vals = []

        for run in range(N_RUNS):
            out = run_simulation(
                L=L,
                T=T,
                theta_deg=theta,
                init="ordered"
            )

            m_vals.append(out["m"])
            epr_vals.append(out["EPR"])

        m_avg = float(np.mean(m_vals))
        epr_avg = float(np.mean(epr_vals))

        results[theta_key].append({
            "T": float(T),
            "m": m_avg,
            "EPR": epr_avg
        })

# ----------------------------
# Save raw results
# ----------------------------

with open(os.path.join(OUTPUT_DIR, f"results_{MODE}.json"), "w") as f:
    json.dump(results, f, indent=2)

# ----------------------------
# Convert to arrays
# ----------------------------

M = np.zeros((len(cone_list), len(T_list)))

for i, theta in enumerate(cone_list):
    for j, entry in enumerate(results[float(theta)]):
        M[i, j] = entry["m"]

# ----------------------------
# Heatmap
# ----------------------------

plt.figure()
plt.imshow(
    M,
    aspect='auto',
    origin='lower',
    extent=[T_list[0], T_list[-1], cone_list[0], cone_list[-1]]
)

plt.colorbar(label="Magnetization |m|")
plt.xlabel("Temperature T")
plt.ylabel("Vision Cone θ (degrees)")
plt.title(f"{MODE.upper()} NRXY Phase Diagram")

plt.savefig(os.path.join(OUTPUT_DIR, f"phase_diagram_{MODE}.png"))
plt.close()

# ----------------------------
# Scatter
# ----------------------------

theta_pick = 180
T_pick = 0.8

print(f"[{MODE}] Running scatter sample at theta={theta_pick}, T={T_pick}")

out = run_simulation(
    L=L,
    T=T_pick,
    theta_deg=theta_pick,
    init="disordered"
)

mx = out["mx"]
my = out["my"]

plt.figure()
plt.scatter(mx, my, s=5)
plt.xlabel("m_x")
plt.ylabel("m_y")
plt.title(f"{MODE} Scatter (θ={theta_pick}, T={T_pick})")

plt.savefig(os.path.join(OUTPUT_DIR, f"scatter_{MODE}.png"))
plt.close()

print("Done. Results saved in:", OUTPUT_DIR)