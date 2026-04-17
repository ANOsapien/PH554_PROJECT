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
parser.add_argument(
    "--cone_assign",
    type=str,
    default="single",
    choices=["single", "random", "rows"],
    help=(
        "[auxiliary only] How to assign the two cone types to spins.\n"
        "  single : all spins share theta_deg (original behaviour)\n"
        "  random : each spin independently assigned theta_deg or theta_deg2 with p=0.5\n"
        "  rows   : even rows → theta_deg, odd rows → theta_deg2"
    )
)
parser.add_argument(
    "--theta_deg2",
    type=float,
    default=None,
    help="[auxiliary only] Full cone angle (degrees) for type-2 spins. "
         "Required when --cone_assign is 'random' or 'rows'."
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="[auxiliary + random cone_assign] RNG seed for reproducible mask generation."
)

args = parser.parse_args()
MODE        = args.mode
CONE_ASSIGN = args.cone_assign
THETA_DEG2  = args.theta_deg2
SEED        = args.seed

# Validate two-cone arguments
if MODE == "auxiliary" and CONE_ASSIGN != "single" and THETA_DEG2 is None:
    parser.error("--theta_deg2 is required when --cone_assign is 'random' or 'rows'.")

if MODE == "selfish" and CONE_ASSIGN != "single":
    print("[WARNING] --cone_assign has no effect in selfish mode; using single cone.")
    CONE_ASSIGN = "single"

# ----------------------------
# Config
# ----------------------------

L      = 50
T_list = np.linspace(0.5, 0.65, 6)

# Primary cone sweep list
cone_list = [306]

N_RUNS = 1

# Label for output directory reflects the cone assignment scheme
if CONE_ASSIGN != "single":
    OUTPUT_DIR = f"{MODE}_results_{CONE_ASSIGN}_cone"
else:
    OUTPUT_DIR = f"{MODE}_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Dynamic import
# ----------------------------

sim_module   = importlib.import_module(MODE)
run_sim_func = sim_module.run_simulation

# ----------------------------
# Wrapper: unified call for both modes
# ----------------------------

def run_simulation(L, T, theta_deg, init="ordered"):
    if MODE == "auxiliary" and CONE_ASSIGN != "single":
        return run_sim_func(
            L=L, T=T, theta_deg=theta_deg, init=init,
            theta_deg2=THETA_DEG2,
            cone_assign=CONE_ASSIGN,
            seed=SEED
        )
    else:
        return run_sim_func(L=L, T=T, theta_deg=theta_deg, init=init)

# ----------------------------
# Run sweep
# ----------------------------

results = {}

for theta in cone_list:
    theta_key = float(theta)
    results[theta_key] = []

    for T in T_list:
        print(f"[{MODE}|{CONE_ASSIGN}] Running: theta={theta:.1f}°, "
              f"theta2={THETA_DEG2}°, T={T:.4f}")

        m_vals   = []
        epr_vals = []

        for run in range(N_RUNS):
            out = run_simulation(L=L, T=T, theta_deg=theta, init="ordered")
            m_vals.append(out["m"])
            epr_vals.append(out["EPR"])

        results[theta_key].append({
            "T":   float(T),
            "m":   float(np.mean(m_vals)),
            "EPR": float(np.mean(epr_vals))
        })

# ----------------------------
# Save raw results
# ----------------------------

meta = {
    "mode":        MODE,
    "cone_assign": CONE_ASSIGN,
    "theta_deg2":  THETA_DEG2,
    "seed":        SEED,
    "L":           L,
    "N_RUNS":      N_RUNS
}

with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

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
plt.ylabel("Vision Cone θ₁ (degrees)")

if CONE_ASSIGN != "single":
    plt.title(f"{MODE.upper()} | {CONE_ASSIGN} cones "
              f"(θ₁={cone_list[0]}°, θ₂={THETA_DEG2}°)")
else:
    plt.title(f"{MODE.upper()} NRXY Phase Diagram")

plt.savefig(os.path.join(OUTPUT_DIR, f"phase_diagram_{MODE}.png"))
plt.close()

# ----------------------------
# Scatter  (mx, my samples)
# ----------------------------

theta_pick = cone_list[0]
T_pick     = 0.82

print(f"[{MODE}|{CONE_ASSIGN}] Scatter sample: theta={theta_pick}°, T={T_pick}")

out = run_simulation(L=L, T=T_pick, theta_deg=theta_pick, init="disordered")

mx = out.get("mx", np.array([out["mx"]]))
my = out.get("my", np.array([out["my"]]))

plt.figure()
plt.scatter(mx, my, s=5)
plt.xlabel("m_x")
plt.ylabel("m_y")
plt.title(f"{MODE} Scatter | {CONE_ASSIGN} | θ₁={theta_pick}°"
          + (f", θ₂={THETA_DEG2}°" if THETA_DEG2 else ""))
plt.savefig(os.path.join(OUTPUT_DIR, f"scatter_{MODE}.png"))
plt.close()

# ----------------------------
# Extra: visualise cone-type mask (auxiliary two-cone only)
# ----------------------------

if MODE == "auxiliary" and CONE_ASSIGN != "single" and "mask" in out:
    mask = out["mask"].astype(int)   # 1 = type-1, 0 = type-2

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(mask, cmap="bwr", vmin=0, vmax=1, origin="lower")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels([f"Type-2 ({THETA_DEG2}°)", f"Type-1 ({theta_pick}°)"])
    ax.set_title(f"Cone-type mask ({CONE_ASSIGN})")
    ax.set_xlabel("j")
    ax.set_ylabel("i (row)")
    fig.savefig(os.path.join(OUTPUT_DIR, "cone_mask.png"), dpi=120)
    plt.close(fig)
    print("Cone mask saved.")

print("Done. Results saved in:", OUTPUT_DIR)