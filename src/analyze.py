#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import csv


def load_data(json_path):
    """
    Expected JSON format:
    {
      "0.0": [
        {"T": 0.3, "m": ..., "EPR": ...},
        ...
      ],
      "60.0": [...]
    }
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    data = {}
    for theta_str, entries in raw.items():
        theta = float(theta_str)
        entries_sorted = sorted(entries, key=lambda x: x["T"])
        T = np.array([float(e["T"]) for e in entries_sorted], dtype=float)
        m = np.array([float(e["m"]) for e in entries_sorted], dtype=float)
        epr = np.array([float(e["EPR"]) for e in entries_sorted], dtype=float)
        data[theta] = {"T": T, "m": m, "EPR": epr}
    return data


def moving_average(y, window=3):
    if window <= 1 or len(y) < window:
        return y.copy()
    kernel = np.ones(window) / window
    # pad at ends so output stays same length
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    ypad = np.pad(y, (pad_left, pad_right), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


def estimate_tc_from_m(T, m):
    """
    Estimate Tc as the temperature where magnetization drops fastest.
    Uses the most negative slope of a lightly smoothed curve.
    """
    if len(T) < 3:
        return float(T[np.argmin(m)])

    ms = moving_average(m, window=min(3, len(m)))
    dm_dT = np.gradient(ms, T)
    idx = int(np.argmin(dm_dT))
    return float(T[idx])


def estimate_tc_from_epr(T, epr):
    """
    Estimate Tc as the temperature where EPR rises fastest.
    If EPR is almost flat, fall back to max value location.
    """
    if len(T) < 3:
        return float(T[np.argmax(epr)])

    es = moving_average(epr, window=min(3, len(epr)))
    de_dT = np.gradient(es, T)

    # If there is a clear rising edge, use it; otherwise use the max EPR point.
    if np.max(np.abs(de_dT)) > 1e-12:
        idx = int(np.argmax(de_dT))
    else:
        idx = int(np.argmax(es))
    return float(T[idx])


def estimate_tc(T, m, epr):
    tc_m = estimate_tc_from_m(T, m)
    return tc_m, tc_m, tc_m


def plot_observable_vs_T(data, observable, title, ylabel, outpath):
    """
    Plot one observable for all angles in one figure.
    """
    plt.figure(figsize=(9, 6))
    for theta in sorted(data.keys()):
        T = data[theta]["T"]
        y = data[theta][observable]
        plt.plot(T, y, marker="o", linewidth=1.8, label=f"{theta:.0f}°")

    plt.title(title)
    plt.xlabel("Temperature T")
    plt.ylabel(ylabel)
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()




def plot_theta_tc(summary_rows, dataset_name, outpath):
    thetas = [r["theta_deg"] for r in summary_rows]
    tcs = [r["tc_estimated"] for r in summary_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(thetas, tcs, marker="o", linewidth=2)
    plt.title(f"Estimated Tc vs Angle — {dataset_name}")
    plt.xlabel("Vision cone angle θ (degrees)")
    plt.ylabel("Estimated Tc")
    plt.xticks(thetas)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_summary_csv(summary_rows, outpath):
    fieldnames = ["theta_deg", "tc_from_m", "tc_from_epr", "tc_estimated"]
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row[k] for k in fieldnames})


def analyze_dataset(json_path, output_dir, dataset_name):
    data = load_data(json_path)
    os.makedirs(output_dir, exist_ok=True)

    # Plots
    plot_observable_vs_T(
        data,
        observable="m",
        title=f"Magnetization m vs T — {dataset_name}",
        ylabel="Magnetization m",
        outpath=os.path.join(output_dir, f"{dataset_name}_m_vs_T.png"),
    )

    plot_observable_vs_T(
        data,
        observable="EPR",
        title=f"Entropy Production Rate EPR vs T — {dataset_name}",
        ylabel="EPR",
        outpath=os.path.join(output_dir, f"{dataset_name}_epr_vs_T.png"),
    )

    summary_rows = []
    for theta in sorted(data.keys()):
        T = data[theta]["T"]
        m = data[theta]["m"]
        epr = data[theta]["EPR"]

        tc_m, tc_epr, tc_est = estimate_tc(T, m, epr)

        summary_rows.append(
            {
                "theta_deg": theta,
                "tc_from_m": tc_m,
                "tc_from_epr": tc_epr,
                "tc_estimated": tc_est,
            }
        )

    # Tc vs theta
    plot_theta_tc(
        summary_rows,
        dataset_name=dataset_name,
        outpath=os.path.join(output_dir, f"{dataset_name}_tc_vs_theta.png"),
    )

    # Save CSV summary
    save_summary_csv(
        summary_rows,
        outpath=os.path.join(output_dir, f"{dataset_name}_tc_summary.csv"),
    )

    return summary_rows


def plot_combined_m(aux_data, selfish_data, aux_summary, selfish_summary, outpath):
    plt.figure(figsize=(9, 6))

    # assume same theta grid
    for theta in sorted(aux_data.keys()):
        T = aux_data[theta]["T"]

        plt.plot(T, aux_data[theta]["m"], linestyle="-", label=f"Aux {theta:.0f}°")
        plt.plot(T, selfish_data[theta]["m"], linestyle="--", label=f"Self {theta:.0f}°")

        # Tc lines
        aux_tc = next(r["tc_estimated"] for r in aux_summary if r["theta_deg"] == theta)
        self_tc = next(r["tc_estimated"] for r in selfish_summary if r["theta_deg"] == theta)

        plt.axvline(aux_tc, linestyle=":", linewidth=1)
        plt.axvline(self_tc, linestyle="--", linewidth=1)

    plt.title("Magnetization Comparison (Aux vs Selfish)")
    plt.xlabel("Temperature T")
    plt.ylabel("Magnetization m")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Analyze auxiliary/selfish simulation JSON data."
    )
    parser.add_argument(
        "--aux-json",
        default="auxiliary_results/results_auxiliary.json",
        help="Path to auxiliary JSON file",
    )
    parser.add_argument(
        "--selfish-json",
        default="selfish_results/results_selfish.json",
        help="Path to selfish JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Directory to save plots and CSVs",
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    aux_dir = outdir / "auxiliary"
    selfish_dir = outdir / "selfish"
    aux_dir.mkdir(exist_ok=True)
    selfish_dir.mkdir(exist_ok=True)

    aux_summary = analyze_dataset(args.aux_json, str(aux_dir), "auxiliary")
    selfish_summary = analyze_dataset(args.selfish_json, str(selfish_dir), "selfish")

    # Print concise terminal summary
    print("\nAuxiliary dataset Tc estimates:")
    for row in aux_summary:
        print(
            f"theta={row['theta_deg']:.0f}°  "
            f"Tc(m)={row['tc_from_m']:.3f}  "
            f"Tc(EPR)={row['tc_from_epr']:.3f}  "
            f"Tc≈{row['tc_estimated']:.3f}"
        )

    print("\nSelfish dataset Tc estimates:")
    for row in selfish_summary:
        print(
            f"theta={row['theta_deg']:.0f}°  "
            f"Tc(m)={row['tc_from_m']:.3f}  "
            f"Tc(EPR)={row['tc_from_epr']:.3f}  "
            f"Tc≈{row['tc_estimated']:.3f}"
        )

    aux_data = load_data(args.aux_json)
    selfish_data = load_data(args.selfish_json)

    plot_combined_m(
        aux_data,
        selfish_data,
        aux_summary,
        selfish_summary,
        outpath=str(outdir / "combined_m_comparison.png"),
    )

    

if __name__ == "__main__":
    main()