import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def compute_Tc_from_magnetization(json_data, smooth=True):
    key = list(json_data.keys())[0]
    data = sorted(json_data[key], key=lambda x: x["T"])

    T = np.array([d["T"] for d in data])
    m = np.array([d["m"] for d in data])

    if smooth and len(m) >= 5:
        m_used = savgol_filter(m, window_length=5, polyorder=2)
    else:
        m_used = m

    dm_dT = np.gradient(m_used, T)

    idx = np.argmin(dm_dT)
    Tc = T[idx]

    return Tc, T, m, m_used, dm_dT


def plot_results(T, m, m_smooth, dm_dT, Tc):
    plt.figure(figsize=(10, 5))

    # Magnetization plot
    plt.subplot(1, 2, 1)
    plt.plot(T, m, marker='o', label="raw m")
    plt.plot(T, m_smooth, linestyle='--', label="smoothed m")
    plt.axvline(Tc, linestyle='--', label=f"Tc ≈ {Tc:.3f}")
    plt.xlabel("T")
    plt.ylabel("m")
    plt.title("Magnetization vs Temperature")
    plt.legend()

    # Derivative plot
    plt.subplot(1, 2, 2)
    plt.plot(T, dm_dT, marker='o')
    plt.axvline(Tc, linestyle='--', label=f"Tc ≈ {Tc:.3f}")
    plt.xlabel("T")
    plt.ylabel("dm/dT")
    plt.title("Derivative of Magnetization")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with open("data.json", "r") as f:
        json_data = json.load(f)

    Tc, T, m, m_smooth, dm_dT = compute_Tc_from_magnetization(json_data)

    print(f"Estimated Tc = {Tc}")

    plot_results(T, m, m_smooth, dm_dT, Tc)