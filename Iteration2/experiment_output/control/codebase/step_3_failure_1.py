# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['text.usetex'] = False

def compute_autocorr(v):
    N_traj, N_time = v.shape
    n_fft = 2 ** int(np.ceil(np.log2(2 * N_time - 1)))
    v_fft = np.fft.fft(v, n=n_fft, axis=1)
    S = v_fft * np.conj(v_fft)
    R = np.fft.ifft(S, axis=1).real
    R_mean = np.mean(R, axis=0)
    R_mean = R_mean[:N_time]
    counts = np.arange(N_time, 0, -1)
    R_mean = R_mean / counts
    return R_mean

def compute_tau_c(R, dt):
    R_norm = R / R[0]
    zero_crossings = np.where(R_norm < 0)[0]
    if len(zero_crossings) > 0:
        idx = zero_crossings[0]
        if idx == 1:
            frac = R_norm[0] / (R_norm[0] - R_norm[1] + 1e-15)
            tau_c = 0.5 * R_norm[0] * (frac * dt)
        else:
            frac = R_norm[idx-1] / (R_norm[idx-1] - R_norm[idx] + 1e-15)
            tau_c = np.trapz(R_norm[:idx], dx=dt) + 0.5 * R_norm[idx-1] * (frac * dt)
    else:
        tau_c = np.trapz(R_norm, dx=dt)
    return tau_c

if __name__ == '__main__':
    data_dir = "data/"
    xi_values = ["0p50", "0p75", "1p00", "1p50", "1p80"]
    results = {}
    tgrid_path = "/home/node/work/projects/levy_turbulence_v1/data/kraichnan_tgrid.npy"
    tgrid_raw = np.load(tgrid_path)
    for xi in xi_values:
        x = np.load(os.path.join(data_dir, "preprocessed_kraichnan_xi" + xi + "_x.npy"))
        if len(tgrid_raw) == x.shape[1]:
            t = tgrid_raw
        else:
            t = np.linspace(tgrid_raw[0], tgrid_raw[-1], x.shape[1])
        dt = t[1] - t[0]
        v = np.diff(x, axis=1) / dt
        R = compute_autocorr(v)
        tau_c = compute_tau_c(R, dt)
        results["Kraichnan_xi" + xi] = {"tau_c": float(tau_c), "T": float(t[-1] - t[0]), "R": R, "dt": float(dt)}
    v_l96 = np.load(os.path.join(data_dir, "preprocessed_lorenz96_velocities.npy")).T
    dt_l96 = 0.05
    R_l96 = compute_autocorr(v_l96)
    tau_c_l96 = compute_tau_c(R_l96, dt_l96)
    results["Lorenz96"] = {"tau_c": float(tau_c_l96), "T": float(v_l96.shape[1] * dt_l96), "R": R_l96, "dt": float(dt_l96)}
    print("--- Velocity Correlation and Time-Scale Diagnostic ---")
    print("Dataset                  | tau_c      | T          | tau_c / T")
    print("-" * 65)
    for key, res in results.items():
        tau_c = res["tau_c"]
        T = res["T"]
        ratio = tau_c / T
        print(key.ljust(24) + " | " + str(round(tau_c, 4)).ljust(10) + " | " + str(round(T, 1)).ljust(10) + " | " + str(ratio)[:10].ljust(10))
    timestamp = int(time.time())
    plot_filename = os.path.join(data_dir, "velocity_autocorr_3_" + str(timestamp) + ".png")
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    for i, (key, res) in enumerate(results.items()):
        R = res["R"]
        dt = res["dt"]
        tau = np.arange(len(R)) * dt
        R_norm = R / R[0]
        mask = tau <= 5.0
        plt.plot(tau[mask], R_norm[mask], label=key + " (tau_c=" + str(round(res['tau_c'], 3)) + ")", color=colors[i % len(colors)], linewidth=2)
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel("Time lag tau", fontsize=14)
    plt.ylabel("Normalized Velocity Autocorrelation Rv(tau) / Rv(0)", fontsize=14)
    plt.title("Lagrangian Velocity Autocorrelation", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print("Plot saved to " + plot_filename)
    tau_c_dict = {key: res["tau_c"] for key, res in results.items()}
    json_filename = os.path.join(data_dir, "computed_tau_c_values.json")
    with open(json_filename, 'w') as f:
        json.dump(tau_c_dict, f, indent=4)
    print("Computed tau_c values saved to " + json_filename)