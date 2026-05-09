# filename: codebase/step_7.py
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

def compute_msd(x):
    return np.mean(x**2, axis=0)

def extract_H(t, msd, start_idx=100):
    log_t = np.log(t[start_idx:])
    log_msd = np.log(msd[start_idx:])
    slope, _ = np.polyfit(log_t, log_msd, 1)
    return slope / 2.0

def estimate_alpha_cf(x_t, k_vals):
    N = len(x_t)
    noise_level = 1.0 / np.sqrt(N)
    phi = np.zeros_like(k_vals)
    for i, k in enumerate(k_vals):
        phi[i] = np.mean(np.cos(k * x_t))
    min_phi = max(0.1, noise_level * 2)
    valid_indices = []
    for i in range(len(k_vals)):
        if phi[i] > min_phi and phi[i] < 0.98:
            valid_indices.append(i)
        elif phi[i] <= min_phi:
            break
    if len(valid_indices) < 3:
        valid_indices = []
        for i in range(len(k_vals)):
            if phi[i] > noise_level and phi[i] < 0.99:
                valid_indices.append(i)
            elif phi[i] <= noise_level:
                break
    if len(valid_indices) < 2:
        return np.nan
    k_valid = k_vals[valid_indices]
    phi_valid = phi[valid_indices]
    y = np.log(-np.log(phi_valid))
    x = np.log(k_valid)
    slope, _ = np.polyfit(x, y, 1)
    return slope

def get_alpha_stats(x, k_vals, num_steps=100, step_size=10):
    alphas = []
    start_idx = max(0, x.shape[1] - num_steps)
    for step in range(start_idx, x.shape[1], step_size):
        a = estimate_alpha_cf(x[:, step], k_vals)
        if not np.isnan(a):
            alphas.append(a)
    if len(alphas) > 0:
        return np.mean(alphas), np.std(alphas)
    else:
        return np.nan, np.nan

def replace_nan(val):
    return None if np.isnan(val) else float(val)

def main():
    data_dir = "data/"
    v_l96 = np.load(os.path.join(data_dir, "preprocessed_lorenz96_velocities.npy"))
    dt_l96 = 0.05
    x_l96 = np.cumsum(v_l96, axis=0) * dt_l96
    x_l96 = x_l96.T
    t_l96 = np.arange(1, x_l96.shape[1] + 1) * dt_l96
    msd_l96 = compute_msd(x_l96)
    H_l96 = extract_H(t_l96, msd_l96, start_idx=1000)
    k_vals = np.logspace(-5, 2, 1000)
    alpha_l96_emp, alpha_l96_err = get_alpha_stats(x_l96, k_vals, num_steps=1000, step_size=50)
    with open(os.path.join(data_dir, "extracted_xi_values.json"), "r") as f:
        xi_dict = json.load(f)
    xi_l96 = xi_dict["Lorenz96"]
    alpha_l96_theo = 2.0 / xi_l96
    synthesis_data = []
    xi_values = ["0p50", "0p75", "1p00", "1p50", "1p80"]
    xi_floats = [0.50, 0.75, 1.00, 1.50, 1.80]
    for xi_str, xi_val in zip(xi_values, xi_floats):
        x_k = np.load(os.path.join(data_dir, "preprocessed_kraichnan_xi" + xi_str + "_x.npy"))
        a_emp, a_err = get_alpha_stats(x_k, k_vals, num_steps=200, step_size=20)
        a_theo = 2.0 / xi_val
        synthesis_data.append({"model": "Kraichnan xi=" + str(xi_val), "alpha_theo": a_theo, "alpha_emp": a_emp, "alpha_err": a_err, "color": "blue", "marker": "o"})
    x_pure = np.load(os.path.join(data_dir, "kolmogorov_kolmogorov_pure_disp.npy"))
    x_multi = np.load(os.path.join(data_dir, "kolmogorov_kolmogorov_multifractal_disp.npy"))
    a_pure_emp, a_pure_err = get_alpha_stats(x_pure, k_vals, num_steps=200, step_size=20)
    a_multi_emp, a_multi_err = get_alpha_stats(x_multi, k_vals, num_steps=200, step_size=20)
    xi_K41 = 1.0 + (2.0 / 3.0)
    alpha_K41 = 2.0 / xi_K41
    p = 2.0
    zeta_2_SL = p / 9.0 + 2.0 * (1.0 - (2.0 / 3.0)**(p / 3.0))
    xi_SL = 1.0 + zeta_2_SL
    alpha_SL = 2.0 / xi_SL
    synthesis_data.append({"model": "Kolmogorov pure", "alpha_theo": alpha_K41, "alpha_emp": a_pure_emp, "alpha_err": a_pure_err, "color": "green", "marker": "s"})
    synthesis_data.append({"model": "Kolmogorov multifractal", "alpha_theo": alpha_SL, "alpha_emp": a_multi_emp, "alpha_err": a_multi_err, "color": "red", "marker": "^"})
    synthesis_data.append({"model": "Lorenz-96", "alpha_theo": alpha_l96_theo, "alpha_emp": alpha_l96_emp, "alpha_err": alpha_l96_err, "color": "purple", "marker": "D"})
    plt.figure(figsize=(10, 8))
    for d in synthesis_data:
        plt.errorbar(d["alpha_theo"], d["alpha_emp"], yerr=d["alpha_err"], fmt=d["marker"], color=d["color"], label=d["model"], capsize=5, capthick=1.5, markersize=8, alpha=0.8)
    plt.plot([0.5, 4.5], [0.5, 4.5], 'k--', label='Theory (alpha = 2/xi)', alpha=0.7)
    plt.axhline(2.0, color='gray', linestyle=':', label='Gaussian Limit (alpha=2)', alpha=0.7)
    plt.xlabel('Theoretical Lévy Index alpha = 2/xi', fontsize=14)
    plt.ylabel('Empirical Lévy Index alpha', fontsize=14)
    plt.title('Universality of Fractional Operator: Empirical vs Theoretical alpha', fontsize=16)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0.5, 4.5)
    plt.ylim(0.5, 4.5)
    plt.tight_layout()
    plot_filename = os.path.join(data_dir, "synthesis_alpha_7_" + str(int(time.time())) + ".png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    for d in synthesis_data:
        d["alpha_emp"] = replace_nan(d["alpha_emp"])
        d["alpha_err"] = replace_nan(d["alpha_err"])
    results = {"Lorenz96": {"xi": float(xi_l96), "H_empirical": float(H_l96), "H_theoretical": float(xi_l96/2), "alpha_empirical": replace_nan(alpha_l96_emp), "alpha_theoretical": float(alpha_l96_theo), "alpha_error": replace_nan(alpha_l96_err)}, "Synthesis_Data": synthesis_data}
    with open(os.path.join(data_dir, "synthesis_results.json"), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()