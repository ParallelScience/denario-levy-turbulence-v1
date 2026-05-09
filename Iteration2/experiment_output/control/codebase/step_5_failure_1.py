# filename: codebase/step_5.py
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
def estimate_alpha_cf(phi, k_vals):
    pos_mask = k_vals > 0
    k_pos = k_vals[pos_mask]
    phi_pos = phi[pos_mask]
    valid_mask = (phi_pos > 0.05) & (phi_pos < 0.95)
    if np.sum(valid_mask) < 3:
        valid_mask = (phi_pos > 0.01) & (phi_pos < 0.99)
    if np.sum(valid_mask) < 3:
        valid_mask = (phi_pos > 0.001) & (phi_pos < 0.999)
    if np.sum(valid_mask) < 2:
        return np.nan
    k_valid = k_pos[valid_mask]
    phi_valid = phi_pos[valid_mask]
    y = np.log(-np.log(phi_valid))
    x = np.log(k_valid)
    slope, _ = np.polyfit(x, y, 1)
    return slope
def compute_phi(x_tau, k_vals):
    phi = np.zeros_like(k_vals)
    for i, k in enumerate(k_vals):
        phi[i] = np.mean(np.cos(k * x_tau))
    return phi
def main():
    data_dir = "data/"
    rg_phi_1p5 = np.load("/home/node/work/projects/levy_turbulence_v1/data/rg_char_func_kraichnan_xi1p5.npy")
    rg_tau_idx_1p5 = np.load("/home/node/work/projects/levy_turbulence_v1/data/rg_tau_values.npy")
    rg_k_range = np.load("/home/node/work/projects/levy_turbulence_v1/data/rg_k_range.npy")
    tau_1p5 = rg_tau_idx_1p5 * 0.01
    alpha_1p5 = []
    for i in range(len(rg_tau_idx_1p5)):
        a = estimate_alpha_cf(rg_phi_1p5[i], rg_k_range)
        alpha_1p5.append(a)
    alpha_1p5 = np.array(alpha_1p5)
    x_1p8 = np.load(os.path.join(data_dir, "preprocessed_kraichnan_xi1p80_x.npy"))
    dt_1p8 = 0.025
    tau_idx_1p8 = [1, 2, 8, 20, 40, 80, 200, 400, 800, 2000]
    tau_1p8 = np.array(tau_idx_1p8) * dt_1p8
    alpha_1p8 = []
    for idx in tau_idx_1p8:
        max_start = x_1p8.shape[1] - idx
        step = max(1, max_start // 50)
        displacements = []
        for start in range(0, max_start, step):
            displacements.append(x_1p8[:, start+idx] - x_1p8[:, start])
        displacements = np.concatenate(displacements)
        phi = compute_phi(displacements, rg_k_range)
        a = estimate_alpha_cf(phi, rg_k_range)
        alpha_1p8.append(a)
    alpha_1p8 = np.array(alpha_1p8)
    def get_crossover(tau, alpha, threshold=1.8):
        for t, a in zip(tau, alpha):
            if not np.isnan(a) and a < threshold:
                return t
        return np.nan
    tau_star_1p5 = get_crossover(tau_1p5, alpha_1p5, threshold=1.8)
    tau_star_1p8 = get_crossover(tau_1p8, alpha_1p8, threshold=1.8)
    plt.figure(figsize=(10, 6))
    plt.plot(tau_1p5, alpha_1p5, 'o-', label='xi=1.5 (Data)', color='blue', linewidth=2)
    plt.plot(tau_1p8, alpha_1p8, 's-', label='xi=1.8 (Data)', color='red', linewidth=2)
    plt.axhline(2.0, color='black', linestyle='--', label='Ballistic/Gaussian (alpha=2)')
    plt.axhline(2.0/1.5, color='blue', linestyle=':', label='Theory xi=1.5 (alpha=1.33)')
    plt.axhline(2.0/1.8, color='red', linestyle=':', label='Theory xi=1.8 (alpha=1.11)')
    plt.xscale('log')
    plt.xlabel('Time lag tau [a.u.]', fontsize=14)
    plt.ylabel('Effective Lévy index alpha(tau)', fontsize=14)
    plt.title('RG Flow of Effective Diffusion Operator', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(data_dir, "rg_flow_alpha_5_" + str(timestamp) + ".png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print("Plot saved to " + plot_filename)
    print("--- RG Flow Verification ---")
    print("Time lag tau | alpha(tau) for xi=1.5 | alpha(tau) for xi=1.8")
    print("-" * 65)
    for i in range(len(tau_1p5)):
        t_str = str(round(tau_1p5[i], 3)).ljust(12)
        a1_str = str(round(alpha_1p5[i], 4)).ljust(21)
        if i < len(alpha_1p8):
            a2_str = str(round(alpha_1p8[i], 4))
        else:
            a2_str = "N/A"
        print(t_str + " | " + a1_str + " | " + a2_str)
    print("\n--- Crossover Times (tau*) ---")
    print("xi=1.5 : tau* = " + str(tau_star_1p5))
    print("xi=1.8 : tau* = " + str(tau_star_1p8))
    def replace_nan(lst):
        return [None if np.isnan(x) else x for x in lst]
    results = {"xi_1p5": {"tau": tau_1p5.tolist(), "alpha": replace_nan(alpha_1p5.tolist()), "tau_star": None if np.isnan(tau_star_1p5) else float(tau_star_1p5)}, "xi_1p8": {"tau": tau_1p8.tolist(), "alpha": replace_nan(alpha_1p8.tolist()), "tau_star": None if np.isnan(tau_star_1p8) else float(tau_star_1p8)}}
    json_filename = os.path.join(data_dir, "rg_flow_results.json")
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print("Extracted alpha(tau) and tau* values saved to " + json_filename)
if __name__ == '__main__':
    main()