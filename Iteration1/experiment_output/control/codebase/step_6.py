# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import time

def verify_effective_operator():
    data_dir = "/home/node/work/projects/levy_turbulence_v1/data"
    out_dir = "data/"
    xi_values = [0.5, 0.75, 1.0, 1.5, 1.8]
    xi_strs = ['0p50', '0p75', '1p00', '1p50', '1p80']
    t_grid_full = np.load(os.path.join(data_dir, "kraichnan_tgrid.npy"))
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    print("--- Effective Operator Identification ---")
    for i in range(len(xi_values)):
        xi = xi_values[i]
        xi_str = xi_strs[i]
        x_path = os.path.join(data_dir, "kraichnan_xi" + xi_str + "_x.npy")
        x_k = np.load(x_path)
        t_k = np.linspace(t_grid_full[0], t_grid_full[-1], x_k.shape[1])
        t_targets = [50, 75, 100]
        t_indices = [np.argmin(np.abs(t_k - t)) for t in t_targets]
        ax = axes[i]
        alpha_th = 2.0 / xi
        t_max_idx = t_indices[-1]
        x_t_max = x_k[:, t_max_idx] - x_k[:, 0]
        std_x = np.std(x_t_max)
        if std_x < 1e-6:
            std_x = 1e-6
        k_range = np.logspace(np.log10(0.05 / std_x), np.log10(5.0 / std_x), 200)
        for t_idx in t_indices:
            t = t_k[t_idx]
            x_t = x_k[:, t_idx] - x_k[:, 0]
            phi = np.mean(np.cos(np.outer(k_range, x_t)), axis=1)
            valid = (phi > 0.1) & (phi < 0.9)
            if np.sum(valid) > 3:
                k_valid = k_range[valid]
                y = -np.log(phi[valid]) / t
                ax.loglog(k_valid, y, 'o', markersize=4, alpha=0.6, label="t=" + str(int(t)))
                if t_idx == t_max_idx:
                    log_y = np.log(y)
                    log_k = np.log(k_valid)
                    log_D = np.mean(log_y - alpha_th * log_k)
                    D = np.exp(log_D)
                    preds_log = log_D + alpha_th * log_k
                    ss_res_log = np.sum((log_y - preds_log)**2)
                    ss_tot_log = np.sum((log_y - np.mean(log_y))**2)
                    r2_log = 1.0 - ss_res_log / ss_tot_log if ss_tot_log > 1e-10 else 0.0
                    preds_lin = D * (k_valid ** alpha_th)
                    ss_res_lin = np.sum((y - preds_lin)**2)
                    ss_tot_lin = np.sum((y - np.mean(y))**2)
                    r2_lin = 1.0 - ss_res_lin / ss_tot_lin if ss_tot_lin > 1e-10 else 0.0
                    ax.loglog(k_valid, preds_lin, 'k--', linewidth=2, label="Fit: D|k|^" + str(round(alpha_th, 2)))
                    print("xi = " + str(xi) + " | alpha_th = " + str(round(alpha_th, 3)) + " | D_alpha = " + str(round(D, 4)) + " | R^2 (log) = " + str(round(r2_log, 4)) + " | R^2 (lin) = " + str(round(r2_lin, 4)))
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('-log(phi(k,t)) / t')
        ax.set_title('Kraichnan xi=' + str(xi))
        ax.legend()
        ax.grid(True, which='both', ls='--', alpha=0.5)
    axes[-1].axis('off')
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = "effective_operator_6_" + str(timestamp) + ".png"
    plot_filepath = os.path.join(out_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    print("\nPlot saved to " + plot_filepath)

if __name__ == '__main__':
    verify_effective_operator()