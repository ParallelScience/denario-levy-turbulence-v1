# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import time

def compute_structure_functions(delta_u, p_values):
    S_p = np.zeros((len(p_values), delta_u.shape[1]))
    for i, p in enumerate(p_values):
        S_p[i, :] = np.mean(np.abs(delta_u)**p, axis=0)
    return S_p

def fit_and_residuals(S_p, r, idx_min, idx_max):
    log_r = np.log(r[idx_min:idx_max])
    zeta_p = np.zeros(S_p.shape[0])
    residuals = np.zeros((S_p.shape[0], idx_max - idx_min))
    r_squared = np.zeros(S_p.shape[0])
    for i in range(S_p.shape[0]):
        valid_mask = S_p[i, idx_min:idx_max] > 0
        log_S_p = np.log(S_p[i, idx_min:idx_max])
        slope, intercept = np.polyfit(log_r, log_S_p, 1)
        zeta_p[i] = slope
        preds = slope * log_r + intercept
        residuals[i, :] = log_S_p - preds
        ss_res = np.sum(residuals[i, :]**2)
        ss_tot = np.sum((log_S_p - np.mean(log_S_p))**2)
        if ss_tot > 1e-10:
            r_squared[i] = 1 - (ss_res / ss_tot)
        else:
            r_squared[i] = 1.0
    return zeta_p, residuals, r_squared

if __name__ == '__main__':
    data_dir = "/home/node/work/projects/levy_turbulence_v1/data"
    out_dir = "data/"
    k41_path = os.path.join(data_dir, "cascade_K41_delta_u.npy")
    mild_path = os.path.join(data_dir, "cascade_logN_mild_delta_u.npy")
    realistic_path = os.path.join(data_dir, "cascade_logN_realistic_delta_u.npy")
    r_scales_path = os.path.join(data_dir, "cascade_r_scales.npy")
    lorenz_sp_path = os.path.join(data_dir, "lorenz96_structure_functions.npy")
    delta_u_k41_raw = np.load(k41_path)
    delta_u_mild_raw = np.load(mild_path)
    delta_u_realistic_raw = np.load(realistic_path)
    r_scales = np.load(r_scales_path)
    lorenz_sp = np.load(lorenz_sp_path)
    k41_scaling = r_scales ** (1.0 / 3.0)
    delta_u_k41 = delta_u_k41_raw * k41_scaling
    delta_u_mild = delta_u_mild_raw * k41_scaling
    delta_u_realistic = delta_u_realistic_raw * k41_scaling
    p_values = np.arange(1, 7)
    S_p_k41 = compute_structure_functions(delta_u_k41, p_values)
    S_p_mild = compute_structure_functions(delta_u_mild, p_values)
    S_p_realistic = compute_structure_functions(delta_u_realistic, p_values)
    idx_min_casc, idx_max_casc = 4, 12
    zeta_k41, res_k41, r2_k41 = fit_and_residuals(S_p_k41, r_scales, idx_min_casc, idx_max_casc)
    zeta_mild, res_mild, r2_mild = fit_and_residuals(S_p_mild, r_scales, idx_min_casc, idx_max_casc)
    zeta_real, res_real, r2_real = fit_and_residuals(S_p_realistic, r_scales, idx_min_casc, idx_max_casc)
    lorenz_r = np.arange(1, 21)
    idx_min_lor, idx_max_lor = 0, 3
    zeta_lor, res_lor, r2_lor = fit_and_residuals(lorenz_sp, lorenz_r, idx_min_lor, idx_max_lor)
    xi_k41 = 1 + zeta_k41[1]
    xi_mild = 1 + zeta_mild[1]
    xi_real = 1 + zeta_real[1]
    xi_lor = 1 + zeta_lor[1]
    print("--- Spectral Roughness xi = 1 + zeta_2 ---")
    print("Cascade K41 (mu=0.0): xi = " + str(round(xi_k41, 4)))
    print("Cascade Mild (mu=0.15): xi = " + str(round(xi_mild, 4)))
    print("Cascade Realistic (mu=0.28): xi = " + str(round(xi_real, 4)))
    print("Lorenz-96: xi = " + str(round(xi_lor, 4)))
    print("\n--- Scaling Exponents zeta_p ---")
    for i, p in enumerate(p_values):
        print(str(p) + "\t" + str(round(zeta_k41[i], 4)) + "\t" + str(round(zeta_mild[i], 4)) + "\t" + str(round(zeta_real[i], 4)) + "\t" + str(round(zeta_lor[i], 4)))
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes[0, 0]
    ax.plot(p_values, zeta_k41, 'o-', label='Cascade K41')
    ax.plot(p_values, zeta_mild, 's-', label='Cascade Mild')
    ax.plot(p_values, zeta_real, '^-', label='Cascade Realistic')
    ax.plot(p_values, zeta_lor, 'd-', label='Lorenz-96')
    p_cont = np.linspace(1, 6, 100)
    zeta_th_k41 = p_cont / 3.0
    zeta_th_mild = p_cont / 3.0 - 0.15 * p_cont * (p_cont - 3) / 18.0
    zeta_th_real = p_cont / 3.0 - 0.28 * p_cont * (p_cont - 3) / 18.0
    ax.plot(p_cont, zeta_th_k41, 'k--', label='Theory K41')
    ax.plot(p_cont, zeta_th_mild, 'k-.', label='Theory Mild')
    ax.plot(p_cont, zeta_th_real, 'k:', label='Theory Realistic')
    ax.set_xlabel('Moment p')
    ax.set_ylabel('Scaling exponent zeta_p')
    ax.set_title('Structure Function Scaling Exponents')
    ax.legend()
    ax.grid(True)
    ax = axes[0, 1]
    for i, p in enumerate(p_values):
        ax.plot(np.log(r_scales[idx_min_casc:idx_max_casc]), res_k41[i], 'o-', label='p='+str(p))
    ax.set_xlabel('log(r)')
    ax.set_ylabel('Residuals')
    ax.set_title('Fit Residuals: Cascade K41')
    ax.legend(fontsize='small')
    ax.grid(True)
    ax = axes[0, 2]
    for i, p in enumerate(p_values):
        ax.plot(np.log(r_scales[idx_min_casc:idx_max_casc]), res_mild[i], 'o-', label='p='+str(p))
    ax.set_xlabel('log(r)')
    ax.set_ylabel('Residuals')
    ax.set_title('Fit Residuals: Cascade Mild')
    ax.legend(fontsize='small')
    ax.grid(True)
    ax = axes[1, 0]
    for i, p in enumerate(p_values):
        ax.plot(np.log(r_scales[idx_min_casc:idx_max_casc]), res_real[i], 'o-', label='p='+str(p))
    ax.set_xlabel('log(r)')
    ax.set_ylabel('Residuals')
    ax.set_title('Fit Residuals: Cascade Realistic')
    ax.legend(fontsize='small')
    ax.grid(True)
    ax = axes[1, 1]
    for i, p in enumerate(p_values):
        ax.plot(np.log(lorenz_r[idx_min_lor:idx_max_lor]), res_lor[i], 'o-', label='p='+str(p))
    ax.set_xlabel('log(lag)')
    ax.set_ylabel('Residuals')
    ax.set_title('Fit Residuals: Lorenz-96')
    ax.legend(fontsize='small')
    ax.grid(True)
    axes[1, 2].axis('off')
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = "eulerian_analysis_2_" + str(timestamp) + ".png"
    plot_filepath = os.path.join(out_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    print("\nPlot saved to " + plot_filepath)