# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from step_4 import hill_estimator
def main():
    plt.rcParams['text.usetex'] = False
    data_dir = 'data/'
    tgrid = np.load(os.path.join(data_dir, 'preprocessed_kraichnan_tgrid.npy'))
    dt = tgrid[1] - tgrid[0]
    t0 = tgrid[0]
    xi_files = {'1.5': 'preprocessed_kraichnan_xi1p50_x.npy', '1.8': 'preprocessed_kraichnan_xi1p80_x.npy'}
    results_hill = {'1.5': {'t': [], 'alpha': []}, '1.8': {'t': [], 'alpha': []}}
    print('--- RG Flow Verification: Tail-Index Estimation (Hill) ---')
    for xi_str, fname in xi_files.items():
        x = np.load(os.path.join(data_dir, fname))
        xi_val = float(xi_str)
        alpha_theory = 2.0 / xi_val
        print('Processing xi=' + str(xi_val) + ' (Theory alpha=' + str(round(alpha_theory, 4)) + ')')
        max_idx = x.shape[1] - 1
        t_indices = np.unique(np.geomspace(100, max_idx, num=15).astype(int))
        for t_idx in t_indices:
            data_t = x[:, t_idx]
            k_vals, alpha_vals = hill_estimator(data_t)
            stable_mask = (k_vals >= 10) & (k_vals <= 30)
            if np.any(stable_mask) and not np.all(np.isnan(alpha_vals[stable_mask])):
                alpha_est = np.nanmean(alpha_vals[stable_mask])
            else:
                alpha_est = np.nan
            results_hill[xi_str]['t'].append(t0 + t_idx * dt)
            results_hill[xi_str]['alpha'].append(alpha_est)
    print('\n--- RG Flow Verification: Characteristic Function ---')
    phi = np.load(os.path.join(data_dir, 'rg_char_func_kraichnan_xi1p5.npy'))
    tau_indices = np.load(os.path.join(data_dir, 'rg_tau_values.npy'))
    k_range = np.load(os.path.join(data_dir, 'rg_k_range.npy'))
    k_mask = (k_range > 0) & (k_range < 0.5)
    k_fit = k_range[k_mask]
    log_k_fit = np.log(k_fit)
    print('Fitting characteristic function in k-range: [' + str(round(np.min(k_fit), 4)) + ', ' + str(round(np.max(k_fit), 4)) + ']')
    plt.figure(figsize=(10, 6))
    alpha_char = []
    tau_char = tau_indices * dt
    for i, tau_idx in enumerate(tau_indices):
        phi_t = phi[i, :]
        phi_fit = np.real(phi_t[k_mask])
        valid_phi = (phi_fit > 0) & (phi_fit < 1)
        if np.sum(valid_phi) > 2:
            log_neg_log_phi = np.log(-np.log(phi_fit[valid_phi]))
            slope, intercept = np.polyfit(log_k_fit[valid_phi], log_neg_log_phi, 1)
            alpha_char.append(slope)
            plt.plot(k_fit[valid_phi], -np.log(phi_fit[valid_phi]), marker='o', linestyle='', markersize=4, label='tau=' + str(round(tau_idx*dt, 1)))
            plt.plot(k_fit[valid_phi], np.exp(intercept) * k_fit[valid_phi]**slope, color='k', alpha=0.3)
        else:
            alpha_char.append(np.nan)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('|k|', fontsize=14)
    plt.ylabel('-log(phi)', fontsize=14)
    plt.title('Characteristic Function Scaling: -log(phi) vs |k|', fontsize=16)
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename_char = 'step_5_char_func_scaling_5_' + str(timestamp) + '.png'
    plt.savefig(os.path.join(data_dir, plot_filename_char), dpi=300)
    plt.close()
    plt.figure(figsize=(10, 6))
    for xi_str in ['1.5', '1.8']:
        t_vals = results_hill[xi_str]['t']
        alpha_vals = results_hill[xi_str]['alpha']
        plt.plot(t_vals, alpha_vals, marker='s', linestyle='-', label='Hill Estimator (xi=' + xi_str + ')')
        xi_val = float(xi_str)
        plt.axhline(2.0 / xi_val, color=plt.gca().lines[-1].get_color(), linestyle='--', label='Theory 2/xi (xi=' + xi_str + ')')
    plt.plot(tau_char, alpha_char, marker='o', linestyle='-', color='k', label='Char Func (xi=1.5)')
    plt.axhline(2.0, color='gray', linestyle=':', label='Gaussian (alpha=2)')
    plt.xscale('log')
    plt.xlabel('Time lag tau', fontsize=14)
    plt.ylabel('Effective alpha(tau)', fontsize=14)
    plt.title('RG Flow of Effective Fractional Exponent alpha(tau)', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plot_filename_rg = 'step_5_rg_flow_alpha_5_' + str(timestamp) + '.png'
    plt.savefig(os.path.join(data_dir, plot_filename_rg), dpi=300)
    plt.close()
    rg_results = {'hill_1.5': {'tau': results_hill['1.5']['t'], 'alpha': results_hill['1.5']['alpha']}, 'hill_1.8': {'tau': results_hill['1.8']['t'], 'alpha': results_hill['1.8']['alpha']}, 'char_func_1.5': {'tau': tau_char.tolist(), 'alpha': alpha_char}}
    with open(os.path.join(data_dir, 'rg_flow_results.json'), 'w') as f:
        json.dump(rg_results, f, indent=4)
if __name__ == '__main__':
    main()