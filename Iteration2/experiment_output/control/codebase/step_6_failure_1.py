# filename: codebase/step_6.py
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
    kolm_pure_path = os.path.join(data_dir, 'kolmogorov_kolmogorov_pure_disp.npy')
    kolm_multi_path = os.path.join(data_dir, 'kolmogorov_kolmogorov_multifractal_disp.npy')
    kolm_pure = np.load(kolm_pure_path)
    kolm_multi = np.load(kolm_multi_path)
    lorenz_path = os.path.join(data_dir, 'lorenz96_snapshots.npy')
    lorenz = np.load(lorenz_path)
    print('--- Kolmogorov Prediction and Intermittency ---')
    xi_pure = 5.0 / 3.0
    alpha_theory_pure = 2.0 / xi_pure
    zeta_2_sl = 0.696
    xi_multi = 1.0 + zeta_2_sl
    alpha_theory_multi = 2.0 / xi_multi
    print('Pure K41: xi = ' + str(round(xi_pure, 4)) + ', Theoretical alpha = ' + str(round(alpha_theory_pure, 4)))
    print('Multifractal (She-Leveque): zeta_2 = ' + str(zeta_2_sl) + ', xi = ' + str(round(xi_multi, 4)) + ', Theoretical alpha = ' + str(round(alpha_theory_multi, 4)))
    t_idx_kolm = kolm_pure.shape[1] - 1
    data_pure = kolm_pure[:, t_idx_kolm]
    data_multi = kolm_multi[:, t_idx_kolm]
    k_vals_pure, alpha_pure = hill_estimator(data_pure)
    k_vals_multi, alpha_multi = hill_estimator(data_multi)
    def get_stable_alpha(k_vals, alpha_vals, k_min, k_max):
        stable_mask = (k_vals >= k_min) & (k_vals <= k_max)
        if np.any(stable_mask) and not np.all(np.isnan(alpha_vals[stable_mask])):
            return np.nanmean(alpha_vals[stable_mask])
        return np.nan
    alpha_est_pure = get_stable_alpha(k_vals_pure, alpha_pure, 10, 30)
    alpha_est_multi = get_stable_alpha(k_vals_multi, alpha_multi, 10, 30)
    print('Empirical alpha for Pure K41 at final time step: ' + (str(round(alpha_est_pure, 4)) if not np.isnan(alpha_est_pure) else 'NaN'))
    print('Empirical alpha for Multifractal at final time step: ' + (str(round(alpha_est_multi, 4)) if not np.isnan(alpha_est_multi) else 'NaN'))
    print('\n--- Lorenz-96 Universality ---')
    print('Methodological Note: For Lorenz-96, we apply the tail-index estimator directly to the state variable increments (differences between snapshots at increasing time lags) as surrogate displacement statistics, rather than integrating velocities.')
    lags = [10, 50, 100, 500]
    lorenz_results = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if len(k_vals_pure) > 0:
        axes[0, 0].plot(k_vals_pure, alpha_pure, label='Pure K41', color='tab:blue')
    axes[0, 0].axhline(alpha_theory_pure, color='tab:blue', linestyle='--', label='Theory Pure')
    if len(k_vals_multi) > 0:
        axes[0, 0].plot(k_vals_multi, alpha_multi, label='Multifractal', color='tab:orange')
    axes[0, 0].axhline(alpha_theory_multi, color='tab:orange', linestyle='--', label='Theory Multi')
    axes[0, 0].set_ylim(0, 5)
    axes[0, 0].set_xlabel('Number of order statistics k', fontsize=12)
    axes[0, 0].set_ylabel('Estimated alpha', fontsize=12)
    axes[0, 0].set_title('Hill Plot: Kolmogorov Tracers', fontsize=14)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, linestyle=':', alpha=0.7)
    counts_p, bins_p = np.histogram(data_pure, bins=30, density=True)
    axes[0, 1].plot((bins_p[:-1]+bins_p[1:])/2, counts_p, 'o-', label='Pure K41')
    counts_m, bins_m = np.histogram(data_multi, bins=30, density=True)
    axes[0, 1].plot((bins_m[:-1]+bins_m[1:])/2, counts_m, 's-', label='Multifractal')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel('Displacement x', fontsize=12)
    axes[0, 1].set_ylabel('PDF P(x, t)', fontsize=12)
    axes[0, 1].set_title('PDF: Kolmogorov Tracers', fontsize=14)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, linestyle=':', alpha=0.7)
    for lag in lags:
        increments = lorenz[lag:, :] - lorenz[:-lag, :]
        data_lorenz = increments.flatten()
        k_vals_l, alpha_l = hill_estimator(data_lorenz, k_max=1000)
        alpha_est_l = get_stable_alpha(k_vals_l, alpha_l, 50, 200)
        print('Lorenz-96 lag=' + str(lag) + ': Estimated alpha = ' + (str(round(alpha_est_l, 4)) if not np.isnan(alpha_est_l) else 'NaN'))
        lorenz_results['lag_' + str(lag)] = alpha_est_l
        if len(k_vals_l) > 0:
            axes[1, 0].plot(k_vals_l, alpha_l, label='lag=' + str(lag))
        counts_l, bins_l = np.histogram(data_lorenz, bins=50, density=True)
        axes[1, 1].plot((bins_l[:-1]+bins_l[1:])/2, counts_l, label='lag=' + str(lag))
    axes[1, 0].set_ylim(0, 5)
    axes[1, 0].set_xlabel('Number of order statistics k', fontsize=12)
    axes[1, 0].set_ylabel('Estimated alpha', fontsize=12)
    axes[1, 0].set_title('Hill Plot: Lorenz-96 Increments', fontsize=14)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, linestyle=':', alpha=0.7)
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel('Increment dx', fontsize=12)
    axes[1, 1].set_ylabel('PDF P(dx)', fontsize=12)
    axes[1, 1].set_title('PDF: Lorenz-96 Increments', fontsize=14)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, linestyle=':', alpha=0.7)
    fig.tight_layout()
    timestamp = int(time.time())
    plot_filename = 'step_6_kolmogorov_lorenz_' + str(timestamp) + '.png'
    plot_filepath = os.path.join(data_dir, plot_filename)
    fig.savefig(plot_filepath, dpi=300)
    plt.close(fig)
    print('\nPlot saved to ' + plot_filepath)
    def sanitize_for_json(val):
        if val is None or np.isnan(val):
            return None
        return float(val)
    results = {'kolmogorov': {'pure_K41': {'xi_theory': xi_pure, 'alpha_theory': alpha_theory_pure, 'alpha_est': sanitize_for_json(alpha_est_pure)}, 'multifractal': {'xi_theory': xi_multi, 'alpha_theory': alpha_theory_multi, 'alpha_est': sanitize_for_json(alpha_est_multi)}}, 'lorenz96': {k: sanitize_for_json(v) for k, v in lorenz_results.items()}}
    json_filepath = os.path.join(data_dir, 'step_6_results.json')
    with open(json_filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print('Results saved to ' + json_filepath)

if __name__ == '__main__':
    main()