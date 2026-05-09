# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import json
import time

def hill_estimator(data, k_min=5, k_max=None):
    y = np.abs(data)
    y = y[y > 0]
    y = np.sort(y)[::-1]
    if k_max is None:
        k_max = len(y) // 2
    if len(y) < k_min:
        return np.array([]), np.array([])
    k_values = np.arange(k_min, k_max + 1)
    alpha_values = np.zeros(len(k_values))
    log_y = np.log(y)
    cumsum_log_y = np.cumsum(log_y)
    for idx, k in enumerate(k_values):
        sum_log = cumsum_log_y[k-1]
        estimator = (sum_log - k * log_y[k]) / k
        if estimator > 1e-10:
            alpha_values[idx] = 1.0 / estimator
        else:
            alpha_values[idx] = np.nan
    return k_values, alpha_values

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = 'data/'
    tgrid = np.load(os.path.join(data_dir, 'preprocessed_kraichnan_tgrid.npy'))
    kraichnan_files = ['preprocessed_kraichnan_xi0p50_x.npy', 'preprocessed_kraichnan_xi0p75_x.npy', 'preprocessed_kraichnan_xi1p00_x.npy', 'preprocessed_kraichnan_xi1p50_x.npy', 'preprocessed_kraichnan_xi1p80_x.npy']
    results = {}
    plt.figure(figsize=(10, 6))
    for fname in kraichnan_files:
        x = np.load(os.path.join(data_dir, fname))
        t = tgrid[:x.shape[1]]
        msd = np.mean(x**2, axis=0)
        t_min_fit = 10.0
        t_max_fit = min(40.0, t[-1] * 0.8)
        valid_idx = (t >= t_min_fit) & (t <= t_max_fit)
        t_fit = t[valid_idx]
        msd_fit = msd[valid_idx]
        slope, intercept = np.polyfit(np.log(t_fit), np.log(msd_fit), 1)
        H_est = slope / 2.0
        xi_str = fname.split('_')[2]
        xi_val = float(xi_str.replace('p', '.'))
        H_theory = xi_val / 2.0
        results[xi_str] = {'H_est': H_est, 'H_theory': H_theory, 'xi': xi_val}
        plt.plot(t, msd, label='xi=' + str(xi_val) + ' (H=' + str(round(H_est, 2)) + ')')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time t')
    plt.ylabel('Mean Squared Displacement')
    plt.title('MSD vs Time for Kraichnan Tracers')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename_msd = 'step_4_kraichnan_msd_' + str(timestamp) + '.png'
    plt.savefig(os.path.join(data_dir, plot_filename_msd), dpi=300)
    plt.close()
    fig_hill, axes_hill = plt.subplots(3, 2, figsize=(14, 15))
    axes_hill = axes_hill.flatten()
    fig_pdf, axes_pdf = plt.subplots(3, 2, figsize=(14, 15))
    axes_pdf = axes_pdf.flatten()
    for i, fname in enumerate(kraichnan_files):
        x = np.load(os.path.join(data_dir, fname))
        t = tgrid[:x.shape[1]]
        xi_str = fname.split('_')[2]
        xi_val = float(xi_str.replace('p', '.'))
        t_idx_int = np.searchsorted(t, 10.0)
        t_idx_large = np.searchsorted(t, 40.0)
        data_int = x[:, t_idx_int]
        data_large = x[:, t_idx_large]
        k_vals_int, alpha_int = hill_estimator(data_int)
        k_vals_large, alpha_large = hill_estimator(data_large)
        stable_mask = (k_vals_large >= 10) & (k_vals_large <= 30)
        if np.any(stable_mask) and not np.all(np.isnan(alpha_large[stable_mask])):
            alpha_est = np.nanmean(alpha_large[stable_mask])
        else:
            alpha_est = np.nan
        alpha_theory = 2.0 / xi_val
        results[xi_str]['alpha_est'] = alpha_est
        results[xi_str]['alpha_theory'] = alpha_theory
        ax_h = axes_hill[i]
        ax_h.plot(k_vals_int, alpha_int, label='t=' + str(round(t[t_idx_int], 1)))
        ax_h.plot(k_vals_large, alpha_large, label='t=' + str(round(t[t_idx_large], 1)))
        ax_h.axhline(alpha_theory, color='k', linestyle='--')
        ax_h.set_title('Hill Plot: xi=' + str(xi_val))
        ax_p = axes_pdf[i]
        counts, bins = np.histogram(data_large, bins=30, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax_p.plot(bin_centers, counts, 'o-')
        ax_p.set_yscale('log')
        ax_p.set_title('PDF at t=' + str(round(t[t_idx_large], 1)) + ': xi=' + str(xi_val))
    fig_hill.tight_layout()
    fig_hill.savefig(os.path.join(data_dir, 'step_4_hill_stability_' + str(timestamp) + '.png'), dpi=300)
    plt.close(fig_hill)
    fig_pdf.tight_layout()
    fig_pdf.savefig(os.path.join(data_dir, 'step_4_kraichnan_pdf_' + str(timestamp) + '.png'), dpi=300)
    plt.close(fig_pdf)
    with open(os.path.join(data_dir, 'kraichnan_msd_alpha_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()