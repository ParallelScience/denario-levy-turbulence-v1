# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import time

def analyze_kraichnan_dispersion(data_dir, out_dir, xi_values, xi_strs):
    t_grid_full = np.load(os.path.join(data_dir, 'kraichnan_tgrid.npy'))
    plt.rcParams['text.usetex'] = False
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    ax_msd = axes1[0]
    ax_H = axes1[1]
    ax_res = axes1[2]
    fig2, ax_alpha = plt.subplots(1, 1, figsize=(6, 6))
    print('--- Kraichnan Single-Particle Dispersion Analysis ---')
    empirical_alphas = []
    theoretical_alphas = []
    for i in range(len(xi_values)):
        xi = xi_values[i]
        xi_str = xi_strs[i]
        x_path = os.path.join(data_dir, 'kraichnan_xi' + xi_str + '_x.npy')
        x_k = np.load(x_path)
        t_k = np.linspace(t_grid_full[0], t_grid_full[-1], x_k.shape[1])
        displacements = x_k - x_k[:, 0:1]
        msd = np.mean(displacements**2, axis=0)
        valid_idx = t_k > 0
        t_valid = t_k[valid_idx]
        msd_valid = msd[valid_idx]
        log_t = np.log(t_valid)
        log_msd = np.log(msd_valid)
        H_local = np.gradient(log_msd, log_t) / 2.0
        fit_mask = t_valid > 50
        log_t_fit = log_t[fit_mask]
        log_msd_fit = log_msd[fit_mask]
        slope, intercept = np.polyfit(log_t_fit, log_msd_fit, 1)
        H_global = slope / 2.0
        residuals = log_msd_fit - (slope * log_t_fit + intercept)
        alpha_emp = 1.0 / H_global
        alpha_th = 2.0 / xi
        empirical_alphas.append(alpha_emp)
        theoretical_alphas.append(alpha_th)
        print('xi = ' + str(xi) + ' | Theoretical H = ' + str(round(xi/2.0, 3)) + ' | Empirical H = ' + str(round(H_global, 3)) + ' | Theoretical alpha = ' + str(round(alpha_th, 3)) + ' | Empirical alpha = ' + str(round(alpha_emp, 3)))
        ax_msd.loglog(t_valid, msd_valid, label='xi=' + str(xi))
        ax_H.semilogx(t_valid, H_local, label='xi=' + str(xi))
        ax_res.semilogx(t_valid[fit_mask], residuals, label='xi=' + str(xi))
    ax_msd.set_xlabel('Time t')
    ax_msd.set_ylabel('MSD <x^2(t)>')
    ax_msd.set_title('Mean Squared Displacement')
    ax_msd.legend()
    ax_msd.grid(True, which='both', ls='--', alpha=0.5)
    ax_H.set_xlabel('Time t')
    ax_H.set_ylabel('Local H(t)')
    ax_H.set_title('Local Scaling Exponent H(t)')
    ax_H.legend()
    ax_H.grid(True, which='both', ls='--', alpha=0.5)
    ax_res.set_xlabel('Time t')
    ax_res.set_ylabel('Fit Residuals (log-log space)')
    ax_res.set_title('Residuals of Asymptotic Log-Log Fit (t > 50)')
    ax_res.legend()
    ax_res.grid(True)
    fig1.tight_layout()
    ax_alpha.plot(theoretical_alphas, empirical_alphas, 'o', markersize=8, label='Data')
    min_alpha = min(min(theoretical_alphas), min(empirical_alphas)) * 0.9
    max_alpha = max(max(theoretical_alphas), max(empirical_alphas)) * 1.1
    ax_alpha.plot([min_alpha, max_alpha], [min_alpha, max_alpha], 'k--', label='1:1 Reference')
    ax_alpha.set_xlabel('Theoretical alpha (2/xi)')
    ax_alpha.set_ylabel('Empirical alpha (1/H)')
    ax_alpha.set_title('Empirical vs Theoretical Fractional Exponent')
    ax_alpha.legend()
    ax_alpha.grid(True)
    fig2.tight_layout()
    timestamp = int(time.time())
    plot1_filepath = os.path.join(out_dir, 'kraichnan_dispersion_3_' + str(timestamp) + '.png')
    fig1.savefig(plot1_filepath, dpi=300)
    print('\nDispersion plot saved to ' + plot1_filepath)
    plot2_filepath = os.path.join(out_dir, 'kraichnan_alpha_comparison_3_' + str(timestamp) + '.png')
    fig2.savefig(plot2_filepath, dpi=300)
    print('Alpha comparison plot saved to ' + plot2_filepath)

if __name__ == '__main__':
    data_dir = '/home/node/work/projects/levy_turbulence_v1/data'
    out_dir = 'data/'
    xi_values = [0.5, 0.75, 1.0, 1.5, 1.8]
    xi_strs = ['0p50', '0p75', '1p00', '1p50', '1p80']
    analyze_kraichnan_dispersion(data_dir, out_dir, xi_values, xi_strs)