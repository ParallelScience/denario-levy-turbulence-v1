# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import time

def extract_effective_alpha(phi, tau_values, k_range, k_min=0.05):
    alpha_eff = np.zeros(len(tau_values))
    residuals_list = []
    k_fit_list = []
    for i, tau in enumerate(tau_values):
        phi_tau = np.abs(phi[i, :])
        mask = (k_range > k_min) & (phi_tau < 0.999) & (phi_tau > 1e-6)
        k_fit = k_range[mask]
        phi_fit = phi_tau[mask]
        if len(k_fit) < 3:
            print("tau = " + str(tau) + " | Not enough points to fit.")
            alpha_eff[i] = np.nan
            residuals_list.append(np.array([]))
            k_fit_list.append(np.array([]))
            continue
        y = np.log(-np.log(phi_fit))
        x = np.log(k_fit)
        weights = phi_fit
        try:
            slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(weights))
        except Exception as e:
            print("tau = " + str(tau) + " | Fit failed: " + str(e))
            slope = np.nan
            intercept = np.nan
        alpha_eff[i] = slope
        if not np.isnan(slope):
            preds = slope * x + intercept
            residuals = y - preds
        else:
            residuals = np.full_like(y, np.nan)
        residuals_list.append(residuals)
        k_fit_list.append(k_fit)
        print("tau = " + str(tau) + " | Extracted alpha_eff = " + str(round(slope, 4)))
    return alpha_eff, residuals_list, k_fit_list

def apply_rolling_average(data, window_size=3):
    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed[i] = np.nanmean(data[start:end])
    return smoothed

if __name__ == '__main__':
    data_dir = "data/"
    phi_path = "/home/node/work/projects/levy_turbulence_v1/data/rg_char_func_kraichnan_xi1p5.npy"
    tau_path = "/home/node/work/projects/levy_turbulence_v1/data/rg_tau_values.npy"
    k_path = "/home/node/work/projects/levy_turbulence_v1/data/rg_k_range.npy"
    phi = np.load(phi_path)
    tau_values = np.load(tau_path)
    k_range = np.load(k_path)
    print("--- RG Flow Analysis for Kraichnan xi=1.5 ---")
    xi = 1.5
    alpha_theory = 2.0 / xi
    k_min = 0.05
    alpha_eff, residuals_list, k_fit_list = extract_effective_alpha(phi, tau_values, k_range, k_min)
    alpha_eff_smooth = apply_rolling_average(alpha_eff, window_size=3)
    plt.rcParams['text.usetex'] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(tau_values, alpha_eff, 'o-', alpha=0.4, label='Raw alpha_eff')
    ax1.plot(tau_values, alpha_eff_smooth, 's-', linewidth=2, label='Smoothed alpha_eff')
    ax1.axhline(2.0, color='k', linestyle='--', label='Ballistic limit (alpha=2)')
    ax1.axhline(alpha_theory, color='r', linestyle='--', label='Theoretical fixed point (alpha=' + str(round(alpha_theory, 3)) + ')')
    ax1.set_xscale('log')
    ax1.set_xlabel('Time lag tau')
    ax1.set_ylabel('Effective Levy index alpha_eff')
    ax1.set_title('RG Flow of Effective Diffusion Operator (xi=1.5)')
    ax1.legend()
    ax1.grid(True, which='both', ls='--', alpha=0.5)
    for i, tau in enumerate(tau_values):
        if len(k_fit_list[i]) > 0 and not np.isnan(alpha_eff[i]):
            ax2.plot(k_fit_list[i], residuals_list[i], 'o-', markersize=4, alpha=0.7, label='tau=' + str(tau))
    ax2.set_xscale('log')
    ax2.set_xlabel('Wavenumber k')
    ax2.set_ylabel('Fit Residuals (log(-log|phi|) space)')
    ax2.set_title('Residuals of Characteristic Function Fits')
    ax2.legend(fontsize='x-small', ncol=2)
    ax2.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = "rg_flow_analysis_4_" + str(timestamp) + ".png"
    plot_filepath = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    print("Saved to " + plot_filepath)