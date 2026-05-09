# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import time

def compute_char_func(x, k_range):
    phi = np.zeros_like(k_range)
    for i, k in enumerate(k_range):
        phi[i] = np.mean(np.cos(k * x))
    return phi

def extract_alpha(k_range, phi, k_min=1e-4):
    mask = (k_range > k_min) & (phi < 0.9) & (phi > 0.3)
    k_fit = k_range[mask]
    phi_fit = phi[mask]
    if len(k_fit) < 3:
        return np.nan, np.nan, np.array([]), np.array([]), np.array([])
    y = np.log(-np.log(phi_fit))
    x = np.log(k_fit)
    slope, intercept = np.polyfit(x, y, 1)
    preds = slope * x + intercept
    residuals = y - preds
    return slope, intercept, k_fit, phi_fit, residuals

if __name__ == '__main__':
    data_dir = "/home/node/work/projects/levy_turbulence_v1/data"
    out_dir = "data/"
    pure_path = os.path.join(data_dir, "kolmogorov_kolmogorov_pure_disp.npy")
    multi_path = os.path.join(data_dir, "kolmogorov_kolmogorov_multifractal_disp.npy")
    tgrid_path = os.path.join(data_dir, "kolmogorov_tgrid.npy")
    pure_disp = np.load(pure_path)
    multi_disp = np.load(multi_path)
    tgrid = np.load(tgrid_path)
    x_pure = pure_disp[:, -1]
    x_multi = multi_disp[:, -1]
    print("--- Kolmogorov Tracer Analysis ---")
    print("Number of trajectories: " + str(len(x_pure)))
    print("Max time t: " + str(tgrid[-1]))
    print("Std dev of x_pure: " + str(round(np.std(x_pure), 2)))
    print("Std dev of x_multi: " + str(round(np.std(x_multi), 2)))
    k_range = np.logspace(-4, 1, 500)
    phi_pure = compute_char_func(x_pure, k_range)
    phi_multi = compute_char_func(x_multi, k_range)
    k_min = 1e-3
    alpha_pure, int_pure, k_fit_pure, phi_fit_pure, res_pure = extract_alpha(k_range, phi_pure, k_min)
    alpha_multi, int_multi, k_fit_multi, phi_fit_multi, res_multi = extract_alpha(k_range, phi_multi, k_min)
    print("\n--- Extracted Empirical Alpha ---")
    print("Theoretical alpha (Pure K41): 1.2")
    print("Empirical alpha (Pure K41): " + str(round(alpha_pure, 4)))
    print("Theoretical alpha (Multifractal): ~1.18")
    print("Empirical alpha (Multifractal): " + str(round(alpha_multi, 4)))
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax1 = axes[0]
    ax1.plot(k_range, phi_pure, 'b-', label='Pure K41')
    ax1.plot(k_range, phi_multi, 'r-', label='Multifractal')
    ax1.set_xscale('log')
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel('Wavenumber k')
    ax1.set_ylabel('Characteristic Function phi(k, t_max)')
    ax1.set_title('Characteristic Functions at Large Times')
    ax1.legend()
    ax1.grid(True, which='both', ls='--', alpha=0.5)
    ax2 = axes[1]
    if len(k_fit_pure) > 0:
        ax2.plot(np.log(k_fit_pure), np.log(-np.log(phi_fit_pure)), 'b.', label='Pure K41 Data')
        ax2.plot(np.log(k_fit_pure), alpha_pure * np.log(k_fit_pure) + int_pure, 'k--', label='Fit Pure (alpha=' + str(round(alpha_pure, 2)) + ')')
    if len(k_fit_multi) > 0:
        ax2.plot(np.log(k_fit_multi), np.log(-np.log(phi_fit_multi)), 'r.', label='Multifractal Data')
        ax2.plot(np.log(k_fit_multi), alpha_multi * np.log(k_fit_multi) + int_multi, 'k:', label='Fit Multi (alpha=' + str(round(alpha_multi, 2)) + ')')
    ax2.set_xlabel('log(k)')
    ax2.set_ylabel('log(-log(phi))')
    ax2.set_title('Tail Index Extraction')
    ax2.legend()
    ax2.grid(True)
    ax3 = axes[2]
    if len(k_fit_pure) > 0:
        ax3.plot(np.log(k_fit_pure), res_pure, 'b-', label='Residuals Pure K41')
    if len(k_fit_multi) > 0:
        ax3.plot(np.log(k_fit_multi), res_multi, 'r-', label='Residuals Multifractal')
    ax3.set_xlabel('log(k)')
    ax3.set_ylabel('Fit Residuals')
    ax3.set_title('Residuals of Fits')
    ax3.legend()
    ax3.grid(True)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = "kolmogorov_analysis_5_" + str(timestamp) + ".png"
    plot_filepath = os.path.join(out_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    print("\nPlot saved to " + plot_filepath)