# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def compute_structure_functions(delta_u, p_values):
    S_p = np.zeros((len(p_values), delta_u.shape[1]))
    abs_delta_u = np.abs(delta_u)
    for i, p in enumerate(p_values):
        S_p[i] = np.mean(abs_delta_u**p, axis=0)
    return S_p

def bootstrap_exponents_fast(delta_u, r_scales, p_values, fit_indices, n_boot=500):
    n_realizations = delta_u.shape[0]
    log_r = np.log(r_scales[fit_indices])
    abs_delta_u = np.abs(delta_u)
    abs_delta_u_p = np.array([abs_delta_u**p for p in p_values])
    zeta_p_boot = np.zeros((n_boot, len(p_values)))
    for b in range(n_boot):
        indices = np.random.choice(n_realizations, n_realizations, replace=True)
        S_p_boot = np.mean(abs_delta_u_p[:, indices, :], axis=1)
        log_S_p = np.log(S_p_boot[:, fit_indices])
        for i in range(len(p_values)):
            slope, _ = np.polyfit(log_r, log_S_p[i], 1)
            zeta_p_boot[b, i] = slope
    zeta_p_mean = np.mean(zeta_p_boot, axis=0)
    zeta_p_std = np.std(zeta_p_boot, axis=0)
    return zeta_p_mean, zeta_p_std

if __name__ == '__main__':
    np.random.seed(42)
    data_dir = "/home/node/work/projects/levy_turbulence_v1/data"
    out_dir = "data"
    path_k41 = os.path.join(data_dir, "cascade_K41_delta_u.npy")
    path_mild = os.path.join(data_dir, "cascade_logN_mild_delta_u.npy")
    path_realistic = os.path.join(data_dir, "cascade_logN_realistic_delta_u.npy")
    path_r_scales = os.path.join(data_dir, "cascade_r_scales.npy")
    r_scales = np.load(path_r_scales)
    delta_u_k41 = np.load(path_k41) * (r_scales ** (1/3))
    delta_u_mild = np.load(path_mild) * (r_scales ** (1/3))
    delta_u_realistic = np.load(path_realistic) * (r_scales ** (1/3))
    p_values = np.arange(1, 7)
    fit_indices = slice(3, 17)
    n_boot = 500
    zeta_p_k41, zeta_p_std_k41 = bootstrap_exponents_fast(delta_u_k41, r_scales, p_values, fit_indices, n_boot)
    zeta_p_mild, zeta_p_std_mild = bootstrap_exponents_fast(delta_u_mild, r_scales, p_values, fit_indices, n_boot)
    zeta_p_realistic, zeta_p_std_realistic = bootstrap_exponents_fast(delta_u_realistic, r_scales, p_values, fit_indices, n_boot)
    xi_k41 = 1 + zeta_p_k41[1]
    xi_std_k41 = zeta_p_std_k41[1]
    xi_mild = 1 + zeta_p_mild[1]
    xi_std_mild = zeta_p_std_mild[1]
    xi_realistic = 1 + zeta_p_realistic[1]
    xi_std_realistic = zeta_p_std_realistic[1]
    zeta_p_theory_k41 = p_values / 3.0
    zeta_p_theory_mild = p_values / 3.0 - 0.15 * p_values * (p_values - 3) / 18.0
    zeta_p_theory_realistic = p_values / 3.0 - 0.28 * p_values * (p_values - 3) / 18.0
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(10, 6))
    plt.errorbar(p_values, zeta_p_k41, yerr=zeta_p_std_k41, fmt='o', color='blue', label='Empirical K41 (mu=0.0)', capsize=5)
    plt.plot(p_values, zeta_p_theory_k41, '--', color='blue', label='Theory K41')
    plt.errorbar(p_values, zeta_p_mild, yerr=zeta_p_std_mild, fmt='s', color='green', label='Empirical Mild (mu=0.15)', capsize=5)
    plt.plot(p_values, zeta_p_theory_mild, '--', color='green', label='Theory Mild')
    plt.errorbar(p_values, zeta_p_realistic, yerr=zeta_p_std_realistic, fmt='^', color='red', label='Empirical Realistic (mu=0.28)', capsize=5)
    plt.plot(p_values, zeta_p_theory_realistic, '--', color='red', label='Theory Realistic')
    plt.xlabel('Moment p')
    plt.ylabel('Scaling Exponent zeta_p')
    plt.title('Structure Function Scaling Exponents vs Moment p')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = os.path.join(out_dir, "zeta_p_vs_p_" + timestamp + ".png")
    plt.savefig(plot_filename, dpi=300)
    print("Plot saved to " + plot_filename)
    print("\n" + "="*80)
    print("STRUCTURE FUNCTION SCALING EXPONENTS (zeta_p)")
    print("="*80)
    print("p | K41 Theory | K41 Emp (mu=0.0)      | Mild Theory | Mild Emp (mu=0.15)    | Real Theory | Real Emp (mu=0.28)")
    print("-" * 80)
    for i, p in enumerate(p_values):
        k41_th = str(round(zeta_p_theory_k41[i], 4))
        k41_emp = str(round(zeta_p_k41[i], 4)) + " +/- " + str(round(zeta_p_std_k41[i], 4))
        mild_th = str(round(zeta_p_theory_mild[i], 4))
        mild_emp = str(round(zeta_p_mild[i], 4)) + " +/- " + str(round(zeta_p_std_mild[i], 4))
        real_th = str(round(zeta_p_theory_realistic[i], 4))
        real_emp = str(round(zeta_p_realistic[i], 4)) + " +/- " + str(round(zeta_p_std_realistic[i], 4))
        print(str(p) + " | " + k41_th.ljust(10) + " | " + k41_emp.ljust(21) + " | " + mild_th.ljust(11) + " | " + mild_emp.ljust(21) + " | " + real_th.ljust(11) + " | " + real_emp)
    print("\n" + "="*80)
    print("SPECTRAL ROUGHNESS (xi = 1 + zeta_2)")
    print("="*80)
    print("Dataset                 | xi (Empirical)        | xi (Theoretical)")
    print("-" * 80)
    k41_xi_emp = str(round(xi_k41, 4)) + " +/- " + str(round(xi_std_k41, 4))
    k41_xi_th = str(round(1 + zeta_p_theory_k41[1], 4))
    print("K41 (mu=0.0)            | " + k41_xi_emp.ljust(21) + " | " + k41_xi_th)
    mild_xi_emp = str(round(xi_mild, 4)) + " +/- " + str(round(xi_std_mild, 4))
    mild_xi_th = str(round(1 + zeta_p_theory_mild[1], 4))
    print("Mild (mu=0.15)          | " + mild_xi_emp.ljust(21) + " | " + mild_xi_th)
    real_xi_emp = str(round(xi_realistic, 4)) + " +/- " + str(round(xi_std_realistic, 4))
    real_xi_th = str(round(1 + zeta_p_theory_realistic[1], 4))
    print("Realistic (mu=0.28)     | " + real_xi_emp.ljust(21) + " | " + real_xi_th)
    print("="*80 + "\n")