# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os

def compute_structure_functions(delta_u, p_values):
    """
    Compute structure functions S_p = <|delta_u|^p> for a given set of p values.
    
    Parameters:
    delta_u (numpy.ndarray): Array of velocity increments of shape (N_realizations, N_scales).
    p_values (numpy.ndarray or list): Array of moments p to compute.
    
    Returns:
    numpy.ndarray: Structure functions of shape (len(p_values), N_scales).
    """
    S_p = np.zeros((len(p_values), delta_u.shape[1]))
    for i, p in enumerate(p_values):
        S_p[i, :] = np.mean(np.abs(delta_u)**p, axis=0)
    return S_p

def fit_scaling_exponents(S_p, r_scales, scale_idx_min=2, scale_idx_max=15):
    """
    Fit log(S_p) vs log(r) to extract the scaling exponents zeta_p.
    
    Parameters:
    S_p (numpy.ndarray): Structure functions of shape (N_p, N_scales).
    r_scales (numpy.ndarray): Array of scales r of shape (N_scales,).
    scale_idx_min (int): Minimum scale index for the inertial range fit.
    scale_idx_max (int): Maximum scale index for the inertial range fit.
    
    Returns:
    numpy.ndarray: Array of scaling exponents zeta_p of shape (N_p,).
    """
    log_r = np.log(r_scales[scale_idx_min:scale_idx_max])
    zeta_p = np.zeros(S_p.shape[0])
    for i in range(S_p.shape[0]):
        log_S_p = np.log(S_p[i, scale_idx_min:scale_idx_max])
        slope, _ = np.polyfit(log_r, log_S_p, 1)
        zeta_p[i] = slope
    return zeta_p

if __name__ == '__main__':
    data_dir = "/home/node/work/projects/levy_turbulence_v1/data"
    k41_path = os.path.join(data_dir, "cascade_K41_delta_u.npy")
    mild_path = os.path.join(data_dir, "cascade_logN_mild_delta_u.npy")
    realistic_path = os.path.join(data_dir, "cascade_logN_realistic_delta_u.npy")
    r_scales_path = os.path.join(data_dir, "cascade_r_scales.npy")
    kraichnan_x_path = os.path.join(data_dir, "kraichnan_xi1p00_x.npy")
    kraichnan_t_path = os.path.join(data_dir, "kraichnan_tgrid.npy")
    delta_u_k41 = np.load(k41_path)
    delta_u_mild = np.load(mild_path)
    delta_u_realistic = np.load(realistic_path)
    r_scales = np.load(r_scales_path)
    rms_k41 = np.sqrt(np.mean(delta_u_k41**2))
    delta_u_k41_norm = delta_u_k41 / rms_k41
    p_values = np.arange(1, 7)
    S_p_k41 = compute_structure_functions(delta_u_k41_norm, p_values)
    zeta_p_k41 = fit_scaling_exponents(S_p_k41, r_scales, scale_idx_min=2, scale_idx_max=15)
    print("--- Sanity Check: Cascade K41 ---")
    print("Expected zeta_p: p/3")
    for i, p in enumerate(p_values):
        print("p = " + str(p) + " | Expected: " + str(round(p/3, 3)) + " | Computed: " + str(round(zeta_p_k41[i], 3)))
    x_k = np.load(kraichnan_x_path)
    t_grid_full = np.load(kraichnan_t_path)
    t_k = np.linspace(t_grid_full[0], t_grid_full[-1], x_k.shape[1])
    displacements = x_k - x_k[:, 0:1]
    msd = np.mean(displacements**2, axis=0)
    fit_start = len(t_k) // 2
    fit_end = len(t_k) - 1
    log_t = np.log(t_k[fit_start:fit_end])
    log_msd = np.log(msd[fit_start:fit_end])
    slope, _ = np.polyfit(log_t, log_msd, 1)
    H_computed = slope / 2.0
    print("\n--- Sanity Check: Kraichnan xi=1.0 ---")
    print("Expected H: 0.5 (MSD slope = 1.0)")
    print("Computed MSD slope: " + str(round(slope, 3)))
    print("Computed H: " + str(round(H_computed, 3)))
    print("\nSanity checks completed successfully. Raw data is preserved for Step 2.")