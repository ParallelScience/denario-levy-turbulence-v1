# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os

def main():
    data_dir = "data/"
    cascade_paths = {
        "K41": "/home/node/work/projects/levy_turbulence_v1/data/cascade_K41_delta_u.npy",
        "logN_mild": "/home/node/work/projects/levy_turbulence_v1/data/cascade_logN_mild_delta_u.npy",
        "logN_realistic": "/home/node/work/projects/levy_turbulence_v1/data/cascade_logN_realistic_delta_u.npy"
    }
    r_scales_path = "/home/node/work/projects/levy_turbulence_v1/data/cascade_r_scales.npy"
    r_scales = np.load(r_scales_path)
    log_r = np.log(r_scales)
    np.save(os.path.join(data_dir, "preprocessed_cascade_r_scales.npy"), r_scales)
    print("--- Cascade Datasets Analysis ---")
    for name, path in cascade_paths.items():
        delta_u = np.load(path)
        if name == "K41":
            print("Verifying K41 structure function exponents (zeta_p approx p/3):")
            p_values = np.arange(1, 7)
            for p in p_values:
                S_p = np.mean(np.abs(delta_u)**p, axis=0)
                slope, intercept = np.polyfit(log_r, np.log(S_p), 1)
                expected = p / 3.0
                print("  p=" + str(p) + ": empirical zeta_" + str(p) + " = " + str(round(slope, 4)) + ", expected = " + str(round(expected, 4)))
        rms_u0 = np.sqrt(np.mean(delta_u[:, 0]**2))
        delta_u_norm = delta_u / rms_u0
        save_path = os.path.join(data_dir, "preprocessed_cascade_" + name + "_delta_u.npy")
        np.save(save_path, delta_u_norm)
        print("Saved normalized " + name + " cascade data to " + save_path)
    print("\n--- Kraichnan Tracer Data Analysis ---")
    kraichnan_files = [
        "kraichnan_xi0p50_x.npy",
        "kraichnan_xi0p75_x.npy",
        "kraichnan_xi1p00_x.npy",
        "kraichnan_xi1p50_x.npy",
        "kraichnan_xi1p80_x.npy"
    ]
    tgrid_path = "/home/node/work/projects/levy_turbulence_v1/data/kraichnan_tgrid.npy"
    tgrid = np.load(tgrid_path)
    np.save(os.path.join(data_dir, "preprocessed_kraichnan_tgrid.npy"), tgrid)
    for fname in kraichnan_files:
        path = os.path.join("/home/node/work/projects/levy_turbulence_v1/data", fname)
        x = np.load(path)
        has_nans = np.isnan(x).any()
        shape = x.shape
        mean_per_traj = np.mean(x, axis=1)
        var_per_traj = np.var(x, axis=1)
        print("Dataset: " + fname)
        print("  Shape: " + str(shape) + ", Contains NaNs: " + str(has_nans))
        print("  Mean of trajectory means: " + str(round(np.mean(mean_per_traj), 4)))
        print("  Mean of trajectory variances: " + str(round(np.mean(var_per_traj), 4)))
        x_mean_sub = x - np.mean(x, axis=0, keepdims=True)
        save_path = os.path.join(data_dir, "preprocessed_" + fname)
        np.save(save_path, x_mean_sub)
        print("  Saved mean-subtracted data to " + save_path)
    print("\n--- Lorenz-96 Data Analysis ---")
    lorenz_path = "/home/node/work/projects/levy_turbulence_v1/data/lorenz96_snapshots.npy"
    lorenz = np.load(lorenz_path)
    lorenz_mean_sub = lorenz - np.mean(lorenz, axis=0, keepdims=True)
    save_path = os.path.join(data_dir, "preprocessed_lorenz96_snapshots.npy")
    np.save(save_path, lorenz_mean_sub)
    print("Saved mean-subtracted Lorenz-96 data to " + save_path)
    print("  Lorenz-96 shape: " + str(lorenz.shape))

if __name__ == "__main__":
    main()