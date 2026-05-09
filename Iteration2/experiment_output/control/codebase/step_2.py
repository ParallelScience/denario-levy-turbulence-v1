# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, '/home/node/data/compsep_data/')
import time
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = 'data/'
    r_scales = np.load(os.path.join(data_dir, 'preprocessed_cascade_r_scales.npy'))
    log_r = np.log(r_scales)
    cascade_names = ['K41', 'logN_mild', 'logN_realistic']
    cascade_data = {}
    for name in cascade_names:
        multipliers = np.load(os.path.join(data_dir, 'preprocessed_cascade_' + name + '_delta_u.npy'))
        delta_u_physical = multipliers * (r_scales**(1/3))
        cascade_data[name] = delta_u_physical
    lorenz96_Sp = np.load('/home/node/work/projects/levy_turbulence_v1/data/lorenz96_structure_functions.npy')
    lorenz96_lags = np.arange(1, 21)
    p_values = np.arange(1, 7)
    zeta_results = {}
    xi_values = {}
    print('--- Eulerian Spectral Roughness Analysis ---')
    for name in cascade_names:
        delta_u = cascade_data[name]
        zetas = []
        Sp_all = []
        for p in p_values:
            Sp = np.mean(np.abs(delta_u)**p, axis=0)
            Sp_all.append(Sp)
            slope, intercept = np.polyfit(log_r, np.log(Sp), 1)
            zetas.append(slope)
        np.save(os.path.join(data_dir, 'preprocessed_cascade_' + name + '_S_p.npy'), np.array(Sp_all))
        zeta_results[name] = np.array(zetas)
        xi = 1.0 + zetas[1]
        xi_values[name] = xi
        print('Cascade ' + name + ':')
        print('  zeta_p: ' + str(np.round(zetas, 4)))
        print('  Spectral roughness xi (1 + zeta_2): ' + str(round(xi, 4)))
    lorenz_zetas = []
    fit_lags = lorenz96_lags[:3]
    log_fit_lags = np.log(fit_lags)
    for i, p in enumerate(p_values):
        Sp = lorenz96_Sp[i, :3]
        slope, intercept = np.polyfit(log_fit_lags, np.log(Sp), 1)
        lorenz_zetas.append(slope)
    zeta_results['Lorenz-96'] = np.array(lorenz_zetas)
    xi_lorenz = 1.0 + lorenz_zetas[1]
    xi_values['Lorenz-96'] = xi_lorenz
    print('Lorenz-96 (fitted over lags 1-3):')
    print('  zeta_p: ' + str(np.round(lorenz_zetas, 4)))
    print('  Spectral roughness xi (1 + zeta_2): ' + str(round(xi_lorenz, 4)))
    print('\n--- Comparison of Spectral Roughness xi ---')
    for name in cascade_names:
        print('  ' + name + ' xi: ' + str(round(xi_values[name], 4)))
    print('  Lorenz-96 xi: ' + str(round(xi_values['Lorenz-96'], 4)))
    plt.figure(figsize=(8, 6))
    plt.plot(p_values, p_values / 3.0, 'k--', label='K41 Theory (p/3)', linewidth=2, zorder=1)
    markers = ['o', 's', '^', 'd']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i, (name, zetas) in enumerate(zeta_results.items()):
        if name == 'K41':
            plt.plot(p_values, zetas, marker=markers[i], color=colors[i], linestyle='', label=name, markersize=10, zorder=2)
        else:
            plt.plot(p_values, zetas, marker=markers[i], color=colors[i], linestyle='-', label=name, linewidth=2, markersize=8, zorder=2)
    plt.xlabel('Moment order p', fontsize=14)
    plt.ylabel('Scaling exponent zeta_p', fontsize=14)
    plt.title('Structure Function Scaling Exponents zeta_p vs p', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    timestamp = int(time.time())
    plot_filename = 'step_2_zeta_p_vs_p_2_' + str(timestamp) + '.png'
    plot_filepath = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print('Plot saved to ' + plot_filepath)
    json_filepath = os.path.join(data_dir, 'extracted_xi_values.json')
    with open(json_filepath, 'w') as f:
        json.dump(xi_values, f, indent=4)
    print('Extracted xi values saved to ' + json_filepath)

if __name__ == '__main__':
    main()