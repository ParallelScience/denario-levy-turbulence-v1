
# Lévy Flights in High-Reynolds-Number Turbulence: Effective Physical Theory

## Scientific Goal

Discover the minimal effective physical theory underlying Lévy flights / anomalous diffusion (governed by the half-Laplacian and its generalization (-Δ)^{α/2}) in high-Reynolds-number turbulence. Specifically:

1. **Eulerian angle**: What feature of the turbulent energy cascade (structure function scaling, multifractal spectrum, intermittency) determines the fractional Laplacian exponent α?
2. **Lagrangian angle**: What is the physical mechanism by which turbulent velocity fields produce Lévy-stable tracer statistics?
3. **Connection**: Can we derive a renormalization group (RG) argument connecting the Eulerian spectral exponent xi (roughness of the velocity field) to the Lagrangian fractional diffusion exponent α = 2/xi?
4. **High-Re limit**: Does the effective fractional operator converge as Re → ∞, and what is the limiting α for Kolmogorov turbulence (xi = 5/3)?

## Data Summary

Four complementary datasets, covering Eulerian cascade statistics and Lagrangian tracer dynamics.

---

## File Inventory

### 1. Random Multiplicative Energy Cascade (Eulerian, multifractal model)

Simulates the turbulent energy cascade from large to small scales via log-normal random multipliers. At each cascade level l, the velocity increment scales as delta_u_l = u_0 * exp(sum_{j≤l} log(W_j) / 3) where W_j ~ LogNormal(-σ²/2, σ²) with σ² = μ·log(2) and μ is the intermittency parameter.

**Note on data quality**: The structure function exponents zeta_p may need to be recomputed by the analysis code. The saved delta_u arrays contain the raw cascade realizations; structure functions S_p(r) = <|delta_u_l|^p> should be computed fresh from the raw data.

- `/home/node/work/projects/levy_turbulence_v1/data/cascade_K41_delta_u.npy` — shape (5000, 20), μ=0.0 (pure K41, no intermittency). Expected: zeta_p = p/3.
- `/home/node/work/projects/levy_turbulence_v1/data/cascade_logN_mild_delta_u.npy` — shape (5000, 20), μ=0.15 (mild intermittency). Expected: zeta_p = p/3 - μ·p(p-3)/18.
- `/home/node/work/projects/levy_turbulence_v1/data/cascade_logN_realistic_delta_u.npy` — shape (5000, 20), μ=0.28 (realistic turbulence). Expected: zeta_p = p/3 - 0.28·p(p-3)/18.
- `/home/node/work/projects/levy_turbulence_v1/data/cascade_K41_S_p.npy` — shape (6, 20), pre-computed S_p(l) for p=1..6 (may need recomputing).
- `/home/node/work/projects/levy_turbulence_v1/data/cascade_r_scales.npy` — shape (20,), r_l = 2^{-l}.
- `/home/node/work/projects/levy_turbulence_v1/data/cascade_zeta_p_K41.npy` — shape (6,), K41 predictions p/3.

Cascade levels: l=0..19 (largest to smallest scale). Each row in delta_u is an independent realization. Each column is a cascade level.

**Physics**: The energy spectrum E(k) ~ k^{-1-2h} where h = <log(W)/log(lambda)> = 1/3 for K41. The spectral roughness parameter xi = 1 + 2·zeta_2/2 = 1 + zeta_2 connects the cascade statistics to the Kraichnan model. For K41: xi = 5/3.

### 2. Kraichnan Model — Lagrangian Tracers in Synthetic Turbulence

Lagrangian tracer trajectories in a 3D Kraichnan velocity field (Gaussian, white-in-time, prescribed power-law spectrum E(k) ~ k^{1-xi}). Each trajectory is the x-component of a 3D tracer path; the velocity at each step is drawn from fresh random Fourier modes, making the field delta-correlated in time (exact Kraichnan model).

**Key exact result** (Falkovich, Gawedzki, Vergassola 2001): For the Kraichnan model, the tracer MSD scales as <x²(t)> ~ t^{xi} (anomalous diffusion with H = xi/2). The coarse-grained transport is governed by the fractional diffusion equation ∂_t P = -D_α(-∂²_x)^{α/2} P with **α = 2/xi**. This is the exact RG fixed point.

**Note**: The current simulation uses N_k_modes=128 Fourier modes with amplitude scaling A(k) ~ k^{(1-xi)/2}. The velocity amplitudes may be insufficiently large to produce strong anomalous diffusion at the available trajectory lengths. Analysis should directly compute the MSD scaling exponent H from the data and compare to the theoretical prediction H = xi/2.

- `/home/node/work/projects/levy_turbulence_v1/data/kraichnan_xi0p50_x.npy` — shape (200, 4001), xi=0.5, theory: H=0.25, α=4.0 (subdiffusion)
- `/home/node/work/projects/levy_turbulence_v1/data/kraichnan_xi0p75_x.npy` — shape (200, 4001), xi=0.75, theory: H=0.375, α=2.67
- `/home/node/work/projects/levy_turbulence_v1/data/kraichnan_xi1p00_x.npy` — shape (200, 4001), xi=1.0, theory: H=0.5, α=2.0 (normal diffusion)
- `/home/node/work/projects/levy_turbulence_v1/data/kraichnan_xi1p50_x.npy` — shape (200, 4001), xi=1.5, theory: H=0.75, α=1.33 (superdiffusion, Lévy-like)
- `/home/node/work/projects/levy_turbulence_v1/data/kraichnan_xi1p80_x.npy` — shape (200, 4001), xi=1.8, theory: H=0.9, α=1.11 (near-ballistic)
- `/home/node/work/projects/levy_turbulence_v1/data/kraichnan_tgrid.npy` — shape (10001,), time grid [0, 100]

Rows = independent trajectories, columns = time steps.

**For the Kolmogorov case**: Physical turbulence has xi ≈ 5/3 ≈ 1.667, giving theoretical α = 2/xi ≈ 1.2. The xi=1.5 and xi=1.8 datasets bracket this value and can be interpolated.

### 3. Kolmogorov-Spectrum Tracers (1D Synthetic Turbulence)

Lagrangian tracer positions in a 1D synthetic turbulent velocity field with Kolmogorov E(k) ~ k^{-5/3} spectrum, synthesized from 64 random-phase Fourier modes. Time evolution: each Fourier mode decorrelates at its eddy turnover time tau_k ~ k^{-2/3}. Positions are displacements from initial position.

- `/home/node/work/projects/levy_turbulence_v1/data/kolmogorov_kolmogorov_pure_disp.npy` — shape (150, 3001), pure K41 spectrum
- `/home/node/work/projects/levy_turbulence_v1/data/kolmogorov_kolmogorov_multifractal_disp.npy` — shape (150, 3001), K41 + intermittency correction (beta model exponent -0.3)
- `/home/node/work/projects/levy_turbulence_v1/data/kolmogorov_tgrid.npy` — shape (8001,), time grid [0, 400]

### 4. Lorenz-96 Model (Deterministic Turbulence Analogue)

Lorenz-96 system: dx_i/dt = (x_{i+1} - x_{i-2})·x_{i-1} - x_i + F with N=40 variables, F=8 (well-developed chaotic regime). Used as a deterministic turbulence analogue with known structure function statistics.

- `/home/node/work/projects/levy_turbulence_v1/data/lorenz96_snapshots.npy` — shape (10000, 40), x_i(t) snapshots at dt=0.05
- `/home/node/work/projects/levy_turbulence_v1/data/lorenz96_structure_functions.npy` — shape (6, 20), S_p(lag) for p=1..6, spatial lag=1..20

S_2(1) ≈ 24.6 (from simulation). Lorenz-96 with F=8 is known to exhibit anomalous diffusion in the long-time limit.

### 5. RG Flow Data

Characteristic functions of Kraichnan xi=1.5 tracer displacements at multiple time lags, computed to observe the RG flow of the effective diffusion operator.

- `/home/node/work/projects/levy_turbulence_v1/data/rg_char_func_kraichnan_xi1p5.npy` — shape (10, 200), φ(k,τ) at tau_idx=[1,5,20,50,100,200,500,1000,2000,5000]
- `/home/node/work/projects/levy_turbulence_v1/data/rg_tau_values.npy` — shape (10,), time indices
- `/home/node/work/projects/levy_turbulence_v1/data/rg_k_range.npy` — shape (200,), k values in [-2, 2]
- `/home/node/work/projects/levy_turbulence_v1/data/rg_alpha_eff.npy` — shape (10,), pre-computed effective alpha (all ≈2.0 — needs reanalysis with better fitting)

---

## Key Theoretical Connections

### Kraichnan RG Fixed Point (exact)
For the Kraichnan model with E(k) ~ k^{1-xi}:
- Scaling: <x²(t)> ~ t^{xi}, H = xi/2
- Effective operator: (-∂²)^{α/2} with **α = 2/xi**
- For K41 (xi = 5/3): **α_K41 ≈ 1.2** (close to but not exactly the half-Laplacian α=1)
- For the half-Laplacian (α=1): requires xi=2 (maximally rough Kraichnan field)

### Structure Function Connection (Eulerian → Lagrangian)
From Taylor's hypothesis and the Kolmogorov refined similarity hypothesis:
- Second-order structure function: S_2(r) ~ r^{zeta_2}
- Energy spectrum: E(k) ~ k^{-(1+zeta_2)} → xi = 1 + zeta_2
- K41: zeta_2 = 2/3, xi = 5/3, α = 6/5 = 1.2
- With intermittency (She-Leveque): zeta_2 ≈ 0.696, xi ≈ 1.696, α ≈ 1.18

### RG Argument Sketch
Under coarse-graining (integrating out wavenumbers k > Λ):
1. The tracer equation dx/dt = v(x,t) + noise renormalizes
2. The velocity correlator <v(x)v(0)> ~ |x|^{xi-1} acts as a long-range noise
3. By the generalized CLT, the accumulated displacement converges to α-stable with α = 2/xi
4. The fixed-point equation is the fractional diffusion equation ∂_t P = -D_α(-∂²)^{α/2} P
5. The RG flow: α(Λ) flows from 2 (at Λ=k_max, ballistic) to 2/xi (at Λ→0, diffusive limit)

---

## Suggested Analyses

1. **Verify Kraichnan RG**: Compute MSD scaling exponent H for each xi dataset and test H = xi/2. Compute characteristic function and extract effective α(τ) as a function of time lag to show convergence to α = 2/xi.

2. **Structure function scaling**: From cascade data, compute zeta_p(mu) for mu=0, 0.15, 0.28. Fit the log-normal formula zeta_p = p/3 - mu·p(p-3)/18. Extract xi = 1 + zeta_2 for each case.

3. **Eulerian → Lagrangian mapping**: Connect the cascade zeta_2 to the Kraichnan xi, and verify that α = 2/xi = 2/(1+zeta_2) across all intermittency values.

4. **RG flow**: Plot α_eff(τ) from the characteristic function data and fit the crossover from α≈2 (short times) to α=2/xi (long times). Extract the crossover time as a function of xi.

5. **Kolmogorov prediction**: For the physical case xi=5/3, compute the predicted fractional exponent α=6/5=1.2. Test whether the kolmogorov_pure tracer data is consistent with this.

6. **Intermittency correction**: Quantify how She-Leveque intermittency corrections shift xi and hence α away from the K41 prediction.

7. **Effective operator identification**: For each dataset, test whether the large-time tracer PDF satisfies the fractional diffusion equation with α=2/xi by checking the characteristic function form φ(k,t) = exp(-D_α|k|^α·t).
