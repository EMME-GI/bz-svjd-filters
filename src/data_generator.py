import numpy as np
import pandas as pd
from numba import njit

@njit
def _seed_numba(seed):
    # Numba keeps its own RNG state: np.random.seed() called from plain Python does NOT seed it.
    np.random.seed(seed)

def seed_all(seed):
    """Seed both NumPy's generator (measurement noise) and Numba's generator (latent paths)."""
    np.random.seed(seed)
    _seed_numba(seed)

# Numba requires non-dictionary parameters
@njit(fastmath=True)
def _generate_paths_numba(k1, vbar1, sig1, mu1, 
                          k2, vbar2, sig2, 
                          k3, mu3, rho3, 
                          lam_m, c0_m, c1_m, c2_m, c3_m,
                          N_days, steps_per_day,
                          v1_0, v2_0, v3_0):
    
    total_steps = N_days * steps_per_day
    h = 1.0 / (252.0 * steps_per_day) # h corresponds to dt_micro of the paper
    sqrt_h = np.sqrt(h)
    
    # Fortran-contiguous (Column-Major) matrix for Eigen
    v_daily = np.zeros((3, N_days)) 
    
    v1, v2, v3 = v1_0, v2_0, v3_0
    v_daily[0, 0], v_daily[1, 0], v_daily[2, 0] = v1, v2, v3
    
    for t in range(1, N_days):
        for _ in range(steps_per_day):
            # Full Truncation Scheme for numerical stability (set to zero if becomes negative due to discretization)
            v1_pos = max(v1, 0.0)
            v2_pos = max(v2, 0.0)
            v3_pos = max(v3, 0.0)
            
            # 1. Computing negative jump intensities (Eq. 9)
            c_minus = c0_m + c1_m * v1_pos + c2_m * v2_pos + c3_m * v3_pos
            prob_salto = 1.0 - np.exp(-c_minus * h)
            
            # 2. Indipendent Bernoulli trials (n_t^- e n_t^y)
            n_minus = 1.0 if np.random.rand() < prob_salto else 0.0
            n_y     = 1.0 if np.random.rand() < prob_salto else 0.0
            
            jump_x2 = 0.0
            jump_y2 = 0.0
            
            # 3. Computing jump dimensions (only if there are jumps)
            if n_minus > 0:
                tilde_x = np.random.exponential(1.0 / lam_m)
                jump_x2 = tilde_x**2
                
            if n_y > 0:
                tilde_y = np.random.exponential(1.0 / lam_m)
                jump_y2 = tilde_y**2
                
            # 4. Brownian increments
            xi_1 = np.random.randn()
            xi_2 = np.random.randn()
            
            # 5. Euler-Maruyama dynamics (Eq. 7 - 9)
            # Note: the paper defines mu_1 * x_t^{-2}, but x_t^- = -n_t^- * tilde_x, 
            # the square incorporates the sign and the Bernoulli (jump_x2 exists only if n_minus=1)
            v1 += k1 * (vbar1 - v1) * h + sig1 * np.sqrt(v1_pos * h) * xi_1 + mu1 * jump_x2
            v2 += k2 * (vbar2 - v2) * h + sig2 * np.sqrt(v2_pos * h) * xi_2
            v3 += -k3 * v3 * h + mu3 * ((1.0 - rho3) * jump_x2 + rho3 * jump_y2)
            
        # Saving at the end of the day (we impose them to be positive)
        v_daily[0, t] = max(v1, 0.0)
        v_daily[1, t] = max(v2, 0.0)
        v_daily[2, t] = max(v3, 0.0)
            
    return v_daily

def generate_high_fidelity_data(theta, N_days=10000, steps_per_day=10, v0=None):
    # Wrapper Python to extract the parameters from the dictionary and call Numba
    # v0 = initial state (v1, v2, v3); default: long-run means for v1, v2 and 1e-4 for v3
    if v0 is None:
        v0 = (theta['vbar1'], theta['vbar2'], 1e-4)
    v_daily = _generate_paths_numba(
        theta['k1'], theta['vbar1'], theta['sig1'], theta['mu1'],
        theta['k2'], theta['vbar2'], theta['sig2'],
        theta['k3'], theta['mu3'], theta['rho3'],
        theta['lam_m'], theta['c0_m'], theta['c1_m'], theta['c2_m'], theta['c3_m'],
        N_days, steps_per_day, float(v0[0]), float(v0[1]), float(v0[2])
    )
    # Converting to Fortran-contiguous for Eigen
    return np.asfortranarray(v_daily)


if __name__ == "__main__":
    seed_all(42)
    
    theta_true = {'k1': 0.6269, 'vbar1': 0.0267, 'sig1': 0.1747, 'rho1': -0.9899, 'k2': 0.7060, 'vbar2': 0.0195, 'sig2': 0.1428, 'rho2': -0.9898, 'k3': 13.7027, 'mu3': 1.3682, 'rho3': 0.7306, 'mu1': 0.0831, 'lam_m': 13.9565, 'lam_p': 248.4211, 'c0_m': 0.0, 'c1_m': 55.4945, 'c2_m': 1.6434, 'c3_m': 451.6922, 'c0_p': 4.5536, 'c1_p': 256.4476, 'c2_p': 212.3698, 'c3_p': 0.0, 'eta': 0.0, 'r': 0.0, 'delta': 0.0}

    N_giorni = 10000 
    # Array forn: order='F', shape=(3, N_days)
    v_history_T = generate_high_fidelity_data(theta_true, N_days=N_giorni, steps_per_day=10)
    
    # Observation matrices (G0: 12x1, G1: 12x3); needs the compiled kfdnf_fast module (imported by main_calibration)
    from cumulant_engine import AFTCumulantCalculator
    from main_calibration import build_full_observation_matrices
    G0_true, G1_true = build_full_observation_matrices(theta_true, AFTCumulantCalculator(theta_true))
    
    # BROADCASTING AND VECTOR ALGEBRA 
    # The result of (12x3) @ (3xN) is (12xN). 
    # Since v_history_T is in F-order, the dot product uses optimal spatial cache.
    Z_series_clean = G0_true + G1_true @ v_history_T
    
    # Adding structural noise
    empirical_vars = np.var(Z_series_clean, axis=1, keepdims=True)
    deltas = np.array([0.1646]*4 + [0.2277]*4 + [0.2693]*4).reshape(12, 1)
    
    noise_std = deltas * np.sqrt(empirical_vars)
    noise_matrix = np.random.randn(12, N_giorni) * noise_std
    
    # Forcing F-order for the C++ layer
    Z_series_noisy = np.asfortranarray(Z_series_clean + noise_matrix)
    
    # Saving
    df = pd.DataFrame(Z_series_noisy.T) # Pandas gradisce (N, features) per CSV
    df.to_csv("traiettorie_bz_10000.csv", index=False, header=False) 
    
    df_true_states = pd.DataFrame(v_history_T.T, columns=['v1', 'v2', 'v3'])
    df_true_states.to_csv("veri_stati_latenti_10000.csv", index=False)