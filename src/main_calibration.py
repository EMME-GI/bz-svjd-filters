import numpy as np
import pandas as pd

# Importing the compiled C++ filters (module kfdnf_fast, built into this folder) and the mathematical engine
import kfdnf_fast
from cumulant_engine import AFTCumulantCalculator

def build_full_observation_matrices(theta, engine):
    """
    Buils Gamma_0 e Gamma_1 grouping by moment:
    - Rows 0-3: Variances (1, 3, 6, 12 months)
    - Rows 4-7: Skewness (1, 3, 6, 12 months)
    - Rows 8-11: Kurtosis (1, 3, 6, 12 months)
    Retursn F-contiguous tensors for Eigen.
    """
    tau_list = [21.0/252.0, 63.0/252.0, 126.0/252.0, 252.0/252.0] 
    
    g1_var, g1_skew, g1_kurt = [], [], []
    g0_var, g0_skew, g0_kurt = [], [], []
    
    for tau in tau_list:
        _, g2, g3, g4, a2, a3, a4 = engine.get_cumulant_loadings(tau)
        
        g1_var.append(g2.T)
        g1_skew.append(g3.T)
        g1_kurt.append(g4.T)
        
        g0_var.append(a2)
        g0_skew.append(a3)
        g0_kurt.append(a4)
        
    # Piling the lists following the dataframe order
    Gamma1_full = np.vstack(g1_var + g1_skew + g1_kurt)  # Dimensions: 12 x 3
    Gamma0_full = np.vstack(g0_var + g0_skew + g0_kurt)  # Dimensions: 12 x 1
    
    return np.asfortranarray(Gamma0_full), np.asfortranarray(Gamma1_full)

def build_omega_matrix(cumulants_df, delta2=0.1646, delta3=0.2277, delta4=0.2693): # Used here only for examples,
                                                                                   # It's better to do in on the fly in practical applications
    """
    Builds the covariance matrix of the measurement error (Omega).
    NOTE: In a loop MLE, we call this function only once outside the objective function.
    """
    Omega = np.zeros((12, 12), order='F')
    
    # Vectorized computation of the empirical variances
    empirical_variances = cumulants_df.var(axis=0).values
    
    # deltas
    deltas = np.zeros(12)
    deltas[0:4] = delta2 ** 2
    deltas[4:8] = delta3 ** 2
    deltas[8:12] = delta4 ** 2
    
    np.fill_diagonal(Omega, deltas * empirical_variances)
            
    return Omega

def build_phi_for_pf(theta, dt=1.0/252.0):
    """
    Builds Phi0 e Phi1 for the Particle Filter.
    The drift contains only the continous deterministic term.
    Jumps impact will be computed from the C++ file at each step.
    """
    base_drift = np.array([
        theta['k1'] * theta['vbar1'] * dt,
        theta['k2'] * theta['vbar2'] * dt,
        0.0 
    ])
    
    Phi1 = np.diag([
        1.0 - theta['k1'] * dt,
        1.0 - theta['k2'] * dt,
        1.0 - theta['k3'] * dt
    ])
    
    # Returns Eigen-friednly data format (Column-Major)
    return np.asfortranarray(base_drift.flatten(order='F')), np.asfortranarray(Phi1)

def build_phi_for_FO(theta, dt=1.0/252.0):
    """
    Builds Phi0 and Phi1 pre-linearized for the FO filter (2018).
    It's the Equazion (27) at page 13 of the paper.
    """
    # FO (2018) defines \bar{\lambda}_- = 2 / \lambda_-^2 under Eq. (27)
    lam_m_bar = 2.0 / (theta['lam_m']**2)
    
    # Building Phi0 (Eq. 27 FO 2018)
    Phi0_linear = dt * np.array([
        theta['k1'] * theta['vbar1'] + theta['mu1'] * lam_m_bar * theta['c0_m'],
        theta['k2'] * theta['vbar2'],
        theta['mu3'] * lam_m_bar * theta['c0_m']
    ])
    
    # Building K1 matrix following Eq. (27)
    K1 = dt * np.array([
        [-theta['k1'] + theta['mu1'] * lam_m_bar * theta['c1_m'], 
          theta['mu1'] * lam_m_bar * theta['c2_m'], 
          theta['mu1'] * lam_m_bar * theta['c3_m']],
          
        [ 0.0, 
         -theta['k2'], 
          0.0],
          
        [ theta['mu3'] * lam_m_bar * theta['c1_m'], 
          theta['mu3'] * lam_m_bar * theta['c2_m'], 
         -theta['k3'] + theta['mu3'] * lam_m_bar * theta['c3_m']]
    ])
    
    # \Phi_1 = I_3 + K_1 following Eq. (27)
    Phi1_linear = np.eye(3) + K1
    
    # Returns F-contiguous tensors for the Eigen backend in C++
    return np.asfortranarray(Phi0_linear.flatten(order='F')), np.asfortranarray(Phi1_linear)

def build_phi_and_grid(theta, dt=1.0/252.0, K=5):
    """
    Builds the vectoralized grid of the DNF for K+1 nodes.
    Returns Phi0_mat in Column-Major (F-contiguous) format for Eigen.
    """
    Phi1 = np.diag([
        1.0 - theta['k1'] * dt,
        1.0 - theta['k2'] * dt,
        1.0 - theta['k3'] * dt
    ])

    base_drift = np.array([
        theta['k1'] * theta['vbar1'] * dt,
        theta['k2'] * theta['vbar2'] * dt,
        0.0 
    ])

    if K == 0:
        return np.asfortranarray(base_drift.reshape(3,1)), np.asfortranarray(Phi1), np.array([1.0])

    lam_m = theta['lam_m']
    
    # 1. Definition of the K+1 space (including the 0 node for the no-jumps case)
    E_x_inf = -1.0 / lam_m      # long term => only exponential term is left
    V_x_inf = 1.0 / (lam_m**2)  # as before
    delta_K = 3.0 + np.log(K) if K > 1 else 0.0 # different from the paper but valid. Works better in practice.
    base_val = (E_x_inf - delta_K * np.sqrt(V_x_inf))
    
    nodes = np.zeros(K + 1)
    nodes[1:] = base_val * (np.arange(1, K + 1) / K)
    
    # 2. Computing marginal probabilities
    bounds_L = np.zeros(K)
    bounds_U = np.zeros(K)
    
    if K == 1:
        bounds_L[0] = -np.inf
        bounds_U[0] = 0.0
    else:
        bounds_L[0] = (nodes[1] + nodes[2]) / 2.0
        bounds_U[0] = 0.0
        for idx in range(1, K - 1): # idx = i - 1, where i is the paper index
            bounds_L[idx] = (nodes[idx+1] + nodes[idx+2]) / 2.0 
            bounds_U[idx] = (nodes[idx] + nodes[idx+1]) / 2.0
        bounds_L[-1] = -np.inf
        bounds_U[-1] = (nodes[-2] + nodes[-1]) / 2.0

    # Static probabilities
    probs = np.zeros(K + 1)
    probs[1:] = np.exp(lam_m * bounds_U) - np.exp(lam_m * bounds_L)
    probs[0] = 1.0 # P(no jump), handled dynamically in C++
    
    # 3. Vectoralization of the (K+1)^2 combinations
    X_grid, Y_grid = np.meshgrid(nodes, nodes, indexing='ij')
    
    X2_flat = (X_grid.flatten())**2
    Y2_flat = (Y_grid.flatten())**2
    
    # 4. Fortran-order initialization for Eigen
    n_total_nodes = (K + 1)**2
    Phi0_mat = np.zeros((3, n_total_nodes), order='F')
    
    Phi0_mat[0, :] = base_drift[0] + theta['mu1'] * X2_flat
    Phi0_mat[1, :] = base_drift[1]
    Phi0_mat[2, :] = base_drift[2] + theta['mu3'] * ((1.0 - theta['rho3']) * X2_flat + theta['rho3'] * Y2_flat)
    
    return Phi0_mat, np.asfortranarray(Phi1), np.asfortranarray(probs)

def extract_latent_states(theta, df_cumulants, n_nodes=5):
    """
    Builds the tensors via the C++ filters.
    """
    dt = 1.0 / 252.0
    engine = AFTCumulantCalculator(theta)
    
    # 1. Observation matrices
    Gamma0, Gamma1 = build_full_observation_matrices(theta, engine)
    
    # 2. Transiction matrices and Grid 
    Phi0_mat, Phi1, node_probs = build_phi_and_grid(theta, dt, n_nodes)
    
    # 3. Omega Matrix
    # In following scripts, Omega will be an parameter of some function to compute it one time only
    Omega = build_omega_matrix(df_cumulants) 
    
    # 4. Timeseries in a F-contiguous format for Eigen Map
    Z_series = np.asfortranarray(df_cumulants.values)
    
    # 5. C++ filter inizialization
    dnf = kfdnf_fast.DiscreteNonlinearFilter()
    
    # 6. Flattening
    Gamma0 = np.asfortranarray(Gamma0).flatten()
    node_probs = np.asfortranarray(node_probs).flatten()
    
    # 7. Execution
    ll = dnf.filter_series(
        Z_series, Phi0_mat, Phi1, Gamma0, Gamma1, node_probs, Omega, 
        theta['sig1'], theta['sig2'], dt,
        theta['c0_m'], theta['c1_m'], theta['c2_m'], theta['c3_m']
    )
    
    v_hist = dnf.get_v_history()
    
    return v_hist, ll

if __name__ == "__main__":
    # BZ Table 5 parameters
    theta_BZ = {
        'k1': 0.6269, 'vbar1': 0.0267, 'sig1': 0.1747, 'rho1': -0.9899,
        'k2': 0.7060, 'vbar2': 0.0195, 'sig2': 0.1428, 'rho2': -0.9898,
        'k3': 13.7027, 'mu3': 1.3682, 'rho3': 0.7306,
        'mu1': 0.0831, 'lam_m': 13.9565, 'lam_p': 248.4211,
        'c0_m': 0.0, 'c1_m': 55.4945, 'c2_m': 1.6434, 'c3_m': 451.6922,
        'c0_p': 4.5536, 'c1_p': 256.4476, 'c2_p': 212.3698, 'c3_p': 0.0,
        'eta': 0.0, 'r': 0.0, 'delta': 0.0
    }
    
    col_names = [f"Var_{m}m" for m in [1,3,6,12]] + \
                [f"Skew_{m}m" for m in [1,3,6,12]] + \
                [f"Kurt_{m}m" for m in [1,3,6,12]]
                
    # Generating dummy data
    T, n_maturities = 100, 4

    c2 = np.abs(np.random.randn(T, n_maturities) * 0.01 + 0.04) # Positive varainces
    c3 = np.random.randn(T, n_maturities) * 0.01 - 0.05         # Skewness (often negative)
    c4 = np.abs(np.random.randn(T, n_maturities) * 0.005 + 0.02)# Positive kurtosis

    dummy_data = np.hstack([c2, c3, c4])
    df_dummy = pd.DataFrame(dummy_data, columns=col_names)
    
    print("Extracting latent states...")
    v_history, log_lik = extract_latent_states(theta_BZ, df_dummy)
    
    print(f"Final Log-Likelihood: {log_lik:.2f}")
    print(f"v_history dimensions: {v_history.shape} (Expected: {df_dummy.shape[0]}x3)")
    print("Extracted states (last 3 days) [v1, v2, v3]:")
    print(v_history[-3:])