import sys
import os

# FORCES SINGLE-THREADING AT LEVEL C/C++ E BLAS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import scipy.optimize as opt

# Project modules live in ../src (the compiled kfdnf_fast module is built there)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))
import kfdnf_fast
from cumulant_engine import AFTCumulantCalculator
from main_calibration import build_full_observation_matrices, build_phi_and_grid, build_phi_for_FO
from data_generator import generate_high_fidelity_data

def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m}m {s:.1f}s"

# =====================================================================
# 1. SETUP OF 20 PARAMETERS AND BOUNDS (Tabella 4 di BZ 2025)
# =====================================================================
PARAM_KEYS = [
    'rho1', 'vbar1', 'k1', 'sig1', 'mu1',    # Factor 1
    'rho2', 'vbar2', 'k2', 'sig2',           # Factor 2
    'rho3', 'k3', 'mu3',                     # Factor 3
    'c0_p', 'c1_p', 'c2_p', 'lam_p',         # Positive jumps
    'c1_m', 'c2_m', 'c3_m', 'lam_m'          # Negative jumps
]

TRUE_PARAMS = np.array([
    -0.98, 0.02, 0.5, 0.1, 0.2,              # Factor 1
    -0.9897, 0.04, 0.3, 0.15,                # Factor 2
    0.06, 50.0, 0.6,                         # Factor 3
    5.0, 245.0, 100.0, 240.0,                # Positive jumps
    20.0, 0.1, 160.0, 7.0                    # Negative jumps
])

BOUNDS = [
    # --- PROCESS 1 (Fast Component) ---
    (-0.999, -0.001),    # rho1 (Correlation)
    (0.001, 0.20),      # vbar1 (Avg volatility) - Max 20%
    (0.01, 5.0),        # k1 (Slow mean reversion)
    (0.01, 0.50),       # sig1 (Vol-of-vol)
    (0.001, 2.0),        # mu1 (Drift)
    
    # --- PROCESS 2 (Continuous Variance) ---
    (-0.999, -0.001),    # rho2
    (0.001, 0.20),      # vbar2
    (0.01, 5.0),        # k2
    (0.01, 0.50),       # sig2
    
    # --- PROCESS 3 (Jumps) ---
    (0.001, 0.999),    # rho3
    (10.0, 100.0),      # k3 (Ultra-fast Mean reversion) -> TRUE = 50
    (0.001, 2.0),        # mu3
    
    # --- POSITIVE JUMPS (p) ---
    (0.1, 20.0),        # c0_p -> TRUE = 5
    (50.0, 500.0),      # c1_p -> TRUE = 245
    (10.0, 300.0),      # c2_p -> TRUE = 100
    (10.0, 500.0),      # lam_p (Frequency) -> TRUE = 240
    
    # --- NEGATIVE JUMPS (m) ---
    (1.0, 80.0),        # c1_m -> TRUE = 20
    (0.01, 2.0),        # c2_m -> TRUE = 0.1
    (20.0, 400.0),      # c3_m -> TRUE = 160
    (1.0, 50.0)         # lam_m (Frequency) -> TRUE = 7
]

# Global lock definition
def init_worker(l):
    global write_lock
    write_lock = l

# =====================================================================
# 2. OBJECTIVE FUNCTION (Negative Log-Likelihood)
# =====================================================================
def objective_function(x, base_theta, Z_series, Omega, filter_type, dt):

    # 1. Parameters Update
    theta_trial = base_theta.copy()
    
    # Dynamical assignement of all 20 parameters
    for i, key in enumerate(PARAM_KEYS):
        theta_trial[key] = x[i]
    
    try:
        # 2. Building matrices of the affine model
        engine = AFTCumulantCalculator(theta_trial)
        G0_true, G1_true = build_full_observation_matrices(theta_trial, engine)
        
        # --- 1. SANITY CHECK: OBSERVATION MATRICES AND SERIES ---
        # Verify that we have no mismatch in the array dimensions or NaN
        if G1_true.shape != (12, 3) or G0_true.size != 12:
            return 1e9
        if np.any(np.isnan(G0_true)) or np.any(np.isnan(G1_true)):
            return 1e9
        # -----------------------------------------------------------
        
        # 3. Execute the filter
        if filter_type == 'FO':
            base_Phi0, base_Phi1 = build_phi_for_FO(theta_trial, dt)
            
            # --- 2. SANITY CHECK: FO MATRICES ---
            # Phi0 must be a vector (3,), Phi1 a matrix (3,3)
            if base_Phi1.shape != (3, 3) or np.any(np.isnan(base_Phi1)) or np.any(np.isnan(base_Phi0)):
                return 1e9
            # ---------------------------------------

            filt = kfdnf_fast.FeunouOkouKF()
            
            llk = filt.filter_series(
                Z_series, 
                np.asfortranarray(base_Phi0), 
                np.asfortranarray(base_Phi1), 
                np.asfortranarray(G0_true.flatten(order='F')), 
                np.asfortranarray(G1_true), 
                Omega, 
                theta_trial['sig1'], theta_trial['sig2'], dt,
                theta_trial['c0_m'], theta_trial['c1_m'], theta_trial['c2_m'], theta_trial['c3_m'],
                theta_trial['lam_m'], theta_trial['mu1'], theta_trial['mu3'], theta_trial['rho3']
            )
            
        elif filter_type == 'DNF':
            Phi0_K, Phi1_K, probs_K = build_phi_and_grid(theta_trial, dt, K=5)
            
            # --- 3. SANITY CHECK: DNF GRIDS ---
            if Phi1_K.shape[0] != 3 or Phi1_K.shape[1] != 3 or np.any(np.isnan(Phi1_K)):
                return 1e9
            if np.any(np.isnan(probs_K)):
                return 1e9
            # ---------------------------------------
            
            filt_dnf = kfdnf_fast.DiscreteNonlinearFilter()
            
            llk = filt_dnf.filter_series(
                Z_series, 
                np.asfortranarray(Phi0_K), 
                np.asfortranarray(Phi1_K), 
                np.asfortranarray(G0_true.flatten(order='F')), 
                np.asfortranarray(G1_true), 
                np.asfortranarray(probs_K.flatten(order='F')), 
                Omega, 
                theta_trial['sig1'], theta_trial['sig2'], dt,
                theta_trial['c0_m'], theta_trial['c1_m'], theta_trial['c2_m'], theta_trial['c3_m']
            )
        else:
            raise ValueError(f"Filter {filter_type} is unknown.")
        
        # If the C++ filters returns NaN (es. degenerate matrices), gives a big penalty 
        if np.isnan(llk) or np.isinf(llk):
            return 1e9
            
        # Returns negative log-likelihood 
        return -llk 

    except Exception:
        # Captures numerical crashes from cumulant_engine and the C++ filters
        return 1e9

# =====================================================================
# 3. HYBRID OPTIMIZATION ENGINE (NM -> GRADIENT_ALGORITHM AND LOOP)
# =====================================================================

def objective_wrapper_normalized(x_norm, base_theta, Z_series, Omega, filter_type, dt, lower_bounds, upper_bounds):
    penalty = 0.0
    # Soft penalty if guess goes over the bounds
    if np.any(x_norm < 0.0):
        penalty += np.sum(1e6 * (x_norm[x_norm < 0.0])**2)
    if np.any(x_norm > 1.0):
        penalty += np.sum(1e6 * (x_norm[x_norm > 1.0] - 1.0)**2)

    x_norm_clipped = np.clip(x_norm, 1e-6, 1.0 - 1e-6)
    x_phys = x_norm_clipped * (upper_bounds - lower_bounds) + lower_bounds
    
    # Here the C++ engine is called: if Eigen crashes, nll will be 1e9
    nll = objective_function(x_phys, base_theta, Z_series, Omega, filter_type, dt)
    
    # SUBSTITUTING THE FLAT WALL WITH A QUADRATIC FUNNEL
    # Using penalty >= 1e8 to avoid fluttuations of precision on the float
    if np.isnan(nll) or np.isinf(nll) or nll >= 1e8:
        # Computing the distance from the center of the normalized space (0.5)
        dist_from_center = np.sum((x_norm_clipped - 0.5)**2)
        
        # 1e5 is a NLL sufficiently "bad" to be discarded
        # but the quadratic term generates a non-vanishing derivative that "pushes"
        # the solver to the center of the parameter space
        return 1e5 + (1e4 * dist_from_center) + penalty

    return nll + penalty


def hybrid_optimization(base_theta, Z_series, Omega, filter_type, true_params, dt, n_cycles=3):
    lower_bounds = np.array([b[0] for b in BOUNDS])
    upper_bounds = np.array([b[1] for b in BOUNDS])
    
    N_DIM = len(true_params)
    normalized_bounds = [(0.0, 1.0) for _ in range(N_DIM)]
    
    # 1. Normalization of the true parameter space
    true_norm = (true_params - lower_bounds) / (upper_bounds - lower_bounds)

    # 2. NOISE
    # Variance 0.01 -> Std = 0.1
    noise = np.random.normal(loc=0.0, scale=0.1, size=N_DIM)
    current_x0_norm = true_norm + noise
    
    # Checking that the initial noise in mathematically meaningfull
    current_x0_norm = np.clip(current_x0_norm, 1e-6, 1.0 - 1e-6)

    # 3. ITERATIVE CHAIN (NM -> GRADIENT_ALGORITHM -> NM -> GRADIENT_ALGORITHM...)
    for _ in range(n_cycles):
        
        # FASE 1: Nelder-Mead (fminsearch) starting from current_x0_norm
        res_nm = opt.minimize(
            objective_wrapper_normalized, current_x0_norm, 
            args=(base_theta, Z_series, Omega, filter_type, dt, lower_bounds, upper_bounds), 
            method='Nelder-Mead',
            options={
                'maxiter': 150,       
                'maxfev': 600,        
                'fatol': 1e-6,        
                'xatol': 1e-6,        
                'adaptive': True
            }
        )
        
        # FASE 2: trust-constr
        res_bfgs = opt.minimize(
            objective_wrapper_normalized, res_nm.x, 
            args=(base_theta, Z_series, Omega, filter_type, dt, lower_bounds, upper_bounds), 
            method='trust-constr',  # <-- open-source equivalent of fmincon
            bounds=normalized_bounds,
            options={'maxiter': 50, 
                     'xtol': 1e-5, 
                     'barrier_tol': 1e-5}
        )
        
        # "using each time the optimized values from the previous iteration as the new initial values"
        current_x0_norm = res_bfgs.x

    best_params_phys = current_x0_norm * (upper_bounds - lower_bounds) + lower_bounds
    
    # Computing the log-likelihood
    final_llk = objective_function(best_params_phys, base_theta, Z_series, Omega, filter_type, dt)
        
    return best_params_phys, -final_llk, int(res_bfgs.success) # Returns positive likelihood

# =====================================================================
# 3.5 WORKER FOR PARALLELIZATION
# =====================================================================
def process_single_trajectory(n, theta_base, deltas, dt, N_days, n_cycles, filename):
    """Execute a single full Monte Carlo simulation (Data Gen + FO + DNF)"""
    # WORKER PRNG REINIZIALIZATION
    # Combining time, PID and trajectory number to obtain a unique seed 
    worker_seed = (int(time.time() * 1000) + os.getpid() + n) % (2**32 - 1)
    np.random.seed(worker_seed)

    try:
        start_t = time.time()
        
        # 1. Generating true data
        v_daily = generate_high_fidelity_data(theta_base, N_days=N_days, steps_per_day=10)
        engine_true = AFTCumulantCalculator(theta_base)
        Gamma0_true, Gamma1_true = build_full_observation_matrices(theta_base, engine_true)
        
        Cum_clean = (Gamma0_true.reshape(12, 1) + Gamma1_true @ v_daily).T

        # 2. Omega matrix and noise
        var_cum_empirica = np.var(Cum_clean, axis=0)  
        omega_diag = var_cum_empirica * (deltas**2)    
        Omega = np.asfortranarray(np.diag(omega_diag))
        
        noise = np.random.normal(0, np.sqrt(omega_diag), size=(N_days, 12))
        Z_series = np.asfortranarray(Cum_clean + noise)
        
        # 3. Optimizationn of DNF and FO
        p_dnf, llk_dnf, succ_dnf = hybrid_optimization(theta_base, Z_series, Omega, 'DNF', TRUE_PARAMS, dt, n_cycles)
        p_fo, llk_fo, succ_fo = hybrid_optimization(theta_base, Z_series, Omega, 'FO', TRUE_PARAMS, dt, n_cycles)
        
        # --- I/O CHECKPOINTING ---
        record_str = f"{n+1}"
        for i in range(len(PARAM_KEYS)):
            record_str += f",{p_dnf[i]},{p_fo[i]}"
            
        # Enqueing the 2 Likelihoods in the record and endl
        record_str += f",{llk_dnf},{llk_fo},{succ_dnf},{succ_fo}\n"

        with write_lock:
            with open(filename, "a") as f:
                f.write(record_str)
            
        elapsed = time.time() - start_t
        print(f"Trajectory {n+1} completed in {elapsed:.1f}s and saved on disk.")
        
        return {
            'trajectory_id': n,
            'p_dnf': p_dnf,
            'p_fo': p_fo,
            'llk_dnf': llk_dnf,
            'llk_fo': llk_fo
        }
    except Exception as e:
        print(f"Error on Trajectory {n+1}: {e}")
        return None

# =====================================================================
# 4. MAIN LOOP: SIMULATION AND EVALUATION
# =====================================================================
if __name__ == "__main__":
    # 1. Base values
    theta_base = {'k1': 0.5, 'vbar1': 0.02, 'sig1': 0.1, 'rho1': -0.9800,
                 'k2': 0.3, 'vbar2': 0.04, 'sig2': 0.15, 'rho2': -0.9897, 
                 'k3': 50.0,'mu3': 0.6, 'rho3': 0.06, 
                 'mu1': 0.2, 'lam_m': 7.0, 'lam_p': 240.0, 
                 'c0_m': 0.0, 'c1_m': 20.0, 'c2_m': 0.1, 'c3_m': 160.0, 
                 'c0_p': 5.0, 'c1_p': 245.0, 'c2_p': 100.0, 'c3_p': 0.0, 
                 'eta': 0.0, 'r': 0.0, 'delta': 0.0}

    deltas = np.array([0.15]*4 + [0.20]*4 + [0.25]*4)

    N_TRAJECTORIES = 250 # original value = 250
    N_CYCLES = 50        # original value = 50
    N_DAYS = 2520
    dt = 1.0 / 252.0
    
    # Automatic core number detection (leave 2 free core to avoid blocking the OS)
    num_cores = max(1, mp.cpu_count() - 2)
    
    filename = f"MLE_{N_TRAJECTORIES}traj_{N_CYCLES}cycles.csv"

    # CHECKPOINTING LOGIC
    
    completed_trajectories = 0
    
    if os.path.exists(filename):
        print(f"Found existing saved file: {filename}")
        # Reading CSV file to understand wich is the last record
        df_esistente = pd.read_csv(filename)
        completed_trajectories = len(df_esistente)
        print(f"Restoring: {completed_trajectories} trajectory already simulated.")
    else:
        # If the file doesn't exist, we create it
        print(f"Creating new saving file: {filename}")
        with open(filename, 'w') as f:
            header_cols = ['Trajectory']
            for key in PARAM_KEYS:
                header_cols.extend([f'DNF_{key}', f'FO_{key}'])
            header_cols.extend(['LLK_DNF', 'LLK_FO', 'DNF_success', 'FO_success'])
            f.write(",".join(header_cols) + "\n")

    trajectories_left = N_TRAJECTORIES - completed_trajectories

    if trajectories_left <= 0:
        print("No trajectories left. Exiting.")
        sys.exit(0)

    print(f"\nSTARTING MLE SIMULATION - FULL 20 PARAMETERS")
    print(f"To simulate: {trajectories_left} tajectories on a total of {N_TRAJECTORIES} total trajectories.")
    print(f"Parallel computing: {num_cores} Cores allocated\n")
    
    global_start_time = time.time()

    m = mp.Manager()
    lock = m.Lock()

    args_list = [(n, theta_base, deltas, dt, N_DAYS, N_CYCLES, filename) for n in range(completed_trajectories, N_TRAJECTORIES)]

    # Parallel execution with Lock send to the init
    with mp.Pool(processes=num_cores, initializer=init_worker, initargs=(lock,), maxtasksperchild=1) as pool:
        results = pool.starmap(process_single_trajectory, args_list)

    # Filtering failures
    valid_results = [r for r in results if r is not None]
    
    if valid_results:
        results_dnf = np.array([r['p_dnf'] for r in valid_results])
        results_fo = np.array([r['p_fo'] for r in valid_results])

        print("\n================== METRICS: BIAS E RMSE ==================")
        print(f"{'Parameter':<8} | {'True Val':<10} | {'DNF Bias':<10} | {'DNF RMSE':<10} | {'FO Bias':<10} | {'FO RMSE':<10}")
        print("-" * 75)
        
        for i, name in enumerate(PARAM_KEYS):
            true_val = TRUE_PARAMS[i]
            
            bias_dnf = true_val - np.mean(results_dnf[:, i])
            rmse_dnf = np.sqrt(np.mean((results_dnf[:, i] - true_val)**2))
            
            bias_fo = true_val - np.mean(results_fo[:, i])
            rmse_fo = np.sqrt(np.mean((results_fo[:, i] - true_val)**2))
            
            print(f"{name:<8} | {true_val:<10.4f} | {bias_dnf:<10.4f} | {rmse_dnf:<10.4f} | {bias_fo:<10.4f} | {rmse_fo:<10.4f}")

        # --- LIKELIHOOD ANALYSIS ---
        llk_dnf_array = np.array([r['llk_dnf'] for r in valid_results])
        llk_fo_array = np.array([r['llk_fo'] for r in valid_results])
            
        mean_llk_dnf = np.mean(llk_dnf_array)
        se_llk_dnf = np.std(llk_dnf_array) / np.sqrt(len(valid_results))
            
        mean_llk_fo = np.mean(llk_fo_array)
        se_llk_fo = np.std(llk_fo_array) / np.sqrt(len(valid_results))
            
        print("\n================== LOG-LIKELIHOOD ANALYSIS ==================")
        print(f"DNF Mean LLK: {mean_llk_dnf:12.4f}  (SE: {se_llk_dnf:.4f})")
        print(f"FO  Mean LLK: {mean_llk_fo:12.4f}  (SE: {se_llk_fo:.4f})")
        print(f"Avg Difference (DNF - FO): {(mean_llk_dnf - mean_llk_fo):.4f}")
        print("==================================================================\n") 

    total_time = time.time() - global_start_time
    print("\n" + "="*75)
    print(f"Totale Execution Time: {format_time(total_time)} on {len(valid_results)} trajectories.")
    print("="*75 + "\n")
