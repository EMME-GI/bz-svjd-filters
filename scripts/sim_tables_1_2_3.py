import sys
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))
import kfdnf_fast
from cumulant_engine import AFTCumulantCalculator
from main_calibration import build_full_observation_matrices, build_phi_and_grid, build_phi_for_FO, build_phi_for_pf
from data_generator import generate_high_fidelity_data

def benchmark_trajectory(m, theta_val, deltas_val, matrices, T_days=2520):
    try:
        dt = 1.0 / 252.0
        
        # 1. Generating latent states
        v_history_T = generate_high_fidelity_data(theta_val, N_days=T_days, steps_per_day=1)
        
        # 2. Generating Z_clean
        Z_clean_T = matrices['G0_true'] + matrices['G1_true'] @ v_history_T
        
        # 3. Measurement noise
        empirical_vars = np.var(Z_clean_T, axis=1, keepdims=True)
        noise_std = deltas_val.reshape(12, 1) * np.sqrt(empirical_vars)
        noise_matrix = np.random.randn(12, T_days) * noise_std
        
        Z_noisy = np.asfortranarray((Z_clean_T + noise_matrix).T)
        v_history = np.asfortranarray(v_history_T.T)
        
        Omega = np.zeros((12, 12), order='F')
        np.fill_diagonal(Omega, (deltas_val ** 2) * empirical_vars.flatten())

        risultato = {'Traiettoria': m + 1}

        # --- STATISTICS ---
        risultato['True_Mean_v1'], risultato['True_Mean_v2'], risultato['True_Mean_v3'] = np.mean(v_history, axis=0)
        risultato['True_Std_v1'],  risultato['True_Std_v2'],  risultato['True_Std_v3']  = np.std(v_history, axis=0)
        risultato['True_Skew_v1'], risultato['True_Skew_v2'], risultato['True_Skew_v3'] = skew(v_history, axis=0)
        risultato['True_Kurt_v1'], risultato['True_Kurt_v2'], risultato['True_Kurt_v3'] = kurtosis(v_history, axis=0, fisher=False)
        
        # --- PARTICLE FILTER ---
        # Uses 'pure_Phi' without pre-computed jump expectations
        filt_pf = kfdnf_fast.ParticleFilter()
        t0 = time.time()
        ll_pf = filt_pf.filter_series(
            Z_noisy, matrices['pure_Phi0'], matrices['pure_Phi1'], 
            matrices['G0_flattened'], matrices['G1_true'],
            Omega, theta_val['sig1'], theta_val['sig2'], dt,
            theta_val['c0_m'], theta_val['c1_m'], theta_val['c2_m'], theta_val['c3_m'],
            theta_val['lam_m'], theta_val['mu1'], theta_val['mu3'], theta_val['rho3'],
            theta_val['vbar1'], theta_val['vbar2'], 1e-4,
            10000 # number of PF particles (BZ: 100,000, stratified resampling)
        )
        risultato['Time_PF'] = time.time() - t0
        risultato['LogLik_PF'] = ll_pf
        v_est_pf = filt_pf.get_v_history()
        risultato['PF_RMSE_v1'], risultato['PF_RMSE_v2'], risultato['PF_RMSE_v3'] = np.sqrt(np.mean((v_est_pf - v_history)**2, axis=0))
        risultato['PF_Mean_v1'], risultato['PF_Mean_v2'], risultato['PF_Mean_v3'] = np.mean(v_est_pf, axis=0)
        risultato['PF_Std_v1'],  risultato['PF_Std_v2'],  risultato['PF_Std_v3']  = np.std(v_est_pf, axis=0)
        risultato['PF_Skew_v1'], risultato['PF_Skew_v2'], risultato['PF_Skew_v3'] = skew(v_est_pf, axis=0)
        risultato['PF_Kurt_v1'], risultato['PF_Kurt_v2'], risultato['PF_Kurt_v3'] = kurtosis(v_est_pf, axis=0, fisher=False)
        
        # --- FEUNOU & OKOU ---
        # Uses 'linear_Phi' wich takes into account also the jump expectation
        filt_fo = kfdnf_fast.FeunouOkouKF()
        t0 = time.time()
        ll_fo = filt_fo.filter_series(
            Z_noisy, matrices['linear_Phi0'], matrices['linear_Phi1'], 
            matrices['G0_flattened'], matrices['G1_true'], 
            Omega, theta_val['sig1'], theta_val['sig2'], dt,
            theta_val['c0_m'], theta_val['c1_m'], theta_val['c2_m'], theta_val['c3_m'],
            theta_val['lam_m'], theta_val['mu1'], theta_val['mu3'], theta_val['rho3']
        )
        risultato['Time_FO'] = time.time() - t0
        risultato['LogLik_FO'] = ll_fo
        v_est_fo = filt_fo.get_v_history()
        risultato['FO_RMSE_v1'], risultato['FO_RMSE_v2'], risultato['FO_RMSE_v3'] = np.sqrt(np.mean((v_est_fo - v_history)**2, axis=0))
        risultato['FO_Mean_v1'], risultato['FO_Mean_v2'], risultato['FO_Mean_v3'] = np.mean(v_est_fo, axis=0)
        risultato['FO_Std_v1'],  risultato['FO_Std_v2'],  risultato['FO_Std_v3']  = np.std(v_est_fo, axis=0)
        risultato['FO_Skew_v1'], risultato['FO_Skew_v2'], risultato['FO_Skew_v3'] = skew(v_est_fo, axis=0)
        risultato['FO_Kurt_v1'], risultato['FO_Kurt_v2'], risultato['FO_Kurt_v3'] = kurtosis(v_est_fo, axis=0, fisher=False)

        # --- DISCRETE NONLINEAR FILTER ---
        for K_val in range(1, 11):
            Phi0_K, Phi1_K, probs_K = matrices['dnf_grids'][K_val]
            
            filt_dnf = kfdnf_fast.DiscreteNonlinearFilter()
            
            t0 = time.time()
            ll_dnf = filt_dnf.filter_series(
                Z_noisy, Phi0_K, Phi1_K, matrices['G0_flattened'], matrices['G1_true'], probs_K, 
                Omega, theta_val['sig1'], theta_val['sig2'], dt,
                theta_val['c0_m'], theta_val['c1_m'], theta_val['c2_m'], theta_val['c3_m']
            )
            t_elapsed = time.time() - t0
            
            v_est_dnf_K = filt_dnf.get_v_history()
            risultato[f'DNF_K{K_val}_RMSE_v1'], risultato[f'DNF_K{K_val}_RMSE_v2'], risultato[f'DNF_K{K_val}_RMSE_v3'] = np.sqrt(np.mean((v_est_dnf_K - v_history)**2, axis=0))
            risultato[f'Time_DNF_K{K_val}'] = t_elapsed
            risultato[f'LogLik_DNF_K{K_val}'] = ll_dnf
            
            if K_val == 5:
                risultato['DNF_Mean_v1'], risultato['DNF_Mean_v2'], risultato['DNF_Mean_v3'] = np.mean(v_est_dnf_K, axis=0)
                risultato['DNF_Std_v1'],  risultato['DNF_Std_v2'],  risultato['DNF_Std_v3']  = np.std(v_est_dnf_K, axis=0)
                risultato['DNF_Skew_v1'], risultato['DNF_Skew_v2'], risultato['DNF_Skew_v3'] = skew(v_est_dnf_K, axis=0)
                risultato['DNF_Kurt_v1'], risultato['DNF_Kurt_v2'], risultato['DNF_Kurt_v3'] = kurtosis(v_est_dnf_K, axis=0, fisher=False)
                
        if m % 50 == 0 and m > 0:
            print(f"   -> {m} trajectories completed...")
            
        return risultato
    except Exception as e:
        print(f"Fatal error for trajecotry {m+1}: {e}")
        return None

if __name__ == "__main__":
    # BZ Table 4 parameters
    theta_tab4 = {'k1': 0.5, 'vbar1': 0.02, 'sig1': 0.1, 'rho1': -0.9800, 
                  'k2': 0.3, 'vbar2': 0.04, 'sig2': 0.15, 'rho2': -0.9897, 
                  'k3': 50.0,'mu3': 0.6, 'rho3': 0.06, 'mu1': 0.2, 
                  'lam_m': 7.0, 'lam_p': 240.0, 
                  'c0_m': 0.0, 'c1_m': 20.0, 'c2_m': 0.1, 'c3_m': 160.0, 
                  'c0_p': 5.0, 'c1_p': 245.0, 'c2_p': 100.0, 'c3_p': 0.0, 
                  'eta': 0.0, 'r': 0.0, 'delta': 0.0}
    
    deltas_alti = np.array([0.15]*4 + [0.20]*4 + [0.25]*4)
    N_TRAJECTORIES = 500
    dt_sim = 1.0 / 252.0
    n_cores = 6

    print(f"Simulating Section 4. Allocated cores: {n_cores}. Simulating {N_TRAJECTORIES} Monte-Carlo trajectories.")
    
    t_start_scenario = time.time()
    
    engine_scen = AFTCumulantCalculator(theta_tab4)
    G0_true, G1_true = build_full_observation_matrices(theta_tab4, engine_scen)
    
    # 1. Particle Filter matrices 
    pure_Phi0, pure_Phi1 = build_phi_for_pf(theta_tab4, dt_sim)
    
    # 2. FO matrices 
    linear_Phi0, linear_Phi1 = build_phi_for_FO(theta_tab4, dt_sim)
    
    dnf_grids_precomputed = {}
    for K_val in range(1, 11):
        Phi0_K, Phi1_K, probs_K = build_phi_and_grid(theta_tab4, dt_sim, K=K_val)
        dnf_grids_precomputed[K_val] = (
            np.asfortranarray(Phi0_K),
            np.asfortranarray(Phi1_K),
            np.asfortranarray(probs_K.flatten(order='F'))
        )
        
    matrices = {
        'G0_true': np.asfortranarray(G0_true),
        'G1_true': np.asfortranarray(G1_true),
        'G0_flattened': np.asfortranarray(G0_true.flatten(order='F')),
        'pure_Phi0': np.asfortranarray(pure_Phi0.flatten(order='F')),
        'pure_Phi1': np.asfortranarray(pure_Phi1),
        'linear_Phi0': np.asfortranarray(linear_Phi0.flatten(order='F')),
        'linear_Phi1': np.asfortranarray(linear_Phi1),
        'dnf_grids': dnf_grids_precomputed
    }

    risultati = Parallel(n_jobs=n_cores)(
        delayed(benchmark_trajectory)(m, theta_tab4, deltas_alti, matrices, 2520) 
        for m in range(N_TRAJECTORIES)
    )
    
    record_validi = [r for r in risultati if r is not None]
    df_output = pd.DataFrame(record_validi)
    filename = "Simulation_results.csv"
    df_output.to_csv(filename, index=False)
    
    tempo_scenario = (time.time() - t_start_scenario) / 60
    print(f"Completed. File '{filename}' written in {tempo_scenario:.2f} minutes.")