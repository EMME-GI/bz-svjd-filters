import pandas as pd
import numpy as np
import os
import sys

# =====================================================================
# 1. PARAMETERS DEFINITION (Ground Truth)
# =====================================================================
PARAM_KEYS = [
    'rho1', 'vbar1', 'k1', 'sig1', 'mu1',    # Factor 1
    'rho2', 'vbar2', 'k2', 'sig2',           # Factor 2
    'rho3', 'k3', 'mu3',                     # Factor 3
    'c0_p', 'c1_p', 'c2_p', 'lam_p',         # Positive Jumps
    'c1_m', 'c2_m', 'c3_m', 'lam_m'          # Negative Jumps
]

TRUE_PARAMS = np.array([
    -0.98, 0.02, 0.5, 0.1, 0.2,              
    -0.9897, 0.04, 0.3, 0.15,                
    0.06, 50.0, 0.6,                         
    5.0, 245.0, 100.0, 240.0,                
    20.0, 0.1, 160.0, 7.0                    
])

# =====================================================================
# 2. ANALYSIS ENGINE & FORMAT
# =====================================================================
def genera_tabella_da_csv(filename):
    """Reads the raw CSV and builds the Bias and RMSE table."""
    
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found in the working directory.")
        return

    print(f"\nReading data from file: {filename}")
    df = pd.read_csv(filename)
    n_traj = len(df)
    print(f"Found {n_traj} trajectories succesfully saved.\n")

    print("================== METRICS: BIAS & RMSE ==================")
    print(f"{'Parameter':<8} | {'True Val':<10} | {'DNF Bias':<10} | {'DNF RMSE':<10} | {'FO Bias':<10} | {'FO RMSE':<10}")
    print("-" * 75)

    for i, name in enumerate(PARAM_KEYS):
        true_val = TRUE_PARAMS[i]
        
        # Extracting columns for the current parameter
        dnf_vals = df[f'DNF_{name}'].values
        fo_vals = df[f'FO_{name}'].values
        
        # Computing DNF Metrics
        # Bias = True Value - Average Estimation
        bias_dnf = true_val - np.mean(dnf_vals)
        rmse_dnf = np.sqrt(np.mean((dnf_vals - true_val)**2))
        
        # Computing FO metrics
        bias_fo = true_val - np.mean(fo_vals)
        rmse_fo = np.sqrt(np.mean((fo_vals - true_val)**2))
        
        # Printing formatted line
        print(f"{name:<8} | {true_val:<10.4f} | {bias_dnf:<10.4f} | {rmse_dnf:<10.4f} | {bias_fo:<10.4f} | {rmse_fo:<10.4f}")

    print("=" * 75 + "\n")

    # --- LIKELIHOOD ANALYSIS ---
    if 'LLK_DNF' in df.columns and 'LLK_FO' in df.columns:
        llk_dnf_array = df['LLK_DNF'].values
        llk_fo_array = df['LLK_FO'].values
        
        mean_llk_dnf = np.mean(llk_dnf_array)
        se_llk_dnf = np.std(llk_dnf_array) / np.sqrt(n_traj)
        
        mean_llk_fo = np.mean(llk_fo_array)
        se_llk_fo = np.std(llk_fo_array) / np.sqrt(n_traj)
        
        print("\n================== LOG-LIKELIHOOD ANALYSIS ==================")
        print(f"DNF Mean LLK: {mean_llk_dnf:12.4f}  (SE: {se_llk_dnf:.4f})")
        print(f"FO  Mean LLK: {mean_llk_fo:12.4f}  (SE: {se_llk_fo:.4f})")
        print(f"Avg Differece (DNF - FO): {(mean_llk_dnf - mean_llk_fo):.4f}")
    else:
        print("\n Likelihood columns not found in your file (File generated without tracking LLK).")

    print("=" * 75 + "\n")

# =====================================================================
# 3. EXECUTION
# =====================================================================
if __name__ == "__main__":
    NOME_FILE = sys.argv[1] if len(sys.argv) > 1 else "MLE_6traj_10cycles.csv"
    
    genera_tabella_da_csv(NOME_FILE)