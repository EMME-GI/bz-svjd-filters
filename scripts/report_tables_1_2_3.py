import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# File names
NOMI_FILES = ["Simulation_results.csv"]

for NOME_FILE in NOMI_FILES:
    try:
        df = pd.read_csv(NOME_FILE)
        N_SIM = len(df)
        
        print(f"\n{'='*80}")
        print(f"Analysis and latex generation completed: {NOME_FILE}")
        print(f"{'='*80}")

        # =====================================================================
        # 1. GENERATING FIGURE 1 (RRMSE Plot)
        # =====================================================================
        K_vals = np.arange(1, 11)
        
        # Computing RRMSE
        rrmse_dnf = [np.sqrt(np.mean(((df[f'LogLik_DNF_K{k}'] - df['LogLik_PF']) / df['LogLik_PF'])**2)) for k in K_vals]
        rrmse_fo_val = np.sqrt(np.mean(((df['LogLik_FO'] - df['LogLik_PF']) / df['LogLik_PF'])**2))
        rrmse_fo = [rrmse_fo_val] * len(K_vals)

        plt.figure(figsize=(10, 5))
        plt.plot(K_vals, rrmse_dnf, 'o', markerfacecolor='none', markeredgecolor='dodgerblue', markersize=7, label='KF-DNF')
        plt.plot(K_vals, rrmse_fo, 'x', color='navy', markersize=7, label='KF')
        
        plt.xlabel('Grid size $K$', fontsize=12)
        plt.ylabel('Relative root-mean-square error', fontsize=12)
        plt.xticks(K_vals)
        plt.legend(loc='upper right', frameon=True, edgecolor='black')
        
        # Saving
        fig_name = f"Figure_1_RRMSE_{NOME_FILE.replace('.csv', '')}"
        plt.savefig(f"{fig_name}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(f"{fig_name}.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated and saved the figures: {fig_name}.pdf/.png")

        # =====================================================================
        # 2. GENERATING LATEX FILE (Tables 1, 2, 3)
        # =====================================================================
        latex_code = ""

        # --- TABLE 1: RRMSE ---
        latex_code += "\\begin{table}[htbp]\n\\centering\n"
        latex_code += "\\caption{Comparison of the filtered volatility factors to their simulated values.}\n"
        latex_code += "\\label{tab:rmse_comparison}\n"
        latex_code += "\\begin{tabular}{l" + "c"*10 + "}\n\\toprule\n"
        
        factors_map = {1: 'First volatility factor', 2: 'Second volatility factor', 3: 'Third volatility factor'}
        
        for v_idx, factor_name in factors_map.items():
            latex_code += f"\\multicolumn{{11}}{{l}}{{\\textbf{{Panel {'ABC'[v_idx-1]}: {factor_name}}}}} \\\\\n"
            latex_code += "Grid size $K$ & " + " & ".join([str(k) for k in K_vals]) + " \\\\\n\\midrule\n"
            
            pf_rmse_mean = df[f'PF_RMSE_v{v_idx}'].mean()
            
            # RMSE Row (PF, absolute)
            latex_code += "RMSE & " + " & ".join([f"{pf_rmse_mean:.4f}"] * 10) + " \\\\\n"
            
            # KF-DNF Row
            dnf_ratios = [df[f'DNF_K{k}_RMSE_v{v_idx}'].mean() / pf_rmse_mean for k in K_vals]
            latex_code += "KF-DNF & " + " & ".join([f"{r:.4f}" for r in dnf_ratios]) + " \\\\\n"
            
            # KF/FO Row
            fo_ratio = df[f'FO_RMSE_v{v_idx}'].mean() / pf_rmse_mean
            latex_code += "KF & " + " & ".join([f"{fo_ratio:.4f}"] * 10) + " \\\\\n"
            
            if v_idx < 3:
                latex_code += "\\midrule\n"
            else:
                latex_code += "\\bottomrule\n"
                
        latex_code += "\\end{tabular}\n\\end{table}\n\n"

        # --- TABLE 2: Summary Statistics ---
        latex_code += "\\begin{table}[htbp]\n\\centering\n"
        latex_code += "\\caption{Summary statistics of the true and filtered states for the PF, KF-DNF, and KF methods.}\n"
        latex_code += "\\label{tab:summary_stats}\n"
        latex_code += "\\begin{tabular}{lcccc}\n\\toprule\n"
        
        models = [('True', 'True'), ('PF', 'PF'), ('DNF', 'KF-DNF'), ('FO', 'KF')]
        
        for v_idx, factor_name in factors_map.items():
            latex_code += f"\\multicolumn{{5}}{{l}}{{\\textbf{{Panel {'ABC'[v_idx-1]}: {factor_name}}}}} \\\\\n"
            
            # Factor 3 is multiplied by 10^4 in BZ
            if v_idx == 3:
                latex_code += "& Mean ($\\times 10^4$) & Standard deviation & Skewness & Kurtosis \\\\\n\\midrule\n"
            else:
                latex_code += "& Mean & Standard deviation & Skewness & Kurtosis \\\\\n\\midrule\n"
            
            for prefix, name in models:
                m_mean = df[f'{prefix}_Mean_v{v_idx}'].mean()
                m_err  = df[f'{prefix}_Mean_v{v_idx}'].std()
                s_mean = df[f'{prefix}_Std_v{v_idx}'].mean()
                s_err  = df[f'{prefix}_Std_v{v_idx}'].std()
                sk_mean = df[f'{prefix}_Skew_v{v_idx}'].mean()
                sk_err  = df[f'{prefix}_Skew_v{v_idx}'].std()
                ku_mean = df[f'{prefix}_Kurt_v{v_idx}'].mean()
                ku_err  = df[f'{prefix}_Kurt_v{v_idx}'].std()
                
                if v_idx == 3:
                    m_mean *= 1e4
                    m_err *= 1e4

                latex_code += f"{name} & {m_mean:.4f} ({m_err:.4f}) & {s_mean:.4f} ({s_err:.4f}) & {sk_mean:.4f} ({sk_err:.4f}) & {ku_mean:.4f} ({ku_err:.4f}) \\\\\n"
                
            if v_idx < 3:
                latex_code += "\\midrule\n"
            else:
                latex_code += "\\bottomrule\n"
                
        latex_code += "\\end{tabular}\n\\end{table}\n\n"

        # --- TABLE 3: Computation times ---
        # Times in seconds
        latex_code += "\\begin{table}[htbp]\n\\centering\n"
        latex_code += "\\caption{Comparison of computation times for the PF, KF-DNF, and KF methods (in seconds).}\n"
        latex_code += "\\label{tab:comp_times}\n"
        latex_code += "\\begin{tabular}{l" + "c"*10 + "}\n\\toprule\n"
        latex_code += "Grid size $K$ & " + " & ".join([str(k) for k in K_vals]) + " \\\\\n\\midrule\n"
        
        # PF row
        pf_t_mean = df['Time_PF'].mean()
        pf_t_err = df['Time_PF'].std()
        pf_str = f"{pf_t_mean:.2f} ({pf_t_err:.2f})"
        latex_code += "PF & \\multicolumn{10}{c}{" + pf_str + "} \\\\\n"
        
        # KF-DNF row
        dnf_times_str = []
        for k in K_vals:
            t_m = df[f'Time_DNF_K{k}'].mean()
            t_s = df[f'Time_DNF_K{k}'].std()
            dnf_times_str.append(f"{t_m:.2f} ({t_s:.2f})")
        latex_code += "KF-DNF & " + " & ".join(dnf_times_str) + " \\\\\n"
        
        # FO/KF row
        fo_t_mean = df['Time_FO'].mean()
        fo_t_err = df['Time_FO'].std()
        fo_str = f"{fo_t_mean:.2f} ({fo_t_err:.2f})"
        latex_code += "KF & \\multicolumn{10}{c}{" + fo_str + "} \\\\\n\\bottomrule\n"
        latex_code += "\\end{tabular}\n\\end{table}\n"

        # Saving file
        tex_filename = f"tables_output_{NOME_FILE.replace('.csv', '.tex')}"
        with open(tex_filename, "w") as text_file:
            text_file.write(latex_code)
        
        print(f"Generated and saved LaTeX code: {tex_filename}")

        # ==================================================================
        # PRINTING PHASE 
        # ==================================================================

        # --- TABLe 1: RMSE ---
        print("\n TABLE 1: RRMSE (Ratio Filter / Particle Filter) varying K")
        print("-" * 80)
        for v_idx, v_name in [(1, 'v1 (Fast)'), (2, 'v2 (Slow)'), (3, 'v3 (Jumps)')]:
            print(f"\n--- Factor: {v_name} ---")
            header = f"{'Filter':<15} | " + " | ".join([f"K={k:<5}" for k in range(1, 11)])
            print(header)
            print("-" * len(header))
            
            pf_rmse_mean = df[f'PF_RMSE_v{v_idx}'].mean()

            pf_rmse_str = f"{pf_rmse_mean:.4f}"
            print(f"{'RMSE':<15} | " + " | ".join([f"{pf_rmse_str:<5}"] * 10))
            
            dnf_ratios = [f"{df[f'DNF_K{k}_RMSE_v{v_idx}'].mean() / pf_rmse_mean:.4f}" for k in range(1, 11)]
            print(f"{'KF-DNF':<15} | " + " | ".join(dnf_ratios))
            
            fo_ratio = f"{df[f'FO_RMSE_v{v_idx}'].mean() / pf_rmse_mean:.4f}"
            print(f"{'FO (KF paper)':<15} | " + " | ".join([f"{fo_ratio:<5}"] * 10))

        # --- TABLE 2: MOMENTS & UNCERTANTIES ---
        print("\n\n TABLE 2: Statistical Moments (Average on 500 simulations) with associated uncertanties")
        print("-" * 80)
        models = [('True', 'Ground Truth'), ('PF', 'Particle Filter'), ('FO', 'FO (KF paper)'), ('DNF', 'KF-DNF (K=5)')]

        for v_idx in [1, 2, 3]:
            print(f"\n--- Factor v{v_idx} ---")
            header = f"{'Model':<18} | {'Average':<15} | {'Std.Dev':<15} | {'Skewness':<15} | {'Kurtosis':<15}"
            print(header)
            print("-" * len(header))
            
            for prefix, name in models:
                m_mean = df[f'{prefix}_Mean_v{v_idx}'].mean()
                m_err  = df[f'{prefix}_Mean_v{v_idx}'].std() 
                
                s_mean = df[f'{prefix}_Std_v{v_idx}'].mean()
                s_err  = df[f'{prefix}_Std_v{v_idx}'].std() 
                
                sk_mean = df[f'{prefix}_Skew_v{v_idx}'].mean()
                sk_err  = df[f'{prefix}_Skew_v{v_idx}'].std() 
                
                ku_mean = df[f'{prefix}_Kurt_v{v_idx}'].mean()
                ku_err  = df[f'{prefix}_Kurt_v{v_idx}'].std() 
                
                if v_idx == 3:
                    m_mean*=1e4
                    m_err*=1e4

                print(f"{name:<18} | {m_mean:.4f}({m_err:.4f}) | {s_mean:.4f}({s_err:.4f}) | {sk_mean:.4f}({sk_err:.4f}) | {ku_mean:.4f}({ku_err:.4f})")

        # --- TABELLA 3: TEMPI ---
        print("\n\n TABLE 3: Average computational times (seconds) varying K")
        print("-" * 80)
        header = f"{'Filter':<15} | " + " | ".join([f"K={k:<9}" for k in range(1, 11)])
        print(header)
        print("-" * len(header))

        dnf_times = [f"{df[f'Time_DNF_K{k}'].mean():.2f}({df[f'Time_DNF_K{k}'].std():.2f})" for k in range(1, 11)]
        print(f"{'KF-DNF':<15} | " + " | ".join(dnf_times))

        for prefix, name in [('PF', 'Particle Filter'), ('FO', 'FO (KF paper)')]:
            t_mean = df[f'Time_{prefix}'].mean()
            t_err  = df[f'Time_{prefix}'].std() 
            valore_costante = f"{t_mean:.2f}({t_err:.2f})"
            print(f"{name:<15} | " + " | ".join([f"{valore_costante:<9}"] * 10))
        print("\n" + "="*80)

    except FileNotFoundError:
        print(f"Error: file '{NOME_FILE}' not found in the folder.")