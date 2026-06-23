import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.md_engine import run_md
from src.analysis import compute_warren_cowley, compute_structure_factor

def compute_sq0(traj, species, box_L):
    try:
        q_centers, S_tot, S_AA, S_BB, S_AB = compute_structure_factor(
            traj, species, box_L, q_max=4.0, bins=20
        )
        if len(S_AA) > 0:
            # S_AA for first two bins
            return float(np.mean(S_AA[:2]))
        else:
            return 0.0
    except Exception as e:
        print(f"Error computing Sq0: {e}")
        return 0.0

def run_grid(test_mode=False):
    T_values = [80, 100, 120, 140, 160, 180, 200]
    xB_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    if test_mode:
        print("=== TEST MODE: Running only T=100, xB=0.5 ===")
        T_values = [100]
        xB_values = [0.5]
    
    os.makedirs('data/phase_diagram', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    csv_file = 'data/phase_diagram/phase_grid_long.csv'
    if test_mode:
        csv_file = 'data/phase_diagram/phase_grid_test.csv'
    
    results = []
    
    N_total = 300
    rho_star = 0.80
    box_L = (N_total / rho_star)**(1/3)
    
    # Check if we should resume
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        completed_pairs = set(zip(df_existing['T'], df_existing['xB']))
        results = df_existing.to_dict('records')
    else:
        completed_pairs = set()
        with open(csv_file, 'w') as f:
            f.write("T,xB,alpha1,Sq0,phase_label\n")
            
    total_runs = len(T_values) * len(xB_values)
    current_run = 0
            
    for T in T_values:
        for xB in xB_values:
            current_run += 1
            if (T, xB) in completed_pairs and not test_mode:
                print(f"[{current_run}/{total_runs}] Skipping completed point T={T}, xB={xB}")
                continue
                
            print(f"[{current_run}/{total_runs}] Running point T={T}K, xB={xB} ...")
            N_B = int(np.round(N_total * xB))
            N_A = N_total - N_B
            
            # Dynamic time scale: deeper quenches need more time
            n_equil_dyn = 20000 if T <= 140 else 5000
            n_prod_dyn = 100000 if T <= 140 else 20000
            
            # Disable file saving to avoid 63 huge trajectory files filling up disk
            config = dict(
                N_A=N_A, N_B=N_B, rho_star=rho_star,
                T_equil_K=float(T), T_prod_K=float(T), dt_fs=2.0,
                n_equil=n_equil_dyn, n_prod=n_prod_dyn, n_save=100,
                ensemble_prod='NVT', tau_nh_fs=100.0,
                xi=0.7, seed=int(T + xB*100),
                use_torch=False
            )
            
            res = run_md(config)
            traj = res['trajectory']
            species = res['species']
            
            n_frames = len(traj)
            start_idx = int(n_frames * 0.4)
            traj_prod = traj[start_idx:]
            
            # compute warren cowley for all frames in prod and average
            # 1.5 is the first coordination shell
            alpha_list = compute_warren_cowley(traj_prod, species, box_L, r_cut_wc=1.5) 
            mean_alpha1 = float(np.mean(alpha_list))
            
            # compute Sq0
            sq0 = compute_sq0(traj, species, box_L)
            
            # Classify
            if mean_alpha1 > 0.15:
                label = 'PS'
            elif mean_alpha1 < -0.15:
                label = 'ORD'
            else:
                label = 'DIS'
                
            row = {'T': T, 'xB': xB, 'alpha1': mean_alpha1, 'Sq0': sq0, 'phase_label': label}
            results.append(row)
            
            # Save immediately
            with open(csv_file, 'a') as f:
                f.write(f"{T},{xB},{mean_alpha1:.4f},{sq0:.4f},{label}\n")
                
            print(f"  -> alpha1={mean_alpha1:.4f}, Sq0={sq0:.4f}, Label={label}")
            
    print("Grid run complete.")
            
    # Plotting
    df = pd.DataFrame(results)
    plot_phase_diagram(df)

def plot_phase_diagram(df):
    plt.figure(figsize=(8, 6))
    
    colors = {'PS': 'red', 'ORD': 'blue', 'DIS': 'green'}
    for label, color in colors.items():
        subset = df[df['phase_label'] == label]
        if not subset.empty:
            plt.scatter(subset['xB'], subset['T'], c=color, label=label, s=100, edgecolors='k', zorder=5)
            
    # Soft background interpolation
    if len(df) > 10:
        grid_xB, grid_T = np.mgrid[0.1:0.9:100j, 80:200:100j]
        grid_alpha1 = griddata((df['xB'], df['T']), df['alpha1'], (grid_xB, grid_T), method='cubic')
        
        contour = plt.contourf(grid_xB, grid_T, grid_alpha1, levels=20, cmap='coolwarm', alpha=0.4, zorder=1)
        plt.colorbar(contour, label=r'Warren-Cowley $\alpha_1$')
        
        # approximate phase boundary
        plt.contour(grid_xB, grid_T, grid_alpha1, levels=[0.15], colors='red', linewidths=2, linestyles='--', zorder=2)
    
    plt.xlabel(r'Xe Mole Fraction $x_B$', fontsize=14)
    plt.ylabel(r'Temperature $T$ (K)', fontsize=14)
    plt.title(r'T-x Phase Diagram ($\xi=0.7$)', fontsize=16)
    plt.legend()
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('figures/task8_phase_diagram_Tx.png', dpi=300)
    print("Phase diagram saved to figures/task8_phase_diagram_Tx.png")

if __name__ == '__main__':
    run_grid()
