import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import linregress

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analysis import compute_partial_msd
from src.md_engine import SIGMA_REF, ANGSTROM, TAU_REF

def main():
    os.makedirs('data/msd', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    print("=" * 60)
    print("TASK 5: DYNAMICAL ANALYSIS (MSD & Diffusion)")
    print("=" * 60)
    
    xi_values = [0.7, 1.0, 1.3]
    D_Kr = {}
    D_Xe = {}
    
    fig_msd, axes_msd = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    box_L = (500 / 0.8)**(1/3)
    
    for i, xi in enumerate(xi_values):
        print(f"\nAnalyzing xi={xi:.1f} ...")
        traj_file = f'data/trajectories/traj_xi{xi:.1f}.npy'
        species_file = f'data/trajectories/species_xi{xi:.1f}.npy'
        if not os.path.exists(traj_file) or not os.path.exists(species_file):
            print(f"  Warning: {traj_file} or {species_file} not found. Skipping.")
            continue
            
        traj = np.load(traj_file)
        species = np.load(species_file)
        n_frames = len(traj)
        time_tau = np.arange(n_frames) * 50 * 2.0 * 1e-15 / TAU_REF
        
        # --- MSD ---
        print(f"  Computing MSD...")
        msd_A, msd_B = compute_partial_msd(traj, species, box_L)
        np.savez(f'data/msd/msd_xi{xi:.1f}.npz', time=time_tau, msd_A=msd_A, msd_B=msd_B)
        
        half = len(time_tau) // 2
        t_fit = time_tau[half:]
        
        res_A = linregress(t_fit, msd_A[half:])
        res_B = linregress(t_fit, msd_B[half:])
        
        D_A_star = res_A.slope / 6.0
        D_B_star = res_B.slope / 6.0
        
        conversion = (SIGMA_REF * ANGSTROM)**2 / TAU_REF
        D_Kr[xi] = D_A_star * conversion
        D_Xe[xi] = D_B_star * conversion
        
        axes_msd[0].plot(time_tau, msd_A, label=f'$\\xi={xi:.1f}$')
        axes_msd[1].plot(time_tau, msd_B, label=f'$\\xi={xi:.1f}$')
        
        # Mark linear fit region on MSD plot
        axes_msd[0].plot(t_fit, res_A.intercept + res_A.slope * t_fit, 'k--', alpha=0.5)
        axes_msd[1].plot(t_fit, res_B.intercept + res_B.slope * t_fit, 'k--', alpha=0.5)
        
    axes_msd[0].set_ylabel(r'MSD$_{Kr}$ ($(\sigma^*)^2$)')
    axes_msd[0].set_title('Kr Mean Square Displacement')
    axes_msd[0].legend()
    axes_msd[0].grid(True)
    
    axes_msd[1].set_ylabel(r'MSD$_{Xe}$ ($(\sigma^*)^2$)')
    axes_msd[1].set_xlabel(r'Time ($\tau^*$)')
    axes_msd[1].set_title('Xe Mean Square Displacement')
    axes_msd[1].legend()
    axes_msd[1].grid(True)
    fig_msd.tight_layout()
    fig_msd.savefig('figures/task5_msd_diffusion.png')
    
    print("\n" + "="*50)
    print("TASK 5 RESULTS SUMMARY")
    print("="*50)
    for xi in xi_values:
        if xi in D_Kr:
            print(f"xi = {xi:.1f}:")
            print(f"  D_Kr = {D_Kr[xi]:.4e} m^2/s")
            print(f"  D_Xe = {D_Xe[xi]:.4e} m^2/s")
    
    os.makedirs('report', exist_ok=True)
    # Append to report
    with open('report/results_summary.md', 'a') as f:
        f.write("\n## Task 5: Diffusion Coefficients\n")
        f.write("| $\\xi$ | $D_{Kr}$ (m²/s) | $D_{Xe}$ (m²/s) |\n")
        f.write("|---|---|---|\n")
        for xi in xi_values:
            if xi in D_Kr:
                f.write(f"| {xi:.1f} | {D_Kr[xi]:.4e} | {D_Xe[xi]:.4e} |\n")

if __name__ == '__main__':
    main()
