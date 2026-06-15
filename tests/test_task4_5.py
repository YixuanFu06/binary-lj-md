import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import linregress

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analysis import compute_partial_rdf, compute_warren_cowley, compute_partial_msd
from src.md_engine import SIGMA_REF, ANGSTROM, TAU_REF

def main():
    os.makedirs('data/rdf', exist_ok=True)
    os.makedirs('data/msd', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    xi_values = [0.7, 1.0, 1.3]
    
    wc_means = {}
    D_Kr = {}
    D_Xe = {}
    
    # R_cut for Warren-Cowley: 1.3 * sigma_AB (in reduced units)
    # sigma_AB = (sigma_AA + sigma_BB)/2
    r_cut_wc = 1.3 * (3.636 + 3.924) / 2.0 / SIGMA_REF
    
    fig_rdf, axes_rdf = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    fig_wc, ax_wc = plt.subplots(1, 1, figsize=(8, 5))
    fig_msd, axes_msd = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    N_A = 250
    N_B = 250
    species = np.zeros(N_A + N_B, dtype=int)
    species[N_A:] = 1
    box_L = (500 / 0.8)**(1/3)
    
    for i, xi in enumerate(xi_values):
        print(f"Analyzing xi={xi:.1f} ...")
        traj_file = f'data/trajectories/traj_xi{xi:.1f}.npy'
        traj = np.load(traj_file)
        
        n_frames = len(traj)
        time_tau = np.arange(n_frames) * 50 * 2.0 * 1e-15 / TAU_REF
        
        # --- TASK 4: RDF ---
        print(f"  Computing RDF for xi={xi:.1f} ...")
        r, gAA, gBB, gAB = compute_partial_rdf(traj, species, box_L)
        np.savez(f'data/rdf/rdf_xi{xi:.1f}.npz', r=r, gAA=gAA, gBB=gBB, gAB=gAB)
        
        ax = axes_rdf[i]
        ax.plot(r, gAA, label='Kr-Kr (AA)')
        ax.plot(r, gBB, label='Xe-Xe (BB)')
        ax.plot(r, gAB, label='Kr-Xe (AB)')
        ax.set_ylabel('g(r)')
        ax.set_title(f'RDF for $\\xi={xi:.1f}$')
        ax.legend()
        ax.grid(True)
        
        # Annotate first peaks
        peak_AA_idx = np.argmax(gAA)
        peak_BB_idx = np.argmax(gBB)
        peak_AB_idx = np.argmax(gAB)
        ax.plot(r[peak_AA_idx], gAA[peak_AA_idx], 'x', color='blue')
        ax.plot(r[peak_BB_idx], gBB[peak_BB_idx], 'x', color='orange')
        ax.plot(r[peak_AB_idx], gAB[peak_AB_idx], 'x', color='green')
        
        # --- TASK 4: Warren-Cowley ---
        print(f"  Computing Warren-Cowley parameter for xi={xi:.1f} ...")
        alpha_1 = compute_warren_cowley(traj, species, box_L, r_cut_wc)
        ax_wc.plot(time_tau, alpha_1, label=f'$\\xi={xi:.1f}$')
        
        mean_alpha = np.mean(alpha_1[int(n_frames*0.4):])
        wc_means[xi] = mean_alpha
        
        # --- TASK 5: MSD ---
        print(f"  Computing MSD for xi={xi:.1f} ...")
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
        
    axes_rdf[2].set_xlabel(r'Distance ($\sigma^*$)')
    fig_rdf.tight_layout()
    fig_rdf.savefig('figures/rdf_comparison.png')
    
    ax_wc.set_xlabel(r'Time ($\tau^*$)')
    ax_wc.set_ylabel(r'$\alpha_1$')
    ax_wc.set_title('Warren-Cowley Short-Range Order Parameter')
    ax_wc.legend()
    ax_wc.grid(True)
    fig_wc.tight_layout()
    fig_wc.savefig('figures/warren_cowley_timeseries.png')
    
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
    fig_msd.savefig('figures/msd_diffusion.png')
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    for xi in xi_values:
        print(f"xi = {xi:.1f}:")
        print(f"  <alpha_1> = {wc_means[xi]:.4f}")
        print(f"  D_Kr      = {D_Kr[xi]:.4e} m^2/s")
        print(f"  D_Xe      = {D_Xe[xi]:.4e} m^2/s")
    
    os.makedirs('report', exist_ok=True)
    with open('report/results_summary.md', 'w') as f:
        f.write("# Task 4 & 5 Results Summary\n\n")
        f.write("## Warren-Cowley Short-Range Order Parameter\n")
        for xi in xi_values:
            f.write(f"- $\\xi = {xi:.1f}$: $\\langle \\alpha_1 \\rangle = {wc_means[xi]:.4f}$\n")
        f.write("\n## Diffusion Coefficients\n")
        f.write("| $\\xi$ | $D_{Kr}$ (m²/s) | $D_{Xe}$ (m²/s) |\n")
        f.write("|---|---|---|\n")
        for xi in xi_values:
            f.write(f"| {xi:.1f} | {D_Kr[xi]:.4e} | {D_Xe[xi]:.4e} |\n")

if __name__ == '__main__':
    main()
