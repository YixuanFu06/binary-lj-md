import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analysis import compute_partial_rdf, compute_warren_cowley
from src.md_engine import SIGMA_REF, TAU_REF

def main():
    os.makedirs('data/rdf', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    print("=" * 60)
    print("TASK 4: STRUCTURAL ANALYSIS (RDF & Warren-Cowley)")
    print("=" * 60)
    
    xi_values = [0.7, 1.0, 1.3]
    wc_means = {}
    
    # R_cut for Warren-Cowley: 1.3 * sigma_AB (in reduced units)
    r_cut_wc = 1.3 * (3.636 + 3.924) / 2.0 / SIGMA_REF
    
    fig_rdf, axes_rdf = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    fig_wc, ax_wc = plt.subplots(1, 1, figsize=(8, 5))
    
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
        
        # --- RDF ---
        print(f"  Computing RDF...")
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
        
        # --- Warren-Cowley ---
        print(f"  Computing Warren-Cowley parameter...")
        alpha_1 = compute_warren_cowley(traj, species, box_L, r_cut_wc)
        ax_wc.plot(time_tau, alpha_1, label=f'$\\xi={xi:.1f}$')
        
        mean_alpha = np.mean(alpha_1[int(n_frames*0.4):])
        wc_means[xi] = mean_alpha
        
    axes_rdf[2].set_xlabel(r'Distance ($\sigma^*$)')
    fig_rdf.tight_layout()
    fig_rdf.savefig('figures/task4_rdf_comparison.png')
    
    ax_wc.set_xlabel(r'Time ($\tau^*$)')
    ax_wc.set_ylabel(r'$\alpha_1$')
    ax_wc.set_title('Warren-Cowley Short-Range Order Parameter')
    ax_wc.legend()
    ax_wc.grid(True)
    fig_wc.tight_layout()
    fig_wc.savefig('figures/task4_warren_cowley_timeseries.png')
    
    print("\n" + "="*50)
    print("TASK 4 RESULTS SUMMARY")
    print("="*50)
    for xi in xi_values:
        if xi in wc_means:
            print(f"xi = {xi:.1f}: <alpha_1> = {wc_means[xi]:.4f}")
    
    os.makedirs('report', exist_ok=True)
    # Rewrite the beginning of the report file (Task 4)
    with open('report/results_summary.md', 'w') as f:
        f.write("# Task Results Summary\n\n")
        f.write("## Task 4: Warren-Cowley Short-Range Order Parameter\n")
        for xi in xi_values:
            if xi in wc_means:
                f.write(f"- $\\xi = {xi:.1f}$: $\\langle \\alpha_1 \\rangle = {wc_means[xi]:.4f}$\n")

if __name__ == '__main__':
    main()
