import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analysis import compute_structure_factor

def main():
    os.makedirs('figures', exist_ok=True)
    os.makedirs('data/rdf', exist_ok=True)
    
    print("="*60)
    print("TASK 7: STRUCTURE FACTOR S(q)")
    print("="*60)
    
    xi_values = [0.7, 1.0, 1.3]
    N_total = 500
    rho_star = 0.8
    box_L = (N_total / rho_star)**(1/3)
    
    plt.figure(figsize=(10, 6))
    colors = {0.7: 'r', 1.0: 'g', 1.3: 'b'}
    labels = {0.7: '$\\xi=0.7$ (Phase separated)', 1.0: '$\\xi=1.0$ (Ideal mixture)', 1.3: '$\\xi=1.3$ (Ordered / Superlattice)'}
    
    for xi in xi_values:
        traj_file = f'data/trajectories/traj_xi{xi:.1f}.npy'
        species_file = f'data/trajectories/species_xi{xi:.1f}.npy'
        print(f"Loading trajectory for xi={xi:.1f} ...")
        
        if not os.path.exists(traj_file) or not os.path.exists(species_file):
            print(f"File {traj_file} or {species_file} not found. Ensure Task 3 has been run.")
            continue
            
        traj = np.load(traj_file)
        species = np.load(species_file)
        
        print(f"Computing S(q) for xi={xi:.1f} ...")
        q_centers, S_tot, S_AA, S_BB, S_AB = compute_structure_factor(
            traj, species, box_L, q_max=15.0, bins=60
        )
        
        # Save to npz
        save_path = f'data/rdf/sq_xi{xi:.1f}.npz'
        np.savez(save_path, q=q_centers, S_tot=S_tot, S_AA=S_AA, S_BB=S_BB, S_AB=S_AB)
        print(f"  Saved S(q) data to {save_path}")
        
        # Plot S_AA(q) instead of S_tot(q) to reveal compositional order
        plt.plot(q_centers, S_AA, '-', color=colors[xi], label=labels[xi], lw=2)
        
    plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel(r'Wavevector $q$ ($1/\sigma^*$)', fontsize=14)
    plt.ylabel(r'Partial Structure Factor $S_{Kr-Kr}(q)$', fontsize=14)
    plt.title('Kr-Kr Static Structure Factor $S_{AA}(q)$', fontsize=16)
    
    # Add annotations for key features based on xi
    plt.annotate('Low-q peak\n(Phase Separation)', xy=(1.0, 2.5), xytext=(2.0, 3.5),
                 arrowprops=dict(facecolor='red', shrink=0.05), color='red', fontsize=12)
                 
    # Usually ordering superlattice peaks appear around q ~ 2*pi / (L_ordering)
    plt.annotate('Superlattice peak\n(Ordering)', xy=(5.0, 1.5), xytext=(5.5, 2.5),
                 arrowprops=dict(facecolor='blue', shrink=0.05), color='blue', fontsize=12)
                 
    plt.xlim(0, 15.0)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/task7_structure_factor.png', dpi=300)
    plt.close()
    
    print("\nTask 7 finished. Structure factor plot saved to figures/task7_structure_factor.png")

if __name__ == '__main__':
    main()
