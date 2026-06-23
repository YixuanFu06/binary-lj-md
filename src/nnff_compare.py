import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matgl
from matgl.ext.ase import PESCalculator
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units
import torch

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.md_engine import (
    compute_forces_and_energy, make_default_params, load_physical_constants, M_KR, M_XE,
    SIGMA_KR, SIGMA_XE, EPS_KR_KB, EPS_XE_KB
)

# LJ to eV conversion
K_B_EV = 8.617333262145e-5 # eV/K
EPS_REF_EV = 171.0 * K_B_EV
SIGMA_REF = 3.636 # Angstrom
FORCE_REF_EV_A = EPS_REF_EV / SIGMA_REF

def get_lj_energy_forces(pos, box_L, species, xi=1.0):
    N = len(pos)
    params = make_default_params(xi=xi)
    f, pe, vir = compute_forces_and_energy(pos, species, box_L, params, use_torch=False)
    
    # convert pe to eV
    pe_ev = pe * EPS_REF_EV
    # convert forces to eV/A
    f_ev_a = f * FORCE_REF_EV_A
    return pe_ev, f_ev_a

def single_point_comparison():
    print("--- Single Point Comparison ---")
    traj_file = 'data/trajectories/traj_xi1.0.npy'
    species_file = 'data/trajectories/species_xi1.0.npy'
    
    if not os.path.exists(traj_file):
        print("Trajectory not found, skipping single point comparison.")
        return
        
    traj = np.load(traj_file)
    species = np.load(species_file)
    
    N = traj.shape[1]
    rho_star = 0.8
    box_L = (N / rho_star)**(1/3)
    box_L_A = box_L * SIGMA_REF
    
    pot = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2")
    calc = PESCalculator(potential=pot)
    
    symbols = ['Kr' if s == 0 else 'Xe' for s in species]
    
    n_snapshots = min(20, len(traj) // 10)
    indices = np.linspace(0, len(traj)-1, n_snapshots, dtype=int)
    
    results = []
    
    for idx in indices:
        pos_star = traj[idx]
        pos_A = pos_star * SIGMA_REF
        
        # M3GNet
        atoms = Atoms(symbols=symbols, positions=pos_A, cell=[box_L_A]*3, pbc=True)
        atoms.calc = calc
        
        e_m3g = atoms.get_potential_energy()
        f_m3g = atoms.get_forces()
        
        # LJ
        e_lj, f_lj = get_lj_energy_forces(pos_star, box_L, species, xi=1.0)
        
        dE_per_atom = np.abs(e_lj - e_m3g) / N
        dF_mae = np.mean(np.abs(f_lj - f_m3g))
        
        results.append({
            'frame': idx,
            'E_LJ': e_lj,
            'E_M3G': e_m3g,
            'dE_per_atom': dE_per_atom,
            'dF_mae': dF_mae
        })
        print(f"Frame {idx}: dE/N = {dE_per_atom:.4f} eV, dF = {dF_mae:.4f} eV/A")
        
    df = pd.DataFrame(results)
    os.makedirs('data/nnff_compare', exist_ok=True)
    df.to_csv('data/nnff_compare/energy_force_comparison.csv', index=False)
    print("Saved comparison to data/nnff_compare/energy_force_comparison.csv")

def pes_comparison():
    print("--- PES Dimer Comparison ---")
    pot = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2")
    calc = PESCalculator(potential=pot)
    
    r_values = np.linspace(2.5, 8.0, 50)
    pairs = [('Kr', 'Kr', 0, 0), ('Kr', 'Xe', 0, 1), ('Xe', 'Xe', 1, 1)]
    
    plt.figure(figsize=(15, 5))
    for i, (sym1, sym2, sp1, sp2) in enumerate(pairs):
        e_m3g_list = []
        e_lj_list = []
        for r in r_values:
            atoms = Atoms(symbols=[sym1, sym2], positions=[[0,0,0], [r,0,0]])
            atoms.calc = calc
            e_m3g = atoms.get_potential_energy()
            
            atom1 = Atoms(symbols=[sym1], positions=[[0,0,0]]); atom1.calc=calc; e1=atom1.get_potential_energy()
            atom2 = Atoms(symbols=[sym2], positions=[[0,0,0]]); atom2.calc=calc; e2=atom2.get_potential_energy()
            e_m3g_list.append(e_m3g - e1 - e2)
            
            pos_star = np.array([[0,0,0], [r/SIGMA_REF,0,0]])
            species = np.array([sp1, sp2])
            pe, _ = get_lj_energy_forces(pos_star, 100.0, species, xi=1.0)
            e_lj_list.append(pe)
            
        plt.subplot(1, 3, i+1)
        plt.plot(r_values, e_m3g_list, 'b-', label='M3GNet')
        plt.plot(r_values, e_lj_list, 'r--', label='LJ')
        
        # Find minimums
        min_idx_m3g = np.argmin(e_m3g_list)
        r0_m3g, eps_m3g = r_values[min_idx_m3g], e_m3g_list[min_idx_m3g]
        min_idx_lj = np.argmin(e_lj_list)
        r0_lj, eps_lj = r_values[min_idx_lj], e_lj_list[min_idx_lj]
        
        # Mark minimums
        plt.plot(r0_m3g, eps_m3g, 'b*', markersize=10)
        plt.plot(r0_lj, eps_lj, 'r*', markersize=10)
        
        plt.xlabel('Separation r (Å)')
        plt.ylabel('Potential Energy (eV)')
        plt.title(f'{sym1}-{sym2} Dimer PES\n$r_0^{{M3G}}={r0_m3g:.2f}\AA, \epsilon^{{M3G}}={eps_m3g:.3f}$eV\n$r_0^{{LJ}}={r0_lj:.2f}\AA, \epsilon^{{LJ}}={eps_lj:.3f}$eV', fontsize=10)
        plt.legend()
        plt.grid(True)
        
    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('figures/task9_pes_dimer_comparison.png', dpi=300)
    print("Saved PES to figures/task9_pes_dimer_comparison.png")

def short_md():
    print("--- Short MD Comparison ---")
    pot = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2")
    calc = PESCalculator(potential=pot)
    
    N = 50
    T_K = 150.0
    rho_star = 0.8
    box_L = (N / rho_star)**(1/3)
    box_L_A = box_L * SIGMA_REF
    
    symbols = ['Kr'] * (N//2) + ['Xe'] * (N - N//2)
    np.random.seed(42)
    pos = np.random.rand(N, 3) * box_L_A
    atoms = Atoms(symbols=symbols, positions=pos, cell=[box_L_A]*3, pbc=True)
    
    atoms.calc = calc
    from ase.optimize import LBFGS
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=0.5, steps=50)
    
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_K)
    dyn = Langevin(atoms, 2.0 * units.fs, temperature_K=T_K, friction=0.01)
    
    traj_m3g = []
    def save_traj():
        traj_m3g.append(atoms.get_positions())
        
    dyn.attach(save_traj, interval=10)
    print("Running M3GNet MD for 500 steps...")
    dyn.run(500)
    
    from src.md_engine import run_md
    print("Running LJ MD for 500 steps...")
    config = dict(
        N_A=N//2, N_B=N - N//2, rho_star=rho_star,
        T_equil_K=T_K, T_prod_K=T_K, dt_fs=2.0,
        n_equil=100, n_prod=500, n_save=10,
        ensemble_prod='NVT', tau_nh_fs=100.0,
        xi=1.0, seed=42, use_torch=False
    )
    res = run_md(config)
    print("MD completed.")
    
    from src.analysis import compute_partial_rdf
    
    # LJ trajectory is in reduced units, scale by SIGMA_REF
    traj_lj_star = np.array(res['trajectory'])
    traj_lj_A = traj_lj_star * SIGMA_REF
    species = np.array([0] * (N//2) + [1] * (N - N//2))
    
    # Compute RDFs
    r_lj, _, _, g_AB_lj = compute_partial_rdf(traj_lj_A, species, box_L_A, dr=0.2, r_max=box_L_A/2.0)
    traj_m3g_np = np.array(traj_m3g)
    r_m3g, _, _, g_AB_m3g = compute_partial_rdf(traj_m3g_np, species, box_L_A, dr=0.2, r_max=box_L_A/2.0)
    
    # Plotting AB RDF
    plt.figure(figsize=(6, 5))
    plt.plot(r_m3g, g_AB_m3g, 'b-', label='M3GNet Kr-Xe')
    plt.plot(r_lj, g_AB_lj, 'r--', label='LJ Kr-Xe')
    plt.xlabel('r (Å)')
    plt.ylabel('$g_{Kr-Xe}(r)$')
    plt.title('Radial Distribution Function Comparison')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('figures/task9_rdf_lj_vs_m3gnet.png', dpi=300)
    print("Saved RDF comparison to figures/task9_rdf_lj_vs_m3gnet.png")
def benchmark_performance():
    print("--- Benchmark ---")
    N_values = [50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]
    
    pot = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2")
    calc = PESCalculator(potential=pot)
    
    t_lj = []
    t_m3g = []
    
    from src.md_engine import run_md
    for N in N_values:
        print(f"Benchmarking N={N}...")
        rho_star = 0.8
        box_L = (N / rho_star)**(1/3)
        box_L_A = box_L * SIGMA_REF
        
        current_steps = 100 if N <= 500 else 10
        
        symbols = ['Kr'] * (N//2) + ['Xe'] * (N - N//2)
        pos = np.random.rand(N, 3) * box_L_A
        atoms = Atoms(symbols=symbols, positions=pos, cell=[box_L_A]*3, pbc=True)
        atoms.calc = calc
        
        start = time.time()
        dyn = VelocityVerlet(atoms, 2.0 * units.fs, logfile=None)
        dyn.run(current_steps)
        m3g_time = (time.time() - start) * 1000 / current_steps
        t_m3g.append(m3g_time)
        
        config = dict(
            N_A=N//2, N_B=N - N//2, rho_star=rho_star,
            T_equil_K=150.0, T_prod_K=150.0, dt_fs=2.0,
            n_equil=0, n_prod=current_steps, n_save=current_steps,
            ensemble_prod='NVT', tau_nh_fs=100.0,
            xi=1.0, seed=42, use_torch=False
        )
        start = time.time()
        run_md(config)
        lj_time = (time.time() - start) * 1000 / current_steps
        t_lj.append(lj_time)
        
    plt.figure(figsize=(6, 5))
    plt.plot(N_values, t_lj, 'ro-', label='LJ (Numpy)')
    plt.plot(N_values, t_m3g, 'bs-', label='M3GNet (ASE)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Atoms (N)')
    plt.ylabel('Time per step (ms)')
    plt.title('Performance Benchmark')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig('figures/task9_performance_benchmark.png', dpi=300)
    print("Saved benchmark to figures/task9_performance_benchmark.png")

if __name__ == '__main__':
    single_point_comparison()
    pes_comparison()
    short_md()
    benchmark_performance()
