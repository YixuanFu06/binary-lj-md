import os
import sys

# Limit numpy multithreading to avoid CPU oversubscription when using multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Add the project root directory to sys.path to resolve imports correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.md_engine import run_md

def run_standard_condition(xi):
    print(f"Starting standard run for xi={xi:.1f} ...")
    config = dict(
        N_A=250, N_B=250, rho_star=0.8,
        T_equil_K=200.0, T_prod_K=150.0,
        dt_fs=2.0,
        n_equil=5000, n_prod=50000, n_save=50,
        ensemble_prod='NVT',
        tau_nh_fs=100.0,
        xi=xi,
        seed=42 + int(xi*10),
        traj_file=f'data/trajectories/traj_xi{xi:.1f}.npy',
        thermo_file=f'data/thermo/thermo_xi{xi:.1f}.csv',
        species_file=f'data/trajectories/species_xi{xi:.1f}.npy',
        use_torch=False,
    )
    run_md(config)
    print(f"Finished standard run for xi={xi:.1f}")

def run_concentration_series(x_B):
    print(f"Starting concentration run for x_B={x_B:.1f} ...")
    N_total = 300
    N_B = int(np.round(N_total * x_B))
    N_A = N_total - N_B
    config = dict(
        N_A=N_A, N_B=N_B, rho_star=0.8,
        T_equil_K=120.0, T_prod_K=120.0,
        dt_fs=2.0,
        n_equil=5000, n_prod=30000, n_save=50,
        ensemble_prod='NVT',
        tau_nh_fs=100.0,
        xi=0.7,
        seed=100 + int(x_B*10),
        traj_file=f'data/trajectories/traj_xB{x_B:.1f}_xi0.7.npy',
        thermo_file=None, # Optional: could save thermo if desired, but explicitly only traj was requested
        species_file=f'data/trajectories/species_xB{x_B:.1f}_xi0.7.npy',
        use_torch=False,
    )
    run_md(config)
    print(f"Finished concentration run for x_B={x_B:.1f}")

def main():
    print("=" * 70)
    print("TASK 3 — BINARY SYSTEM PRODUCTION RUNS")
    print("=" * 70)
    
    os.makedirs('data/trajectories', exist_ok=True)
    os.makedirs('data/thermo', exist_ok=True)

    xi_values = [0.7, 1.0, 1.3]
    xB_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Run in parallel to save time
    # Max workers = 6 is a good balance for typical node CPUs
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = []
        
        # Submit standard conditions
        for xi in xi_values:
            futures.append(executor.submit(run_standard_condition, xi))
            
        # Submit concentration series
        for x_B in xB_values:
            futures.append(executor.submit(run_concentration_series, x_B))
            
        # Wait for all to complete
        for i, f in enumerate(futures):
            f.result() # Will raise any exception that occurred
            print(f"Progress: {i+1}/{len(futures)} tasks completed.")
            
    print("All Task 3 simulations completed successfully!")

if __name__ == '__main__':
    main()
