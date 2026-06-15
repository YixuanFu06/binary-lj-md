import os
import sys

# Add the project root directory to sys.path to resolve imports correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.md_engine import (
    make_default_params,
    init_fcc_lattice,
    get_reduced_masses,
    validate_forces,
    run_md
)

def test_task1_validation():
    print("=" * 70)
    print("TASK 1 — VALIDATION TEST")
    print("System : 256 Kr atoms,  T = 120 K,  ρ* = 0.80")
    print("Protocol: 2 000 NVT equil → 10 000 NVE production, dt = 2 fs")
    print("=" * 70)

    # --- Cross-validate force backends ---
    print("\n1) Cross-validating PyTorch vs NumPy forces …")
    params = make_default_params(xi=1.0)
    pos, species, box_L = init_fcc_lattice(256, 0, rho_star=0.8, seed=42)
    fe, pe, we = validate_forces(pos, species, box_L, params)
    assert fe < 1e-8, f"Force mismatch too large: {fe}"
    print("  ✓ PyTorch and NumPy forces agree.\n")

    # --- Production run ---
    print("2) Running MD …")
    config = dict(
        N_A=256, N_B=0, rho_star=0.8,
        T_equil_K=120.0, T_prod_K=120.0,
        dt_fs=2.0,
        n_equil=2000, n_prod=10000, n_save=10,
        xi=1.0,
        ensemble_prod='NVE',
        tau_nh_fs=100.0,
        seed=42,
        traj_file='data/trajectories/test_kr256.npy',
        thermo_file='data/thermo/test_kr256.csv',
        use_torch=False,   # NumPy for speed; forces already cross-validated
    )
    results = run_md(config)

    # --- Verdict ---
    drift = abs(results.get('E_drift_percent', 999))
    print(f"\n{'='*70}")
    if drift < 0.1:
        print(f"✅  PASS — energy drift = {drift:.5f} %  (< 0.1 %)")
    else:
        print(f"❌  FAIL — energy drift = {drift:.5f} %  (≥ 0.1 %)")
    print(f"{'='*70}")
    
    assert drift < 0.1, f"Energy drift too large: {drift:.5f} % (limit 0.1%)"

if __name__ == '__main__':
    test_task1_validation()
