import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to sys.path to resolve imports correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.md_engine import run_md, TAU_REF

def validate_energy_conservation():
    print("=" * 70)
    print("TASK 2.1 — ENERGY CONSERVATION TEST (NVE)")
    print("=" * 70)
    
    # Run with dt = 1 fs
    config_1fs = dict(
        N_A=256, N_B=0, rho_star=0.8,
        T_equil_K=120.0, T_prod_K=120.0,
        dt_fs=1.0,
        n_equil=2000, n_prod=10000, n_save=10,
        ensemble_prod='NVE',
        seed=42,
        use_torch=False,
    )
    res_1fs = run_md(config_1fs)
    
    # Run with dt = 2 fs
    config_2fs = dict(
        N_A=256, N_B=0, rho_star=0.8,
        T_equil_K=120.0, T_prod_K=120.0,
        dt_fs=2.0,
        n_equil=2000, n_prod=10000, n_save=10,
        ensemble_prod='NVE',
        seed=42,
        use_torch=False,
    )
    res_2fs = run_md(config_2fs)
    
    thermo_1fs = res_1fs['thermo']
    thermo_2fs = res_2fs['thermo']
    
    # Plotting
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(8, 5))
    
    t_1fs = thermo_1fs[:, 0] * 1.0 * 1e-15 / TAU_REF
    t_2fs = thermo_2fs[:, 0] * 2.0 * 1e-15 / TAU_REF
    
    e_1fs = thermo_1fs[:, 4] / 256
    e_2fs = thermo_2fs[:, 4] / 256
    
    plt.plot(t_1fs, e_1fs - e_1fs[0], label='dt = 1 fs')
    plt.plot(t_2fs, e_2fs - e_2fs[0], label='dt = 2 fs', alpha=0.8)
    
    plt.xlabel(r'Time ($\tau^*$)')
    plt.ylabel(r'$\Delta E_{tot}/N$ ($\epsilon^*$)')
    plt.title('Energy Conservation in NVE Ensemble')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/validation_energy_conservation.png')
    plt.close()
    
    # Calculate drift per tau*
    drift_1fs = abs(e_1fs[-1] - e_1fs[0]) / (t_1fs[-1] - t_1fs[0])
    drift_2fs = abs(e_2fs[-1] - e_2fs[0]) / (t_2fs[-1] - t_2fs[0])
    
    print(f"Drift dt=1fs: {drift_1fs:.6e} ε*/(atom·τ*)")
    print(f"Drift dt=2fs: {drift_2fs:.6e} ε*/(atom·τ*)")
    
    assert drift_1fs < 0.01, f"Drift too large for dt=1fs: {drift_1fs}"
    assert drift_2fs < 0.01, f"Drift too large for dt=2fs: {drift_2fs}"
    print("✓ Energy conservation validated.\n")

def validate_temperature_equilibration():
    print("=" * 70)
    print("TASK 2.2 — TEMPERATURE EQUILIBRATION TEST (NVT)")
    print("=" * 70)
    
    # Set n_equil=0 and use production run for NVT to capture the equilibration process
    config_nvt = dict(
        N_A=256, N_B=0, rho_star=0.8,
        T_equil_K=120.0, T_prod_K=120.0,
        dt_fs=2.0,
        n_equil=0, n_prod=10000, n_save=10,
        ensemble_prod='NVT',
        tau_nh_fs=100.0,
        seed=42,
        use_torch=False,
    )
    res_nvt = run_md(config_nvt)
    thermo = res_nvt['thermo']
    
    t = thermo[:, 0] * 2.0 * 1e-15 / TAU_REF
    temp = thermo[:, 1]
    
    plt.figure(figsize=(8, 5))
    plt.plot(t, temp, label='Instantaneous T', alpha=0.8)
    plt.axhline(120.0, color='r', linestyle='--', label='Target T (120 K)')
    
    plt.xlabel(r'Time ($\tau^*$)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature Equilibration in NVT Ensemble')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/validation_temperature.png')
    plt.close()
    
    # Calculate average temperature after equilibration (last 50% of the run)
    half_idx = len(temp) // 2
    avg_temp = np.mean(temp[half_idx:])
    print(f"\nAverage temperature (post-equilibration): {avg_temp:.2f} K")
    
    error_pct = abs(avg_temp - 120.0) / 120.0 * 100
    print(f"Temperature error: {error_pct:.2f}%")
    
    assert error_pct < 2.0, f"Temperature did not converge within 2%: {avg_temp} K"
    print("✓ Temperature equilibration validated.\n")

if __name__ == '__main__':
    validate_energy_conservation()
    validate_temperature_equilibration()
