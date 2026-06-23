import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.md_engine import run_md, make_default_params, _compute_forces_numpy, EPS_REF
from src.analysis import compute_mixing_enthalpy, compute_heat_capacity

def block_average(data, num_blocks=5):
    """Simple block averaging to get standard error."""
    block_size = len(data) // num_blocks
    if block_size == 0:
        return np.mean(data), 0.0
    blocks = [np.mean(data[i*block_size:(i+1)*block_size]) for i in range(num_blocks)]
    return np.mean(blocks), np.std(blocks, ddof=1) / np.sqrt(num_blocks)

def run_pure_systems():
    print("Running pure Kr (x_B=0.0) ...")
    config_Kr = dict(
        N_A=300, N_B=0, rho_star=0.8,
        T_equil_K=120.0, T_prod_K=120.0, dt_fs=2.0,
        n_equil=5000, n_prod=30000, n_save=50,
        ensemble_prod='NVT', tau_nh_fs=100.0, xi=0.7, seed=42,
        traj_file='data/trajectories/traj_xB0.0_xi0.7.npy',
        thermo_file='data/thermo/thermo_xB0.0.csv',
        species_file='data/trajectories/species_xB0.0_xi0.7.npy',
        use_torch=False
    )
    if not os.path.exists(config_Kr['thermo_file']):
        run_md(config_Kr)
        
    print("Running pure Xe (x_B=1.0) ...")
    config_Xe = dict(
        N_A=0, N_B=300, rho_star=0.8,
        T_equil_K=120.0, T_prod_K=120.0, dt_fs=2.0,
        n_equil=5000, n_prod=30000, n_save=50,
        ensemble_prod='NVT', tau_nh_fs=100.0, xi=0.7, seed=43,
        traj_file='data/trajectories/traj_xB1.0_xi0.7.npy',
        thermo_file='data/thermo/thermo_xB1.0.csv',
        species_file='data/trajectories/species_xB1.0_xi0.7.npy',
        use_torch=False
    )
    if not os.path.exists(config_Xe['thermo_file']):
        run_md(config_Xe)

def evaluate_trajectory_epot(traj, species, box_L, params):
    """Evaluate potential energy for each frame in the production part of a trajectory."""
    n_frames = len(traj)
    start_idx = int(n_frames * 0.4)
    traj_prod = traj[start_idx:]
    
    epot_list = np.zeros(len(traj_prod))
    for i, pos in enumerate(traj_prod):
        _, pe, _ = _compute_forces_numpy(pos, species, box_L, params)
        epot_list[i] = pe
    return epot_list

def parabola(x, Omega):
    return Omega * x * (1.0 - x)

def main():
    os.makedirs('figures', exist_ok=True)
    os.makedirs('report', exist_ok=True)
    
    print("="*60)
    print("TASK 6.1: MIXING ENTHALPY")
    print("="*60)
    run_pure_systems()
    
    # Analyze mixture trajectories from x_B = 0.1 to 0.9
    xB_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    xi = 0.7
    params = make_default_params(xi=xi)
    N_total = 300
    rho_star = 0.8
    box_L = (N_total / rho_star)**(1/3)
    
    epot_mix_mean = np.zeros(len(xB_values))
    epot_mix_err = np.zeros(len(xB_values))
    
    print("Evaluating potential energies from trajectories...")
    for i, x_B in enumerate(xB_values):
        traj = np.load(f'data/trajectories/traj_xB{x_B:.1f}_xi0.7.npy')
        species = np.load(f'data/trajectories/species_xB{x_B:.1f}_xi0.7.npy')
        
        epot_array = evaluate_trajectory_epot(traj, species, box_L, params)
        mean_pe, err_pe = block_average(epot_array / N_total)
        epot_mix_mean[i] = mean_pe
        epot_mix_err[i] = err_pe
        print(f"  x_B = {x_B:.1f}: <E_pot>/N = {mean_pe:.4f} ± {err_pe:.4f}")

    # Pure systems
    thermo_Kr = np.loadtxt('data/thermo/thermo_xB0.0.csv', delimiter=',', skiprows=1)
    thermo_Xe = np.loadtxt('data/thermo/thermo_xB1.0.csv', delimiter=',', skiprows=1)
    
    # E_pot is column 3 (0=step, 1=T, 2=E_kin, 3=E_pot)
    epot_Kr_mean, epot_Kr_err = block_average(thermo_Kr[:, 3] / N_total)
    epot_Xe_mean, epot_Xe_err = block_average(thermo_Xe[:, 3] / N_total)
    
    print(f"  x_B = 0.0: <E_pot>/N = {epot_Kr_mean:.4f} ± {epot_Kr_err:.4f}")
    print(f"  x_B = 1.0: <E_pot>/N = {epot_Xe_mean:.4f} ± {epot_Xe_err:.4f}")
    
    # Compute Delta H_mix
    dH_mix = compute_mixing_enthalpy(epot_mix_mean, epot_Kr_mean, epot_Xe_mean, xB_values)
    
    # Fit Delta H_mix = Omega * x_B * (1 - x_B)
    popt, _ = curve_fit(parabola, xB_values, dH_mix)
    Omega = popt[0]
    print(f"\nExtracted Interaction Parameter Omega = {Omega:.4f} epsilon*")
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.errorbar(xB_values, dH_mix, yerr=epot_mix_err, fmt='o', label='MD Simulation', color='blue')
    
    x_fit = np.linspace(0, 1, 100)
    plt.plot(x_fit, parabola(x_fit, Omega), 'r--', label=f'Regular Solution Fit\n$\\Omega={Omega:.3f}\\ \\epsilon^*$')
    
    plt.plot([0, 1], [0, 0], 'k:', alpha=0.5)
    plt.xlabel('Xe mole fraction $x_B$')
    plt.ylabel(r'Mixing Enthalpy $\Delta H_{mix}$ ($\epsilon^*$)')
    plt.title(f'Enthalpy of Mixing at $\\xi={xi:.1f}$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/task6_mixing_enthalpy.png')
    plt.close()
    
    print("\n" + "="*60)
    print("TASK 6.3: HEAT CAPACITY")
    print("="*60)
    
    # NVT production run for xi=1.0, x_B=0.5. (N=500, T=150K)
    thermo_10 = np.loadtxt('data/thermo/thermo_xi1.0.csv', delimiter=',', skiprows=1)
    # E_tot is column 4
    E_tot_array = thermo_10[:, 4]
    N_std = 500
    T_target = 150.0
    T_star = T_target / EPS_REF
    
    cv_per_atom = compute_heat_capacity(E_tot_array, T_star, N_std)
    dp_limit = 3.0
    
    print(f"C_v / (N k_B) computed: {cv_per_atom:.4f}")
    print(f"Dulong-Petit limit    : {dp_limit:.4f}")
    
    # Append to summary
    with open('report/results_summary.md', 'a') as f:
        f.write("\n## Thermodynamic Analysis (Task 6)\n")
        f.write(f"- **Interaction Parameter $\\Omega$** (at $\\xi=0.7$): {Omega:.4f} $\\epsilon^*$\n")
        f.write(f"- **Heat Capacity $C_v / (N k_B)$** (at $\\xi=1.0$): {cv_per_atom:.4f} (Dulong-Petit limit: 3.0)\n")
        
    print("\nTask 6 completely finished. Results appended to report/results_summary.md")

if __name__ == '__main__':
    main()
