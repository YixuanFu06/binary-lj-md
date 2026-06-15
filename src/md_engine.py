"""
Binary Component Lennard-Jones Molecular Dynamics Simulation Engine
===================================================================

Core MD engine for simulating Kr-Xe mixed noble gas systems using the
Lennard-Jones potential. Supports both PyTorch autograd and analytical
NumPy force computation, Velocity-Verlet integration, and Nosé-Hoover
thermostat for NVT ensemble simulations.

All internal calculations use **reduced units** based on Kr:
    Length:  σ* = σ_Kr = 3.636 Å
    Energy:  ε* = ε_Kr = 171.0 k_B  (i.e. ε_Kr / k_B = 171.0 K)
    Mass:    m* = m_Kr = 83.798 amu
    Time:    τ* = σ* √(m*/ε*)

In reduced units, k_B = 1, and temperature is T* = T[K] / 171.0.

Course: Computational Physics, Spring 2026
"""
import os
import csv
import time as time_module
from typing import Tuple, Dict, Optional
from dataclasses import dataclass, field
import numpy as np
import torch
import yaml

# ====================================================================
# Physical Constants & Parameter Loading
# ====================================================================

# Default values if config.yaml is missing or incomplete
DEFAULT_PHYSICAL_CONSTANTS = {
    'K_B': 1.380649e-23,
    'AMU': 1.66053906660e-27,
    'ANGSTROM': 1.0e-10,
    'M_KR': 83.798,
    'M_XE': 131.293,
    'EPS_KR_KB': 171.0,
    'EPS_XE_KB': 221.0,
    'SIGMA_KR': 3.636,
    'SIGMA_XE': 3.924,
    'reference_species': 'KR'
}

def load_physical_constants(config_path: Optional[str] = None) -> Dict:
    """Load physical constants from a YAML file, with defaults as fallback."""
    if config_path is None:
        # Find config.yaml relative to this script (assumed to be in src/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.abspath(os.path.join(script_dir, '..', 'config.yaml'))
        
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            if data is None:
                data = {}
            merged = DEFAULT_PHYSICAL_CONSTANTS.copy()
            merged.update({k: v for k, v in data.items() if v is not None})
            return merged
        except Exception as e:
            print(f"Warning: Failed to parse {config_path} ({e}). Using default physical parameters.")
            return DEFAULT_PHYSICAL_CONSTANTS.copy()
    else:
        # Silently fall back to defaults if not found
        return DEFAULT_PHYSICAL_CONSTANTS.copy()

# Load constant variables
_constants = load_physical_constants()

K_B = float(_constants['K_B'])
AMU = float(_constants['AMU'])
ANGSTROM = float(_constants['ANGSTROM'])

# Atomic properties
M_KR = float(_constants['M_KR'])
M_XE = float(_constants['M_XE'])
EPS_KR_KB = float(_constants['EPS_KR_KB'])
EPS_XE_KB = float(_constants['EPS_XE_KB'])
SIGMA_KR = float(_constants['SIGMA_KR'])
SIGMA_XE = float(_constants['SIGMA_XE'])

# Reference units (based on selected reference species)
_ref_spec = _constants.get('reference_species', 'KR').upper()
if _ref_spec == 'KR':
    SIGMA_REF = SIGMA_KR
    EPS_REF = EPS_KR_KB
    MASS_REF = M_KR
elif _ref_spec == 'XE':
    SIGMA_REF = SIGMA_XE
    EPS_REF = EPS_XE_KB
    MASS_REF = M_XE
else:
    raise ValueError(f"Unknown reference species: {_ref_spec}")

TAU_REF = (SIGMA_REF * ANGSTROM) * np.sqrt(
    (MASS_REF * AMU) / (EPS_REF * K_B)
)  # [s]


# ====================================================================
# 1.1  Parameter Setup
# ====================================================================

def _lj_potential_scalar(r: float, eps: float, sigma: float) -> float:
    """Compute LJ potential V(r) = 4ε[(σ/r)^12 − (σ/r)^6] at a single distance."""
    sr6 = (sigma / r) ** 6
    return 4.0 * eps * (sr6 * sr6 - sr6)


@dataclass
class LJParams:
    """
    Lennard-Jones parameters for a binary A(Kr)–B(Xe) mixture in reduced units.

    Attributes
    ----------
    eps_AA, eps_BB, eps_AB : float
        LJ well depths ε (reduced).
    sigma_AA, sigma_BB, sigma_AB : float
        LJ size parameters σ (reduced).
    r_cut : float
        Interaction cutoff radius (reduced).
    xi : float
        Lorentz-Berthelot cross-interaction modifier (default 1.0).
    v_shift_AA, v_shift_BB, v_shift_AB : float
        Potential values at the cutoff, computed automatically so that
        V_shifted(r_c) = 0.
    """
    eps_AA: float
    eps_BB: float
    eps_AB: float
    sigma_AA: float
    sigma_BB: float
    sigma_AB: float
    r_cut: float
    xi: float = 1.0
    v_shift_AA: float = field(init=False, default=0.0)
    v_shift_BB: float = field(init=False, default=0.0)
    v_shift_AB: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        """Pre-compute the potential shift at the cutoff for each pair type."""
        self.v_shift_AA = _lj_potential_scalar(self.r_cut, self.eps_AA, self.sigma_AA)
        self.v_shift_BB = _lj_potential_scalar(self.r_cut, self.eps_BB, self.sigma_BB)
        self.v_shift_AB = _lj_potential_scalar(self.r_cut, self.eps_AB, self.sigma_AB)


def compute_cross_params(
    eps_AA: float, eps_BB: float,
    sigma_AA: float, sigma_BB: float,
    xi: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute cross-interaction parameters via Lorentz-Berthelot mixing rules.

        σ_AB = (σ_AA + σ_BB) / 2       (Lorentz)
        ε_AB = ξ √(ε_AA · ε_BB)        (modified Berthelot)

    Parameters
    ----------
    eps_AA, eps_BB : float  – Well depths of pure species.
    sigma_AA, sigma_BB : float  – Size parameters of pure species.
    xi : float  – Mixing modifier (1.0 = standard Berthelot).

    Returns
    -------
    (eps_AB, sigma_AB)
    """
    sigma_AB = (sigma_AA + sigma_BB) / 2.0
    eps_AB = xi * np.sqrt(eps_AA * eps_BB)
    return eps_AB, sigma_AB


def make_default_params(xi: float = 1.0) -> LJParams:
    """
    Create LJParams for Kr-Xe in reduced units with cutoff r_c = 2.5 σ_AA.

    Parameters
    ----------
    xi : float – Cross-interaction mixing modifier.
    """
    eps_AA = 1.0
    eps_BB = EPS_XE_KB / EPS_REF               # ≈ 1.2924
    sigma_AA = 1.0
    sigma_BB = SIGMA_XE / SIGMA_REF            # ≈ 1.0792
    eps_AB, sigma_AB = compute_cross_params(eps_AA, eps_BB, sigma_AA, sigma_BB, xi)
    return LJParams(
        eps_AA=eps_AA, eps_BB=eps_BB, eps_AB=eps_AB,
        sigma_AA=sigma_AA, sigma_BB=sigma_BB, sigma_AB=sigma_AB,
        r_cut=2.5 * sigma_AA, xi=xi,
    )


def get_reduced_masses(species: np.ndarray) -> np.ndarray:
    """Return reduced masses: Kr → 1.0, Xe → m_Xe/m_Kr ≈ 1.567."""
    mass_map = np.array([1.0, M_XE / MASS_REF])
    return mass_map[species]


# ====================================================================
# 1.2  Initial Configuration
# ====================================================================

def init_fcc_lattice(
    N_A: int, N_B: int, x_B: float = 0.0,
    rho_star: float = 0.8, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Place N = N_A + N_B atoms on an FCC lattice at number density ρ*.

    Species are assigned randomly so that the B-fraction is N_B / N.
    (The ``x_B`` argument is kept for API compatibility but N_B takes
    precedence.)

    Parameters
    ----------
    N_A : int  – Number of A (Kr) atoms.
    N_B : int  – Number of B (Xe) atoms.
    x_B : float  – Desired B-fraction (used only if N_B == 0 and x_B > 0).
    rho_star : float  – Reduced number density  N σ³ / V  (default 0.8).
    seed : int  – RNG seed.

    Returns
    -------
    pos : ndarray [N, 3]  – Atomic positions (reduced units, in [0, L)).
    species : ndarray [N] of int64  – Species labels (0 = Kr, 1 = Xe).
    box_L : float  – Cubic box side length (reduced).
    """
    N = N_A + N_B
    if N == 0:
        raise ValueError("Total number of atoms must be > 0")
    rng = np.random.default_rng(seed)

    # Number of FCC unit cells per side (4 atoms per unit cell)
    n_cells = int(np.ceil((N / 4.0) ** (1.0 / 3.0)))
    n_sites = 4 * n_cells ** 3

    # Box length from target density: ρ* = N / L³  →  L = (N / ρ*)^{1/3}
    box_L: float = (N / rho_star) ** (1.0 / 3.0)
    a = box_L / n_cells  # lattice constant

    # FCC basis (fractional coords inside one unit cell)
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ])

    # Generate all FCC sites
    sites = np.empty((n_sites, 3))
    idx = 0
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                for b in basis:
                    sites[idx] = (np.array([ix, iy, iz], dtype=float) + b) * a
                    idx += 1

    # Select N sites (randomly if we have more sites than atoms)
    if n_sites > N:
        chosen = rng.choice(n_sites, size=N, replace=False)
        chosen.sort()
        pos = sites[chosen]
    else:
        pos = sites[:N].copy()

    # Assign species
    species = np.zeros(N, dtype=np.int64)
    if N_B > 0:
        b_indices = rng.choice(N, size=N_B, replace=False)
        species[b_indices] = 1

    return pos, species, box_L


def init_velocities(
    N: int, T_target_K: float, masses: np.ndarray, seed: int = 42,
) -> np.ndarray:
    """
    Draw velocities from the Maxwell-Boltzmann distribution and remove
    centre-of-mass drift.

    Parameters
    ----------
    N : int  – Number of atoms.
    T_target_K : float  – Target temperature [K].
    masses : ndarray [N]  – Reduced masses.
    seed : int  – RNG seed.

    Returns
    -------
    vel : ndarray [N, 3]  – Velocities (reduced units).
    """
    rng = np.random.default_rng(seed)
    T_star = T_target_K / EPS_REF  # reduced temperature

    # v_{α,i} ~ N(0, √(T*/m*_α))  (in reduced units k_B = 1)
    sigma_v = np.sqrt(T_star / masses)            # [N]
    vel = rng.normal(size=(N, 3)) * sigma_v[:, np.newaxis]

    # Remove centre-of-mass drift
    total_p = np.sum(masses[:, np.newaxis] * vel, axis=0)
    vel -= total_p / np.sum(masses)

    # Rescale to exact target temperature
    ek = 0.5 * np.sum(masses[:, np.newaxis] * vel ** 2)
    T_actual = 2.0 * ek / (3.0 * N)
    if T_actual > 0:
        vel *= np.sqrt(T_star / T_actual)

    return vel


# ====================================================================
# 1.3  Force Computation
# ====================================================================

def compute_forces_and_energy(
    pos: np.ndarray, species: np.ndarray,
    box_L: float, params: LJParams,
    use_torch: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute forces, potential energy, and virial for the current configuration.

    Parameters
    ----------
    pos : ndarray [N, 3]  – Positions (reduced).
    species : ndarray [N]  – Species labels (0/1).
    box_L : float  – Box side length (reduced).
    params : LJParams  – Interaction parameters.
    use_torch : bool  – True → PyTorch autograd; False → analytical NumPy.

    Returns
    -------
    forces : ndarray [N, 3]  – Forces (reduced).
    pe : float  – Total potential energy (reduced).
    virial : float  – Virial sum W = Σ_{i<j} 24ε[2(σ/r)^12 − (σ/r)^6].
    """
    if use_torch:
        return _compute_forces_torch(pos, species, box_L, params)
    else:
        return _compute_forces_numpy(pos, species, box_L, params)


# ---------------------------------------------------------------
# PyTorch (autograd) implementation
# ---------------------------------------------------------------
def _compute_forces_torch(
    pos: np.ndarray, species: np.ndarray,
    box_L: float, params: LJParams,
) -> Tuple[np.ndarray, float, float]:
    """Force computation via PyTorch autograd (differentiable)."""
    N = len(pos)
    dtype = torch.float64

    # Positions as a leaf tensor with gradient tracking
    pos_t = torch.tensor(pos, dtype=dtype, requires_grad=True)

    # 2×2 parameter lookup tables (indexed by species pair)
    eps_tbl = torch.tensor(
        [[params.eps_AA, params.eps_AB],
         [params.eps_AB, params.eps_BB]], dtype=dtype)
    sig_tbl = torch.tensor(
        [[params.sigma_AA, params.sigma_AB],
         [params.sigma_AB, params.sigma_BB]], dtype=dtype)
    vsh_tbl = torch.tensor(
        [[params.v_shift_AA, params.v_shift_AB],
         [params.v_shift_AB, params.v_shift_BB]], dtype=dtype)

    sp = torch.tensor(species, dtype=torch.long)

    # Pairwise displacement  Δr[i,j,:] = r_i − r_j
    dr = pos_t.unsqueeze(1) - pos_t.unsqueeze(0)          # [N,N,3]
    dr = dr - box_L * torch.round(dr / box_L)             # minimum image

    r2 = (dr * dr).sum(dim=2)                              # [N,N]

    # Per-pair parameters via fancy indexing
    eps_ij = eps_tbl[sp.unsqueeze(1), sp.unsqueeze(0)]     # [N,N]
    sig_ij = sig_tbl[sp.unsqueeze(1), sp.unsqueeze(0)]
    vsh_ij = vsh_tbl[sp.unsqueeze(1), sp.unsqueeze(0)]

    # Masks: upper triangle (avoid double-counting) AND within cutoff
    mask_tri = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    mask_cut = r2 < params.r_cut ** 2
    mask = mask_tri & mask_cut                              # [N,N]

    # Replace masked (invalid) r² with 1.0 to avoid 0-division / inf
    r2_safe = torch.where(mask, r2, torch.ones_like(r2))

    # LJ potential
    sr2 = (sig_ij * sig_ij) / r2_safe
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    v_ij = (4.0 * eps_ij * (sr12 - sr6) - vsh_ij) * mask.to(dtype)

    pe = v_ij.sum()

    # Forces via autograd: F = −∇V
    pe.backward()
    forces = -pos_t.grad.detach().cpu().numpy()
    pe_val = pe.item()

    # Virial (analytical, no graph needed)
    with torch.no_grad():
        w_ij = 24.0 * eps_ij * (2.0 * sr12 - sr6) * mask.to(dtype)
        virial_val = w_ij.sum().item()

    return forces, pe_val, virial_val


# ---------------------------------------------------------------
# NumPy (analytical) implementation
# ---------------------------------------------------------------
def _compute_forces_numpy(
    pos: np.ndarray, species: np.ndarray,
    box_L: float, params: LJParams,
) -> Tuple[np.ndarray, float, float]:
    """Force computation with analytical LJ formula — faster, no graph overhead."""
    N = len(pos)

    # Parameter tables
    eps_tbl = np.array([[params.eps_AA, params.eps_AB],
                        [params.eps_AB, params.eps_BB]])
    sig_tbl = np.array([[params.sigma_AA, params.sigma_AB],
                        [params.sigma_AB, params.sigma_BB]])
    vsh_tbl = np.array([[params.v_shift_AA, params.v_shift_AB],
                        [params.v_shift_AB, params.v_shift_BB]])

    # Pairwise displacements with minimum image convention
    dr = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]      # [N,N,3]
    dr -= box_L * np.round(dr / box_L)
    r2 = np.sum(dr * dr, axis=2)                             # [N,N]

    # Per-pair parameters
    eps_ij = eps_tbl[species[:, np.newaxis], species[np.newaxis, :]]
    sig_ij = sig_tbl[species[:, np.newaxis], species[np.newaxis, :]]
    vsh_ij = vsh_tbl[species[:, np.newaxis], species[np.newaxis, :]]

    # Self-interaction → infinite distance (excluded)
    np.fill_diagonal(r2, np.inf)
    mask = r2 < params.r_cut ** 2                            # [N,N]
    mask_tri = np.triu(np.ones((N, N), dtype=bool), k=1)

    r2_safe = np.where(mask, r2, 1.0)

    sig2 = sig_ij * sig_ij
    sr2 = sig2 / r2_safe
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6

    # Forces (sum over ALL j ≠ i, not just upper triangle)
    f_scalar = 24.0 * eps_ij * (2.0 * sr12 - sr6) / r2_safe * mask
    forces = np.sum(f_scalar[:, :, np.newaxis] * dr, axis=1)  # [N,3]

    # PE and virial (upper triangle only → no double counting)
    pe = np.sum((4.0 * eps_ij * (sr12 - sr6) - vsh_ij) * mask * mask_tri)
    virial = np.sum(24.0 * eps_ij * (2.0 * sr12 - sr6) * mask * mask_tri)

    return forces, float(pe), float(virial)


def validate_forces(
    pos: np.ndarray, species: np.ndarray,
    box_L: float, params: LJParams,
) -> Tuple[float, float, float]:
    """
    Cross-validate PyTorch autograd and NumPy analytical forces.
    Returns (max_force_error, pe_error, virial_error).
    """
    f_t, pe_t, w_t = _compute_forces_torch(pos, species, box_L, params)
    f_n, pe_n, w_n = _compute_forces_numpy(pos, species, box_L, params)
    f_err = float(np.max(np.abs(f_t - f_n)))
    pe_err = abs(pe_t - pe_n)
    w_err = abs(w_t - w_n)
    print(f"  Force max |Δ|  = {f_err:.3e}")
    print(f"  PE    |Δ|      = {pe_err:.3e}")
    print(f"  Virial |Δ|     = {w_err:.3e}")
    return f_err, pe_err, w_err


# ====================================================================
# 1.4  Velocity-Verlet Integrator  (NVE)
# ====================================================================

def velocity_verlet_step(
    pos: np.ndarray, vel: np.ndarray, forces: np.ndarray,
    dt: float, masses: np.ndarray, box_L: float,
    params: LJParams, species: np.ndarray,
    use_torch: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    One Velocity-Verlet step (NVE ensemble).

    Algorithm
    ---------
        v(t+½dt) = v(t) + ½ dt · a(t)
        r(t+dt)  = r(t) + dt · v(t+½dt)       + PBC
        a(t+dt)  ← force evaluation
        v(t+dt)  = v(t+½dt) + ½ dt · a(t+dt)

    Returns
    -------
    (pos_new, vel_new, forces_new, pe, virial)
    """
    m = masses[:, np.newaxis]
    acc = forces / m

    vel_half = vel + 0.5 * dt * acc

    pos_new = pos + dt * vel_half
    pos_new -= box_L * np.floor(pos_new / box_L)

    forces_new, pe, virial = compute_forces_and_energy(
        pos_new, species, box_L, params, use_torch=use_torch,
    )

    vel_new = vel_half + 0.5 * dt * (forces_new / m)
    return pos_new, vel_new, forces_new, pe, virial


# ====================================================================
# 1.5  Nosé-Hoover Thermostat
# ====================================================================

def nose_hoover_step(
    vel: np.ndarray, xi_NH: float,
    T_target: float, Q: float, dt: float,
    N: int, masses: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Apply one **half-step** of the Nosé-Hoover thermostat.

    This function is called **twice** per NVT integration step (once before
    and once after the Velocity-Verlet core) to form a symmetric, time-
    reversible (Trotter) decomposition.

    Operations (in order):
        1. Compute kinetic energy  E_k = ½ Σ m_i v_i²
        2. Update friction:  ξ ← ξ + (dt / 2Q)(2 E_k − g k_B T)
        3. Scale velocities:  v ← v · exp(−ξ dt / 2)

    Parameters
    ----------
    vel : ndarray [N, 3]  – Current velocities (reduced).
    xi_NH : float  – Current Nosé-Hoover friction variable.
    T_target : float  – Target temperature (reduced units, k_B = 1).
    Q : float  – Thermostat mass parameter  Q = g · T* · τ_NH*².
    dt : float  – Time step (reduced).
    N : int  – Number of atoms.
    masses : ndarray [N]  – Reduced masses.

    Returns
    -------
    vel_new : ndarray [N, 3]  – Scaled velocities.
    xi_NH_new : float  – Updated friction coefficient.
    """
    g = 3 * N  # degrees of freedom
    ek = 0.5 * np.sum(masses[:, np.newaxis] * vel ** 2)
    xi_NH = xi_NH + (dt / (2.0 * Q)) * (2.0 * ek - g * T_target)
    vel = vel * np.exp(-xi_NH * dt / 2.0)
    return vel, xi_NH


def nvt_velocity_verlet_step(
    pos: np.ndarray, vel: np.ndarray, forces: np.ndarray,
    xi_nh: float, dt: float, masses: np.ndarray,
    box_L: float, params: LJParams, species: np.ndarray,
    T_target_star: float, Q: float, N: int,
    use_torch: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    One complete NVT step: Nosé-Hoover + Velocity-Verlet.

    Symmetric Trotter decomposition:
        NH½ → kick½ → drift → forces → kick½ → NH½

    Returns  (pos, vel, forces, xi_nh, pe, virial)
    """
    m = masses[:, np.newaxis]

    # --- first NH half-step ---
    vel, xi_nh = nose_hoover_step(vel, xi_nh, T_target_star, Q, dt, N, masses)

    # --- VV: half-kick ---
    vel = vel + 0.5 * dt * forces / m

    # --- VV: drift + PBC ---
    pos = pos + dt * vel
    pos -= box_L * np.floor(pos / box_L)

    # --- new forces ---
    forces, pe, virial = compute_forces_and_energy(
        pos, species, box_L, params, use_torch=use_torch,
    )

    # --- VV: half-kick ---
    vel = vel + 0.5 * dt * forces / m

    # --- second NH half-step ---
    vel, xi_nh = nose_hoover_step(vel, xi_nh, T_target_star, Q, dt, N, masses)

    return pos, vel, forces, xi_nh, pe, virial


# ====================================================================
# 1.6  Main Simulation Loop
# ====================================================================

def run_md(config: Dict) -> Dict:
    """
    Run a full molecular dynamics simulation: initialisation → equilibration
    (NVT) → production (NVT or NVE), saving trajectory and thermo log.

    Parameters  (``config`` dict keys)
    -----------------------------------
    N_A, N_B          : int   – Number of Kr / Xe atoms.
    rho_star          : float – Reduced number density (default 0.8).
    T_equil_K         : float – Equilibration temperature [K].
    T_prod_K          : float – Production temperature [K].
    dt_fs             : float – Time step [fs] (default 2).
    n_equil           : int   – Equilibration steps.
    n_prod            : int   – Production steps.
    n_save            : int   – Save frequency during production.
    xi                : float – Mixing parameter (default 1.0).
    ensemble_prod     : str   – 'NVT' or 'NVE' for production phase.
    tau_nh_fs         : float – NH thermostat time constant [fs] (default 100).
    seed              : int   – Random seed (default 42).
    traj_file         : str   – Path for trajectory ``.npy``.
    thermo_file       : str   – Path for thermo log ``.csv``.
    use_torch         : bool  – Use PyTorch force backend (default True).

    Returns
    -------
    results : dict  – Summary statistics and arrays.
    """
    # ---- unpack with defaults ----
    N_A           = config.get('N_A', 256)
    N_B           = config.get('N_B', 0)
    N             = N_A + N_B
    rho_star      = config.get('rho_star', 0.8)
    T_equil_K     = config.get('T_equil_K', 120.0)
    T_prod_K      = config.get('T_prod_K', 120.0)
    dt_fs         = config.get('dt_fs', 2.0)
    n_equil       = config.get('n_equil', 5000)
    n_prod        = config.get('n_prod', 10000)
    n_save        = config.get('n_save', 50)
    xi_mix        = config.get('xi', 1.0)
    ensemble_prod = config.get('ensemble_prod', 'NVT')
    tau_nh_fs     = config.get('tau_nh_fs', 100.0)
    seed          = config.get('seed', 42)
    traj_file     = config.get('traj_file', None)
    thermo_file   = config.get('thermo_file', None)
    use_torch     = config.get('use_torch', True)

    # ---- reduced-unit conversions ----
    T_equil_star = T_equil_K / EPS_REF
    T_prod_star  = T_prod_K / EPS_REF
    dt_star      = (dt_fs * 1e-15) / TAU_REF
    tau_nh_star  = (tau_nh_fs * 1e-15) / TAU_REF

    # ---- LJ parameters ----
    params = make_default_params(xi=xi_mix)

    # ---- initialisation ----
    print(f"Initialising {N} atoms ({N_A} Kr + {N_B} Xe), ρ* = {rho_star:.2f}")
    x_B = N_B / N if N > 0 else 0.0
    pos, species, box_L = init_fcc_lattice(N_A, N_B, x_B, rho_star, seed)
    masses = get_reduced_masses(species)
    vel = init_velocities(N, T_equil_K, masses, seed + 1)

    # Thermostat mass:  Q = g · T* · τ_NH*²   (g = 3N)
    N_dof = 3 * N
    Q = N_dof * T_equil_star * tau_nh_star ** 2
    xi_nh = 0.0

    # Initial forces
    print("Computing initial forces …")
    forces, pe, virial = compute_forces_and_energy(
        pos, species, box_L, params, use_torch=use_torch,
    )

    print(f"  Box L       = {box_L:.4f} σ* = {box_L * SIGMA_REF:.4f} Å")
    print(f"  r_cut       = {params.r_cut:.4f} σ*   (2 r_cut = {2*params.r_cut:.4f} σ*)")
    print(f"  dt          = {dt_fs} fs = {dt_star:.6f} τ*")
    print(f"  τ_NH        = {tau_nh_fs} fs = {tau_nh_star:.6f} τ*")
    print(f"  Q           = {Q:.6f}")
    print(f"  Initial PE  = {pe:.4f} ε*")

    t_wall = time_module.time()

    # ==================================================================
    # EQUILIBRATION  (NVT, no data collection)
    # ==================================================================
    log_interval_eq = max(1, n_equil // 10)
    print(f"\n{'='*65}")
    print(f"EQUILIBRATION  — {n_equil} steps, T = {T_equil_K} K, NVT (Nosé-Hoover)")
    print(f"{'='*65}")

    for step in range(n_equil):
        pos, vel, forces, xi_nh, pe, virial = nvt_velocity_verlet_step(
            pos, vel, forces, xi_nh, dt_star, masses,
            box_L, params, species, T_equil_star, Q, N,
            use_torch=use_torch,
        )
        if step % log_interval_eq == 0:
            ek = 0.5 * np.sum(masses[:, np.newaxis] * vel ** 2)
            T_K = 2.0 * ek / (3.0 * N) * EPS_REF
            print(f"  Step {step:>7d}/{n_equil}  T = {T_K:7.2f} K  "
                  f"PE/N = {pe/N:9.5f} ε*  [{time_module.time()-t_wall:.1f}s]")

    # If production temperature differs, adjust thermostat
    if abs(T_prod_K - T_equil_K) > 0.1:
        Q = N_dof * T_prod_star * tau_nh_star ** 2
        print(f"\n  → Thermostat adjusted to T_prod = {T_prod_K} K  (Q = {Q:.6f})")

    # ==================================================================
    # PRODUCTION
    # ==================================================================
    log_interval_pr = max(1, n_prod // 10)
    print(f"\n{'='*65}")
    print(f"PRODUCTION     — {n_prod} steps, T = {T_prod_K} K, {ensemble_prod}")
    print(f"{'='*65}")

    trajectory = []
    thermo_data = []

    for step in range(n_prod):
        if ensemble_prod == 'NVT':
            pos, vel, forces, xi_nh, pe, virial = nvt_velocity_verlet_step(
                pos, vel, forces, xi_nh, dt_star, masses,
                box_L, params, species, T_prod_star, Q, N,
                use_torch=use_torch,
            )
        else:  # NVE
            pos, vel, forces, pe, virial = velocity_verlet_step(
                pos, vel, forces, dt_star, masses, box_L,
                params, species, use_torch=use_torch,
            )

        # Save snapshot
        if step % n_save == 0:
            ek = 0.5 * np.sum(masses[:, np.newaxis] * vel ** 2)
            T_star_inst = 2.0 * ek / (3.0 * N)
            E_tot = ek + pe
            V = box_L ** 3
            P = (N * T_star_inst + virial / 3.0) / V

            trajectory.append(pos.copy())
            thermo_data.append([step, T_star_inst * EPS_REF, ek, pe, E_tot, P])

        # Progress log
        if step % log_interval_pr == 0:
            ek = 0.5 * np.sum(masses[:, np.newaxis] * vel ** 2)
            T_K = 2.0 * ek / (3.0 * N) * EPS_REF
            E_tot = ek + pe
            print(f"  Step {step:>7d}/{n_prod}  T = {T_K:7.2f} K  "
                  f"E_tot/N = {E_tot/N:10.6f} ε*  [{time_module.time()-t_wall:.1f}s]")

    wall = time_module.time() - t_wall
    print(f"\nSimulation finished in {wall:.1f} s  "
          f"({wall / (n_equil + n_prod) * 1e3:.2f} ms/step)")

    # ---- arrays ----
    trajectory = np.array(trajectory)     # [n_frames, N, 3]
    thermo_data = np.array(thermo_data)   # [n_frames, 6]

    # ---- save ----
    if traj_file:
        os.makedirs(os.path.dirname(os.path.abspath(traj_file)), exist_ok=True)
        np.save(traj_file, trajectory)
        print(f"  Trajectory  → {traj_file}   shape {trajectory.shape}")

    if thermo_file:
        os.makedirs(os.path.dirname(os.path.abspath(thermo_file)), exist_ok=True)
        header = "step,T_K,E_kin,E_pot,E_tot,P"
        np.savetxt(thermo_file, thermo_data, delimiter=',',
                   header=header, comments='', fmt='%.8e')
        print(f"  Thermo log  → {thermo_file}   {len(thermo_data)} rows")

    # ---- summary statistics ----
    results: Dict = {
        'T_mean_K':   float(np.mean(thermo_data[:, 1])),
        'T_std_K':    float(np.std(thermo_data[:, 1])),
        'E_tot_mean': float(np.mean(thermo_data[:, 4])),
        'E_tot_std':  float(np.std(thermo_data[:, 4])),
        'PE_mean':    float(np.mean(thermo_data[:, 3])),
        'P_mean':     float(np.mean(thermo_data[:, 5])),
        'trajectory': trajectory,
        'thermo':     thermo_data,
        'species':    species,
        'box_L':      box_L,
        'params':     params,
    }

    print(f"\n  ⟨T⟩      = {results['T_mean_K']:.2f} ± {results['T_std_K']:.2f} K")
    print(f"  ⟨E_tot⟩/N = {results['E_tot_mean']/N:.6f} ε*")
    print(f"  σ(E_tot)/N = {results['E_tot_std']/N:.6f} ε*")

    if ensemble_prod == 'NVE':
        E0 = thermo_data[0, 4]
        E_end = thermo_data[-1, 4]
        drift_pct = (E_end - E0) / abs(E0) * 100.0
        print(f"  Energy drift  = {drift_pct:+.5f} %")
        results['E_drift_percent'] = drift_pct

    return results

