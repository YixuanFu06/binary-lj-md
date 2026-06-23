import numpy as np

def _unwrapped_trajectory(traj, box_L):
    """
    Unwrap trajectory to remove periodic boundary jumps.
    traj: [n_frames, N, 3]
    """
    n_frames = len(traj)
    unwrapped = np.zeros_like(traj)
    unwrapped[0] = traj[0]
    
    for t in range(1, n_frames):
        dr = traj[t] - traj[t-1]
        dr -= box_L * np.round(dr / box_L)
        unwrapped[t] = unwrapped[t-1] + dr
        
    return unwrapped

def compute_partial_rdf(traj, species, box_L, dr=0.05, r_max=10.0):
    """
    Compute partial radial distribution functions gAA, gBB, gAB.
    Uses only the last 60% of the trajectory (production phase).
    """
    n_frames = len(traj)
    start_idx = int(n_frames * 0.4)
    traj_prod = traj[start_idx:]
    n_prod = len(traj_prod)
    
    # Cap r_max to box_L / 2 to avoid PBC artifacts
    r_max = min(r_max, box_L / 2.0)
    
    N = len(species)
    idx_A = np.where(species == 0)[0]
    idx_B = np.where(species == 1)[0]
    N_A = len(idx_A)
    N_B = len(idx_B)
    V = box_L ** 3
    
    nbins = int(np.ceil(r_max / dr))
    r_edges = np.linspace(0, r_max, nbins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    
    hist_AA = np.zeros(nbins)
    hist_BB = np.zeros(nbins)
    hist_AB = np.zeros(nbins)
    
    for pos in traj_prod:
        dr_mat = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dr_mat -= box_L * np.round(dr_mat / box_L)
        dist = np.linalg.norm(dr_mat, axis=2)
        
        np.fill_diagonal(dist, np.inf)
        
        dist_AA = dist[np.ix_(idx_A, idx_A)].flatten()
        dist_BB = dist[np.ix_(idx_B, idx_B)].flatten()
        dist_AB = dist[np.ix_(idx_A, idx_B)].flatten()
        
        h_AA, _ = np.histogram(dist_AA, bins=r_edges)
        h_BB, _ = np.histogram(dist_BB, bins=r_edges)
        h_AB, _ = np.histogram(dist_AB, bins=r_edges)
        
        hist_AA += h_AA
        hist_BB += h_BB
        hist_AB += h_AB
        
    hist_AA = hist_AA / n_prod
    hist_BB = hist_BB / n_prod
    hist_AB = hist_AB / n_prod
    
    pairs_AA = N_A * N_A if N_A > 0 else 1
    pairs_BB = N_B * N_B if N_B > 0 else 1
    pairs_AB = N_A * N_B if (N_A > 0 and N_B > 0) else 1
    
    shell_vol = 4.0 * np.pi * r_centers**2 * dr
    
    gAA = (V / pairs_AA) * hist_AA / shell_vol
    gBB = (V / pairs_BB) * hist_BB / shell_vol
    gAB = (V / pairs_AB) * hist_AB / shell_vol
    
    return r_centers, gAA, gBB, gAB

def compute_warren_cowley(traj, species, box_L, r_cut_wc):
    """
    Compute Warren-Cowley short-range order parameter alpha_1(t).
    """
    n_frames = len(traj)
    
    N = len(species)
    idx_A = np.where(species == 0)[0]
    idx_B = np.where(species == 1)[0]
    N_A = len(idx_A)
    N_B = len(idx_B)
    
    if N_A == 0 or N_B == 0:
        return np.zeros(n_frames)
        
    x_B = N_B / N
    alpha_1 = np.zeros(n_frames)
    
    for t in range(n_frames):
        pos = traj[t]
        dr_mat = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dr_mat -= box_L * np.round(dr_mat / box_L)
        dist = np.linalg.norm(dr_mat, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        mask = dist < r_cut_wc
        
        n_B_around_A = np.sum(mask[np.ix_(idx_A, idx_B)], axis=1)
        Z_AB = np.mean(n_B_around_A)
        
        n_total_around_A = np.sum(mask[np.ix_(idx_A, np.arange(N))], axis=1)
        Z = np.mean(n_total_around_A)
        
        if Z > 0:
            alpha_1[t] = 1.0 - Z_AB / (x_B * Z)
        else:
            alpha_1[t] = 0.0
            
    return alpha_1

def compute_partial_msd(traj, species, box_L):
    """
    Compute partial Mean Square Displacement using multiple time origins.
    """
    n_frames = len(traj)
    unwrapped = _unwrapped_trajectory(traj, box_L)
    
    idx_A = np.where(species == 0)[0]
    idx_B = np.where(species == 1)[0]
    
    msd_A = np.zeros(n_frames)
    msd_B = np.zeros(n_frames)
    
    for t in range(n_frames):
        # difference between separated frames: shape [N_origins, N, 3]
        diff = unwrapped[t:] - unwrapped[:n_frames-t]
        sq_dist = np.sum(diff**2, axis=2) # [N_origins, N]
        
        if len(idx_A) > 0:
            msd_A[t] = np.mean(sq_dist[:, idx_A])
        if len(idx_B) > 0:
            msd_B[t] = np.mean(sq_dist[:, idx_B])
            
    return msd_A, msd_B

def compute_mixing_enthalpy(epot_mix_array, epot_pureA, epot_pureB, x_B_values):
    """
    Computes mixing enthalpy per atom.
    epot_mix_array: array of mean potential energy per atom for each x_B
    epot_pureA: mean potential energy per atom for pure A
    epot_pureB: mean potential energy per atom for pure B
    """
    dH_mix = np.zeros(len(x_B_values))
    for i, x_B in enumerate(x_B_values):
        dH_mix[i] = epot_mix_array[i] - (1.0 - x_B) * epot_pureA - x_B * epot_pureB
    return dH_mix

def compute_heat_capacity(E_tot_array, T_star, N):
    """
    Computes heat capacity C_v / (N k_B) from total energy fluctuations in NVT ensemble.
    E_tot_array: array of total energies in reduced units (epsilon*)
    T_star: reduced temperature
    """
    var_E = np.var(E_tot_array, ddof=1)
    cv_per_atom_kb = var_E / (N * T_star**2)
    return cv_per_atom_kb

def get_q_vectors(box_L, q_max):
    n_max = int(q_max * box_L / (2 * np.pi))
    n_grid = np.arange(-n_max, n_max + 1)
    nx, ny, nz = np.meshgrid(n_grid, n_grid, n_grid, indexing='ij')
    n_vecs = np.vstack([nx.ravel(), ny.ravel(), nz.ravel()]).T
    
    q_vecs = 2.0 * np.pi / box_L * n_vecs
    q_mags = np.linalg.norm(q_vecs, axis=1)
    
    mask = (q_mags > 1e-6) & (q_mags <= q_max)
    return q_vecs[mask], q_mags[mask]

def compute_structure_factor(traj, species, box_L, q_max=15.0, bins=50):
    """
    Computes S(q) directly from trajectory by averaging |rho(q)|^2
    Uses downsampled q-vectors and time frames to accelerate computation.
    """
    n_frames = len(traj)
    start_idx = int(n_frames * 0.4)
    traj_prod = traj[start_idx::5]  
    
    q_vecs, q_mags = get_q_vectors(box_L, q_max)
    
    q_edges = np.linspace(0, q_max, bins + 1)
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    
    shell_indices = np.digitize(q_mags, q_edges) - 1
    
    # Cap the number of q-vectors per shell to 200 for performance
    selected_q_idx = []
    for i in range(bins):
        idx_in_shell = np.where(shell_indices == i)[0]
        if len(idx_in_shell) > 200:
            idx_in_shell = np.random.choice(idx_in_shell, 200, replace=False)
        selected_q_idx.extend(idx_in_shell)
    
    q_vecs = q_vecs[selected_q_idx]
    q_mags = q_mags[selected_q_idx]
    shell_indices = shell_indices[selected_q_idx]
    
    N = traj.shape[1]
    idx_A = np.where(species == 0)[0]
    idx_B = np.where(species == 1)[0]
    N_A = len(idx_A)
    N_B = len(idx_B)
    
    S_tot = np.zeros(bins)
    S_AA = np.zeros(bins)
    S_BB = np.zeros(bins)
    S_AB = np.zeros(bins)
    counts = np.zeros(bins)
    
    for i in shell_indices:
        if 0 <= i < bins:
            counts[i] += 1
            
    for pos in traj_prod:
        qr = np.dot(q_vecs, pos.T) 
        
        rho_tot = np.sum(np.exp(-1j * qr), axis=1)
        rho_A = np.sum(np.exp(-1j * qr[:, idx_A]), axis=1)
        rho_B = np.sum(np.exp(-1j * qr[:, idx_B]), axis=1)
        
        sq_tot = np.abs(rho_tot)**2 / N
        sq_AA = np.abs(rho_A)**2 / N_A if N_A > 0 else np.zeros_like(sq_tot)
        sq_BB = np.abs(rho_B)**2 / N_B if N_B > 0 else np.zeros_like(sq_tot)
        sq_AB = np.real(rho_A * np.conj(rho_B)) / np.sqrt(N_A * N_B) if (N_A > 0 and N_B > 0) else np.zeros_like(sq_tot)
        
        valid = (shell_indices >= 0) & (shell_indices < bins)
        S_tot += np.bincount(shell_indices[valid], weights=sq_tot[valid], minlength=bins)
        S_AA += np.bincount(shell_indices[valid], weights=sq_AA[valid], minlength=bins)
        S_BB += np.bincount(shell_indices[valid], weights=sq_BB[valid], minlength=bins)
        S_AB += np.bincount(shell_indices[valid], weights=sq_AB[valid], minlength=bins)
        
    n_frames_sampled = len(traj_prod)
    valid_bins = counts > 0
    S_tot[valid_bins] /= (counts[valid_bins] * n_frames_sampled)
    S_AA[valid_bins] /= (counts[valid_bins] * n_frames_sampled)
    S_BB[valid_bins] /= (counts[valid_bins] * n_frames_sampled)
    S_AB[valid_bins] /= (counts[valid_bins] * n_frames_sampled)
    
    return q_centers[valid_bins], S_tot[valid_bins], S_AA[valid_bins], S_BB[valid_bins], S_AB[valid_bins]
