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
