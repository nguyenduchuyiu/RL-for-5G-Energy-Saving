"""Path loss calculation - ported from calculatePathLoss.m"""

import numpy as np


def calculate_path_loss(distance: float, frequency: float, ue_id: int, current_time: float, seed: int) -> float:
    """
    Calculate path loss using 3GPP model
    
    Args:
        distance: Distance in meters
        frequency: Carrier frequency in Hz
        ue_id: UE identifier for RNG seeding
        current_time: Current simulation time
        seed: Base RNG seed
    
    Returns:
        Path loss in dB
    """
    
    # Set up RNG with deterministic seed
    pl_seed = seed + 5000 + ue_id + int(current_time * 100)
    rng = np.random.RandomState(pl_seed)
    
    # Minimum distance
    if distance < 10:
        distance = 10
    
    # Frequency in GHz
    fc = frequency / 1e9
    
    # Antenna heights
    h_bs = 25  # Base station height (m)
    h_ut = 1.5  # UE height (m)
    
    # Calculate LOS probability (3GPP model)
    if distance <= 18:
        p_los = 1.0
    else:
        p_los = 18/distance + np.exp(-distance/36) * (1 - 18/distance)
    
    # Determine LOS or NLOS
    if rng.rand() < p_los:
        # LOS path loss
        path_loss = 32.4 + 21*np.log10(distance) + 20*np.log10(fc)
    else:
        # NLOS path loss
        path_loss = 35.3*np.log10(distance) + 22.4 + 21.3*np.log10(fc) - 0.3*(h_ut - 1.5)
    
    # Add shadow fading
    shadow_fading = rng.randn() * 4
    path_loss = path_loss + shadow_fading
    
    return path_loss


def calculate_path_loss_vectorized(distances: np.ndarray, frequencies: np.ndarray, 
                                   ue_ids: np.ndarray, cell_ids: np.ndarray,
                                   current_time: float, seed: int) -> np.ndarray:
    """
    Vectorized path loss calculation for multiple UE-Cell pairs
    
    Args:
        distances: Distance matrix (n_ues, n_cells) in meters
        frequencies: Cell frequencies array (n_cells,) in Hz
        ue_ids: UE IDs array (n_ues,)
        cell_ids: Cell IDs array (n_cells,)
        current_time: Current simulation time
        seed: Base RNG seed
    
    Returns:
        Path loss matrix (n_ues, n_cells) in dB
    """
    n_ues, n_cells = distances.shape
    
    # Minimum distance
    distances = np.maximum(distances, 10.0)
    
    # Frequency in GHz - broadcast to match distances shape
    fc = frequencies / 1e9  # shape (n_cells,)
    
    # Antenna heights
    h_ut = 1.5
    
    # Calculate LOS probability (3GPP model) - vectorized
    p_los = np.where(
        distances <= 18,
        1.0,
        18/distances + np.exp(-distances/36) * (1 - 18/distances)
    )
    
    # Generate random numbers for LOS/NLOS decision and shadow fading
    # Use deterministic seeding for reproducibility
    path_loss = np.zeros((n_ues, n_cells))
    
    for i in range(n_ues):
        for j in range(n_cells):
            pl_seed = seed + 5000 + int(ue_ids[i]) + int(current_time * 100) + int(cell_ids[j])
            rng = np.random.RandomState(pl_seed)
            
            # LOS/NLOS decision
            is_los = rng.rand() < p_los[i, j]
            
            if is_los:
                # LOS path loss
                pl = 32.4 + 21*np.log10(distances[i, j]) + 20*np.log10(fc[j])
            else:
                # NLOS path loss
                pl = 35.3*np.log10(distances[i, j]) + 22.4 + 21.3*np.log10(fc[j]) - 0.3*(h_ut - 1.5)
            
            # Shadow fading
            shadow_fading = rng.randn() * 4
            path_loss[i, j] = pl + shadow_fading
    
    return path_loss

