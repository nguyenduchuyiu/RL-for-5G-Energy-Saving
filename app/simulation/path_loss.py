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

