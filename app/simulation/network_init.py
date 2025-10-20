"""Network initialization - ported from createLayout.m, configureCells.m, initializeUEs.m"""

import numpy as np
from typing import List, Tuple
from .network import Site, Cell, UE, Measurement
from .scenario_loader import SimParams


def create_layout(sim_params: SimParams, seed: int) -> List[Site]:
    """Create site layout based on scenario"""
    
    rng = np.random.RandomState(seed + 1000)
    
    scenario = sim_params.deployment_scenario
    
    if scenario == 'indoor_hotspot':
        sites = _create_indoor_layout(sim_params, seed, rng)
    elif scenario == 'dense_urban':
        sites = _create_dense_urban_layout(sim_params, seed, rng)
    elif scenario == 'rural':
        sites = _create_rural_layout(sim_params, seed, rng)
    elif scenario == 'urban_macro':
        sites = _create_urban_macro_layout(sim_params, seed, rng)
    else:
        sites = _create_hex_layout(sim_params.num_sites, sim_params.isd, seed, rng)
    
    print(f'Created {len(sites)} sites for {scenario} scenario')
    return sites


def _create_indoor_layout(sim_params: SimParams, seed: int, rng: np.random.RandomState) -> List[Site]:
    """Indoor hotspot layout with TRxPs in 120m x 50m area"""
    floor_width = 120
    floor_height = 50
    cols = 4
    rows = 3
    
    x_spacing = floor_width / (cols + 1)
    y_spacing = floor_height / (rows + 1)
    
    sites = []
    site_id = 1
    
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            if site_id <= sim_params.num_sites:
                sites.append(Site(
                    id=site_id,
                    x=col * x_spacing,
                    y=row * y_spacing,
                    type='indoor_trxp'
                ))
                site_id += 1
    
    return sites


def _create_dense_urban_layout(sim_params: SimParams, seed: int, rng: np.random.RandomState) -> List[Site]:
    """Dense urban layout with macro sites"""
    return _create_hex_layout(sim_params.num_sites, sim_params.isd, seed, rng)


def _create_rural_layout(sim_params: SimParams, seed: int, rng: np.random.RandomState) -> List[Site]:
    """Rural layout with widely spaced macro sites"""
    sites = _create_hex_layout(sim_params.num_sites, sim_params.isd, seed, rng)
    for site in sites:
        site.type = 'rural_macro'
    return sites


def _create_urban_macro_layout(sim_params: SimParams, seed: int, rng: np.random.RandomState) -> List[Site]:
    """Urban macro layout"""
    sites = _create_hex_layout(sim_params.num_sites, sim_params.isd, seed, rng)
    for site in sites:
        site.type = 'urban_macro'
    return sites


def _create_hex_layout(num_sites: int, isd: float, seed: int, rng: np.random.RandomState) -> List[Site]:
    """Create hexagonal layout pattern"""
    
    sites = []
    
    # Central site
    sites.append(Site(id=1, x=0, y=0, type='macro'))
    
    if num_sites == 1:
        return sites
    
    # Create concentric hexagonal rings
    site_id = 2
    ring = 1
    max_rings = 5
    
    while site_id <= num_sites and ring <= max_rings:
        ring_sites = _create_hex_ring(ring, isd, site_id)
        for site in ring_sites:
            if site_id <= num_sites:
                sites.append(site)
                site_id += 1
        ring += 1
    
    # Fill remaining sites randomly if needed
    while site_id <= num_sites:
        angle = rng.rand() * 2 * np.pi
        distance = isd + rng.rand() * (isd * 2)
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        sites.append(Site(id=site_id, x=x, y=y, type='macro'))
        site_id += 1
    
    return sites


def _create_hex_ring(ring: int, isd: float, start_id: int) -> List[Site]:
    """Create sites in a hexagonal ring"""
    
    sites = []
    site_id = start_id
    
    for side in range(6):
        for pos in range(ring):
            angle = side * np.pi / 3
            x = isd * ring * np.cos(angle) + pos * isd * np.cos(angle + np.pi/3)
            y = isd * ring * np.sin(angle) + pos * isd * np.sin(angle + np.pi/3)
            sites.append(Site(id=site_id, x=x, y=y, type='macro'))
            site_id += 1
    
    return sites


def configure_cells(sites: List[Site], sim_params: SimParams) -> List[Cell]:
    """Configure cells based on 3GPP scenario parameters"""
    
    cells = []
    cell_id = 1
    
    print(f'Configuring cells for {sim_params.deployment_scenario} scenario...')
    
    for site in sites:
        # Get cell configuration
        cell_config = _get_cell_config_for_site(site, sim_params)
        
        # Determine number of sectors
        num_sectors = _determine_num_sectors(site, sim_params)
        
        # Create sectors
        for sector_id in range(1, num_sectors + 1):
            azimuth = (sector_id - 1) * (360 / num_sectors)
            is_omnidirectional = (num_sectors == 1)
            
            cell = Cell(
                id=cell_id,
                site_id=site.id,
                sector_id=sector_id,
                azimuth=azimuth,
                x=site.x,
                y=site.y,
                frequency=cell_config['frequency'],
                antenna_height=cell_config['antenna_height'],
                tx_power=cell_config['initial_tx_power'],
                min_tx_power=cell_config['min_tx_power'],
                max_tx_power=cell_config['max_tx_power'],
                cell_radius=cell_config['cell_radius'],
                base_energy_consumption=cell_config['base_power'],
                idle_energy_consumption=cell_config['idle_power'],
                energy_consumption=cell_config['base_power'],
                max_capacity=cell_config['max_capacity'],
                ttt=cell_config.get('ttt', 8),
                a3_offset=cell_config.get('a3_offset', 8),
                is_omnidirectional=is_omnidirectional,
                site_type=site.type
            )
            cells.append(cell)
            cell_id += 1
    
    print(f'Configured {len(cells)} cells for {sim_params.deployment_scenario} scenario')
    return cells


def _determine_num_sectors(site: Site, sim_params: SimParams) -> int:
    """Determine number of sectors based on site type"""
    
    if sim_params.deployment_scenario == 'indoor_hotspot':
        return 1
    elif site.type == 'micro':
        return 1
    else:
        return sim_params.num_sectors


def _get_cell_config_for_site(site: Site, sim_params: SimParams) -> dict:
    """Get cell configuration based on site type"""
    
    if site.type == 'indoor_trxp':
        return _get_indoor_cell_config(sim_params)
    elif site.type == 'rural_macro':
        return _get_rural_macro_cell_config(sim_params)
    elif site.type == 'urban_macro':
        return _get_urban_macro_cell_config(sim_params)
    elif site.type == 'micro':
        return _get_micro_cell_config(sim_params)
    else:  # macro
        return _get_macro_cell_config(sim_params)


def _get_indoor_cell_config(sim_params: SimParams) -> dict:
    """Indoor hotspot cell configuration"""
    return {
        'frequency': sim_params.carrier_frequency,
        'antenna_height': 3,
        'initial_tx_power': 23,
        'min_tx_power': sim_params.min_tx_power if sim_params.min_tx_power < 30 else 20,
        'max_tx_power': sim_params.max_tx_power if sim_params.max_tx_power < 40 else 30,
        'cell_radius': 50,
        'base_power': 400,
        'idle_power': 100,
        'max_capacity': 50,
        'ttt': 4,
        'a3_offset': 6
    }


def _get_macro_cell_config(sim_params: SimParams) -> dict:
    """Dense urban macro cell configuration"""
    return {
        'frequency': sim_params.carrier_frequency,
        'antenna_height': sim_params.antenna_height,
        'initial_tx_power': 43,
        'min_tx_power': sim_params.min_tx_power,
        'max_tx_power': sim_params.max_tx_power,
        'cell_radius': sim_params.cell_radius,
        'base_power': sim_params.base_power,
        'idle_power': sim_params.idle_power,
        'max_capacity': 200,
        'ttt': 8,
        'a3_offset': 8
    }


def _get_micro_cell_config(sim_params: SimParams) -> dict:
    """Micro cell configuration"""
    return {
        'frequency': sim_params.carrier_frequency,
        'antenna_height': 10,
        'initial_tx_power': 30,
        'min_tx_power': 20,
        'max_tx_power': 38,
        'cell_radius': 50,
        'base_power': 200,
        'idle_power': 50,
        'max_capacity': 100,
        'ttt': 6,
        'a3_offset': 6
    }


def _get_rural_macro_cell_config(sim_params: SimParams) -> dict:
    """Rural macro cell configuration"""
    return {
        'frequency': sim_params.carrier_frequency,
        'antenna_height': 35,
        'initial_tx_power': 46,
        'min_tx_power': sim_params.min_tx_power if sim_params.min_tx_power > 30 else 35,
        'max_tx_power': sim_params.max_tx_power if sim_params.max_tx_power > 46 else 49,
        'cell_radius': 1000,
        'base_power': 1200,
        'idle_power': 300,
        'max_capacity': 150,
        'ttt': 12,
        'a3_offset': 10
    }


def _get_urban_macro_cell_config(sim_params: SimParams) -> dict:
    """Urban macro cell configuration"""
    return {
        'frequency': sim_params.carrier_frequency,
        'antenna_height': sim_params.antenna_height,
        'initial_tx_power': 43,
        'min_tx_power': sim_params.min_tx_power,
        'max_tx_power': sim_params.max_tx_power,
        'cell_radius': 300,
        'base_power': 1000,
        'idle_power': 250,
        'max_capacity': 250,
        'ttt': 8,
        'a3_offset': 8
    }


def initialize_ues(sim_params: SimParams, sites: List[Site], seed: int) -> List[UE]:
    """Initialize UEs based on 3GPP scenario requirements"""
    
    rng = np.random.RandomState(seed + 2000)
    
    scenario = sim_params.deployment_scenario
    
    if scenario == 'indoor_hotspot':
        ues = _initialize_indoor_hotspot_ues(sim_params, sites, seed, rng)
    elif scenario == 'dense_urban':
        ues = _initialize_dense_urban_ues(sim_params, sites, seed, rng)
    elif scenario == 'rural':
        ues = _initialize_rural_ues(sim_params, sites, seed, rng)
    elif scenario == 'urban_macro':
        ues = _initialize_urban_macro_ues(sim_params, sites, seed, rng)
    else:
        ues = _initialize_default_ues(sim_params, sites, seed, rng)
    
    print(f'Initialized {len(ues)} UEs for {scenario} scenario')
    return ues


def _initialize_indoor_hotspot_ues(sim_params: SimParams, sites: List[Site], seed: int, rng: np.random.RandomState) -> List[UE]:
    """Initialize UEs for indoor hotspot scenario"""
    
    ues = []
    patterns = ['stationary', 'slow_walk', 'normal_walk']
    velocities = [0, 0.5, 1.5]
    weights = [0.4, 0.4, 0.2]
    
    bounds = {'min_x': 10, 'max_x': 110, 'min_y': 5, 'max_y': 45}
    
    for ue_id in range(1, sim_params.num_ues + 1):
        # Generate position
        x = rng.uniform(bounds['min_x'], bounds['max_x'])
        y = rng.uniform(bounds['min_y'], bounds['max_y'])
        
        # Select mobility pattern
        pattern_idx = rng.choice(len(patterns), p=weights)
        velocity = velocities[pattern_idx]
        direction = rng.uniform(0, 2 * np.pi)
        
        ues.append(UE(
            id=ue_id,
            x=x,
            y=y,
            velocity=velocity,
            direction=direction,
            mobility_pattern=patterns[pattern_idx],
            rng_seed=seed + ue_id * 100,
            deployment_scenario=sim_params.deployment_scenario
        ))
    
    return ues


def _initialize_dense_urban_ues(sim_params: SimParams, sites: List[Site], seed: int, rng: np.random.RandomState) -> List[UE]:
    """Initialize UEs for dense urban scenario"""
    
    indoor_ues = int(sim_params.num_ues * sim_params.indoor_ratio)
    outdoor_ues = sim_params.num_ues - indoor_ues
    
    ues = []
    ue_id = 1
    
    # Indoor UEs
    for _ in range(indoor_ues):
        site_idx = rng.randint(0, len(sites))
        site = sites[site_idx]
        
        angle = rng.uniform(0, 2 * np.pi)
        distance = abs(rng.randn()) * 30
        
        x = site.x + distance * np.cos(angle)
        y = site.y + distance * np.sin(angle)
        velocity = sim_params.ue_speed / 3.6
        direction = rng.uniform(0, 2 * np.pi)
        
        ues.append(UE(
            id=ue_id,
            x=x,
            y=y,
            velocity=velocity,
            direction=direction,
            mobility_pattern='indoor_pedestrian',
            rng_seed=seed + ue_id * 100,
            deployment_scenario=sim_params.deployment_scenario
        ))
        ue_id += 1
    
    # Outdoor UEs
    for _ in range(outdoor_ues):
        site_idx = rng.randint(0, len(sites))
        site = sites[site_idx]
        
        angle = rng.uniform(0, 2 * np.pi)
        distance = rng.uniform(50, 150)
        
        x = site.x + distance * np.cos(angle)
        y = site.y + distance * np.sin(angle)
        velocity = sim_params.outdoor_speed / 3.6
        direction = rng.uniform(0, 2 * np.pi)
        
        ues.append(UE(
            id=ue_id,
            x=x,
            y=y,
            velocity=velocity,
            direction=direction,
            mobility_pattern='outdoor_vehicle',
            rng_seed=seed + ue_id * 100,
            deployment_scenario=sim_params.deployment_scenario
        ))
        ue_id += 1
    
    return ues


def _initialize_rural_ues(sim_params: SimParams, sites: List[Site], seed: int, rng: np.random.RandomState) -> List[UE]:
    """Initialize UEs for rural scenario"""
    
    ues = []
    patterns = ['stationary', 'pedestrian', 'slow_vehicle', 'fast_vehicle']
    velocities = [0, 1.0, sim_params.ue_speed/3.6, sim_params.ue_speed/3.6]
    weights = [0.1, 0.4, 0.3, 0.2]
    
    max_radius = sim_params.isd * 3
    cluster_prob = 0.6
    cluster_radius = 200
    
    for ue_id in range(1, sim_params.num_ues + 1):
        if rng.rand() < cluster_prob:
            # Clustered around a site
            site_idx = rng.randint(0, len(sites))
            site = sites[site_idx]
            angle = rng.uniform(0, 2 * np.pi)
            distance = rng.uniform(0, cluster_radius)
            x = site.x + distance * np.cos(angle)
            y = site.y + distance * np.sin(angle)
        else:
            # Uniform distribution
            angle = rng.uniform(0, 2 * np.pi)
            radius = max_radius * np.sqrt(rng.rand())
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
        
        pattern_idx = rng.choice(len(patterns), p=weights)
        velocity = velocities[pattern_idx]
        direction = rng.uniform(0, 2 * np.pi)
        
        ues.append(UE(
            id=ue_id,
            x=x,
            y=y,
            velocity=velocity,
            direction=direction,
            mobility_pattern=patterns[pattern_idx],
            rng_seed=seed + ue_id * 100,
            deployment_scenario=sim_params.deployment_scenario
        ))
    
    return ues


def _initialize_urban_macro_ues(sim_params: SimParams, sites: List[Site], seed: int, rng: np.random.RandomState) -> List[UE]:
    """Initialize UEs for urban macro scenario"""
    
    ues = []
    patterns = ['pedestrian', 'slow_vehicle', 'vehicle']
    velocities = [1.5, sim_params.ue_speed/3.6, sim_params.ue_speed/3.6]
    weights = [0.6, 0.2, 0.2]
    
    max_radius = sim_params.cell_radius * 1.5
    
    for ue_id in range(1, sim_params.num_ues + 1):
        if rng.rand() < sim_params.indoor_ratio:
            # Indoor positioning
            site_idx = rng.randint(0, len(sites))
            site = sites[site_idx]
            angle = rng.uniform(0, 2 * np.pi)
            distance = abs(rng.randn()) * (max_radius * 0.3)
            x = site.x + distance * np.cos(angle)
            y = site.y + distance * np.sin(angle)
        else:
            # Outdoor positioning
            angle = rng.uniform(0, 2 * np.pi)
            radius = max_radius * np.sqrt(rng.rand())
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
        
        pattern_idx = rng.choice(len(patterns), p=weights)
        velocity = velocities[pattern_idx]
        direction = rng.uniform(0, 2 * np.pi)
        
        ues.append(UE(
            id=ue_id,
            x=x,
            y=y,
            velocity=velocity,
            direction=direction,
            mobility_pattern=patterns[pattern_idx],
            rng_seed=seed + ue_id * 100,
            deployment_scenario=sim_params.deployment_scenario
        ))
    
    return ues


def _initialize_default_ues(sim_params: SimParams, sites: List[Site], seed: int, rng: np.random.RandomState) -> List[UE]:
    """Initialize UEs for default scenario"""
    
    ues = []
    patterns = ['stationary', 'pedestrian', 'slow_vehicle', 'fast_vehicle', 'vehicle']
    velocities = [0, 1.5, 5.0, 15.0, 10.0]
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    max_radius = sim_params.isd * np.sqrt(sim_params.num_sites) / (2 * np.pi)
    
    for ue_id in range(1, sim_params.num_ues + 1):
        angle = rng.uniform(0, 2 * np.pi)
        radius = max_radius * np.sqrt(rng.rand())
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        pattern_idx = rng.choice(len(patterns), p=weights)
        velocity = velocities[pattern_idx]
        direction = rng.uniform(0, 2 * np.pi)
        
        ues.append(UE(
            id=ue_id,
            x=x,
            y=y,
            velocity=velocity,
            direction=direction,
            mobility_pattern=patterns[pattern_idx],
            rng_seed=seed + ue_id * 100,
            deployment_scenario=sim_params.deployment_scenario
        ))
    
    return ues

