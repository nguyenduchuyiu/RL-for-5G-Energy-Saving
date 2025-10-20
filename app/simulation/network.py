"""Network entities: Sites, Cells, UEs"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Site:
    id: int
    x: float
    y: float
    type: str  # 'indoor_trxp', 'macro', 'micro', 'rural_macro', 'urban_macro'


@dataclass
class Cell:
    id: int
    site_id: int
    sector_id: int
    azimuth: float
    x: float
    y: float
    frequency: float
    antenna_height: float
    tx_power: float
    min_tx_power: float
    max_tx_power: float
    cell_radius: float
    cpu_usage: float = 0.0
    prb_usage: float = 0.0
    energy_consumption: float = 0.0
    base_energy_consumption: float = 0.0
    idle_energy_consumption: float = 0.0
    max_capacity: float = 0.0
    current_load: float = 0.0
    connected_ues: List[int] = field(default_factory=list)
    ttt: float = 8.0  # Time-to-trigger (ms)
    a3_offset: float = 8.0  # A3 offset (dB)
    is_omnidirectional: bool = False
    site_type: str = 'macro'
    avg_rsrp: float = 0.0
    avg_rsrq: float = 0.0
    avg_sinr: float = 0.0
    total_traffic_demand: float = 0.0
    load_ratio: float = 0.0
    drop_rate: float = 0.0
    avg_latency: float = 0.0


@dataclass
class Measurement:
    cell_id: int
    rsrp: float
    rsrq: float
    sinr: float


@dataclass
class HandoverEvent:
    ue_id: int
    cell_source: int
    cell_target: int
    rsrp_source: float
    rsrp_target: float
    rsrq_source: float
    rsrq_target: float
    sinr_source: float
    sinr_target: float
    a3_offset: float
    ttt: float
    ho_success: bool
    timestamp: float


@dataclass
class UE:
    id: int
    x: float
    y: float
    velocity: float
    direction: float
    mobility_pattern: str
    serving_cell: Optional[int] = None
    rsrp: Optional[float] = None
    rsrq: Optional[float] = None
    sinr: Optional[float] = None
    neighbor_measurements: List[Measurement] = field(default_factory=list)
    ho_timer: float = 0.0
    step_counter: int = 0
    last_direction_change: float = 0.0
    pause_timer: float = 0.0
    connection_timer: float = 0.0
    disconnection_timer: float = 0.0
    last_serving_rsrp: Optional[float] = None
    traffic_demand: float = 0.0
    qos_latency: float = 0.0
    session_active: bool = False
    drop_count: int = 0
    rng_seed: int = 0
    deployment_scenario: str = ''
    handover_history: List[HandoverEvent] = field(default_factory=list)

