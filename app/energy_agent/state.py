class State:
    """State description"""
    
    def __init__(self, cells, max_cells):

        self.cells = cells
        self.max_cells = max_cells

        self.simulation_features = {
            'totalCells'               # number of cells
            'totalUEs'                # number of UEs
            'simTime'                # simulation time
            'timeStep'                # time step
            'timeProgress'                # progress ratio
            'carrierFrequency'                # frequency Hz
            'isd'                # inter-site distance
            'minTxPower'                # dBm
            'maxTxPower'                # dBm
            'basePower'                # watts
            'idlePower'                # watts
            'dropCallThreshold'                # percentage
            'latencyThreshold'                # ms
            'cpuThreshold'                # percentage
            'prbThreshold'                # percentage
            'trafficLambda'                # traffic rate
            'peakHourMultiplier'                # multiplier
        }
        
        self.network_features = {
            'totalEnergy'           # kWh
            'activeCells'              # number of cells
            'avgDropRate'              # percentage
            'avgLatency'              # ms
            'totalTraffic'              # traffic units
            'connectedUEs'              # number of UEs
            'connectionRate'              # percentage
            'cpuViolations'              # number of violations
            'prbViolations'              # number of violations
            'maxCpuUsage'              # percentage
            'maxPrbUsage'              # percentage
            'kpiViolations'              # number of violations
            'totalTxPower'              # total power
            'avgPowerRatio'              # ratio
        }
        
        self.cell_features = {
            'cpuUsage'              # percentage
            'prbUsage'              # percentage
            'currentLoad'              # load units
            'maxCapacity'              # capacity units
            'numConnectedUEs'              # number of UEs
            'txPower'              # dBm
            'energyConsumption'              # watts
            'avgRSRP'              # dBm
            'avgRSRQ'              # dB
            'avgSINR'              # dB
            'totalTrafficDemand'              # traffic units
            'loadRatio'              # ratio
        }
        
        ''' 
        all cells are normalized and padded to max_cells length
        state layout:
        [ sim_f1, sim_f2, ..., sim_f17, 
         net_f1, net_f2, ..., net_f14, 
         cell1_f1, cell2_f1, ..., cell_max_f1, 
         cell1_f2, cell2_f2, ..., cell_max_f2, 
         ...
         cell1_f12, cell2_f12, ..., cell_max_f12,
         ue_density, isd, base_power,dist_to_drop_thresh, dist_to_latency_thresh, dist_to_cpu_thresh, dist_to_prb_thresh, load_per_active_cell, power_efficiency,
         load_delta_cell_1, load_delta_cell_2, ..., load_delta_cell_max,
         ue_delta_cell_1, ue_delta_cell_2, ..., ue_delta_cell_max,        
         ]
        '''            