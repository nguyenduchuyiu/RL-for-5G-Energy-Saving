# energy_agent/state_normalizer.py

import numpy as np

class StateNormalizer:
    """
    Chuẩn hóa vector trạng thái về khoảng [0, 1] sử dụng Min-Max scaling.
    Các giá trị min/max được định nghĩa dựa trên tài liệu 3GPP cho từng kịch bản.
    """
    def __init__(self, state_dim, n_cells=57, **kwargs):
        self.state_dim = state_dim
        self.n_cells = n_cells # Số cell tối đa có thể có

        # Các đặc trưng mô phỏng (17 features)
        # Các giá trị được lấy từ các kịch bản, chọn khoảng rộng nhất có thể
        self.simulation_bounds = {
            'totalCells': [1, 57],
            'totalUEs': [1, 300],
            'simTime': [300, 3600],
            'timeStep': [1, 10],
            'timeProgress': [0, 1],
            'carrierFrequency': [700e6, 4e9], # Rural (low) to Indoor/Urban (high)
            'isd': [20, 1732], # Indoor (low) to Rural (high)
            'minTxPower': [10, 35],
            'maxTxPower': [23, 49],
            'basePower': [50, 1200],
            'idlePower': [15, 300],
            'dropCallThreshold': [1, 5], # Thường là 1-2%, cho khoảng rộng hơn
            'latencyThreshold': [50, 100],
            'cpuThreshold': [90, 95],
            'prbThreshold': [90, 95],
            'trafficLambda': [10, 25],
            'peakHourMultiplier': [1.2, 1.5]
        }
        
        # Các đặc trưng mạng (14 features)
        self.network_bounds = {
            'totalEnergy': [0, 20000],
            'activeCells': [0, 57],
            'avgDropRate': [0, 100], # Tỷ lệ %
            'avgLatency': [0, 200],
            'totalTraffic': [0, 10000],
            'connectedUEs': [0, 300],
            'connectionRate': [0, 100],
            'cpuViolations': [0, 100],
            'prbViolations': [0, 100],
            'maxCpuUsage': [0, 100],
            'maxPrbUsage': [0, 100],
            'kpiViolations': [0, 300], # Tổng số vi phạm
            'totalTxPower': [0, 2000],
            'avgPowerRatio': [0, 1]
        }
        
        # Các đặc trưng của cell (12 features per cell)
        self.cell_bounds = {
            'cpuUsage': [0, 100],
            'prbUsage': [0, 100],
            'currentLoad': [0, 2000],
            'maxCapacity': [0, 2000],
            'numConnectedUEs': [0, 50], # Max UEs per cell
            'txPower': [10, 49], # Min/Max tổng hợp
            'energyConsumption': [0, 5000],
            'avgRSRP': [-140, -70],
            'avgRSRQ': [-20, 0],
            'avgSINR': [-20, 40],
            'totalTrafficDemand': [0, 1000],
            'loadRatio': [0, 1]
        }

    def _normalize_value(self, value, min_val, max_val):
        """Chuẩn hóa một giá trị đơn lẻ."""
        if max_val == min_val:
            return 0.5
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)

    def normalize(self, state_vector):
        """Chuẩn hóa toàn bộ vector trạng thái."""
        state_vector = np.array(state_vector).flatten()
        normalized = np.zeros_like(state_vector)
        
        # 1. Simulation features
        sim_keys = list(self.simulation_bounds.keys())
        sim_len = len(sim_keys)
        for i, key in enumerate(sim_keys):
            if i < len(state_vector):
                min_val, max_val = self.simulation_bounds[key]
                normalized[i] = self._normalize_value(state_vector[i], min_val, max_val)

        # 2. Network features
        net_keys = list(self.network_bounds.keys())
        net_len = len(net_keys)
        for i, key in enumerate(net_keys):
            g_idx = sim_len + i
            if g_idx < len(state_vector):
                min_val, max_val = self.network_bounds[key]
                normalized[g_idx] = self._normalize_value(state_vector[g_idx], min_val, max_val)

        # 3. Cell features
        cell_keys = list(self.cell_bounds.keys())
        cell_feat_len = len(cell_keys)
        start_idx = sim_len + net_len
        
        num_active_cells = int(state_vector[0]) # Lấy số cell thực tế từ state
        for c_idx in range(num_active_cells):
            for f_idx, key in enumerate(cell_keys):
                g_idx = start_idx + c_idx * cell_feat_len + f_idx
                if g_idx < len(state_vector):
                    min_val, max_val = self.cell_bounds[key]
                    normalized[g_idx] = self._normalize_value(state_vector[g_idx], min_val, max_val)
        
        return normalized