function hoSuccess = evaluateHandoverSuccess(ue, neighbor, servingCell, targetCell, currentTime, seed)
    % Create deterministic random stream for handover success
    hoRng = RandStream('mt19937ar', 'Seed', seed + ue.id + neighbor.cellId + floor(currentTime));
    prevStream = RandStream.setGlobalStream(hoRng);
    
    % CRITICAL FIX: Base success now depends on both source and target cell power levels
    baseSuccessProb = 0.98; % Start higher since we'll apply penalties
    
    % SOURCE CELL POWER PENALTY - Critical for energy saving scenarios
    if servingCell.txPower <= servingCell.minTxPower + 2
        sourcePowerPenalty = (servingCell.minTxPower + 2 - servingCell.txPower) * 0.15;
        baseSuccessProb = baseSuccessProb - sourcePowerPenalty;
    end
    
    % TARGET CELL POWER PENALTY - Target at low power = harder to connect
    if targetCell.txPower <= targetCell.minTxPower + 3
        targetPowerPenalty = (targetCell.minTxPower + 3 - targetCell.txPower) * 0.10;
        baseSuccessProb = baseSuccessProb - targetPowerPenalty;
    end
    
    % Signal quality factors (now more critical with power dependence)
    if neighbor.rsrp >= -75
        signalBonus = 0.02;
    elseif neighbor.rsrp >= -85
        signalBonus = 0.01;
    elseif neighbor.rsrp >= -95
        signalBonus = 0.0;
    elseif neighbor.rsrp >= -105
        signalBonus = -0.05; % Penalty for poor signal
    else
        signalBonus = -0.15; % Heavy penalty for very poor signal
    end
    
    % SINR penalty is more severe when cells are at low power
    if neighbor.sinr >= 15
        sinrBonus = 0.02;
    elseif neighbor.sinr >= 5
        sinrBonus = 0.01;
    elseif neighbor.sinr >= 0
        sinrBonus = 0.0;
    elseif neighbor.sinr >= -5
        sinrBonus = -0.03;
    else
        sinrBonus = -0.10; % Heavy penalty for poor SINR
    end
    
    % CRITICAL: Additional penalty if both cells are at minimum power
    if servingCell.txPower <= servingCell.minTxPower + 1 && targetCell.txPower <= targetCell.minTxPower + 1
        baseSuccessProb = baseSuccessProb - 0.20; % 20% additional penalty
    end
    
    % Resource congestion affects handover success
    if targetCell.cpuUsage > 85 || targetCell.prbUsage > 85
        congestionPenalty = 0.05;
        baseSuccessProb = baseSuccessProb - congestionPenalty;
    end
    
    finalSuccessProb = baseSuccessProb + signalBonus + sinrBonus;
    finalSuccessProb = max(0.25, min(0.98, finalSuccessProb)); % Allow much lower success rates
    
    hoSuccess = rand() < finalSuccessProb;
    
    RandStream.setGlobalStream(prevStream);
end