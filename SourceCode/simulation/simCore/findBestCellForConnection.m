function bestCell = findBestCellForConnection(ue, rsrpThreshold, cells)
    bestCell = [];
    bestRsrp = -Inf;
    
    for cellIdx = 1:length(cells)
        % Check if we have measurement for this cell
        cellMeasurement = [];
        for measIdx = 1:length(ue.neighborMeasurements)
            if ue.neighborMeasurements(measIdx).cellId == cells(cellIdx).id
                cellMeasurement = ue.neighborMeasurements(measIdx);
                break;
            end
        end
        
        if ~isempty(cellMeasurement) && cellMeasurement.rsrp >= rsrpThreshold
            if cellMeasurement.rsrp > bestRsrp
                bestRsrp = cellMeasurement.rsrp;
                bestCell = struct();
                bestCell.cellId = cells(cellIdx).id;
                bestCell.rsrp = cellMeasurement.rsrp;
                bestCell.rsrq = cellMeasurement.rsrq;
                bestCell.sinr = cellMeasurement.sinr;
            end
        end
    end
end