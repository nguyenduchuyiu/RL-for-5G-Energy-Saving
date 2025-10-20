function cells = configureCells(sites, simParams)
% Configure cells based on 3GPP scenario parameters
% Refactored for better maintainability and extensibility

    cells = [];
    cellIdx = 1;
    
    % Get cell configurators registry
    configurators = getCellConfigurators();
    
    fprintf('Configuring cells for %s scenario...\n', simParams.deploymentScenario);
    
    for siteIdx = 1:length(sites)
        site = sites(siteIdx);
        
        % Get appropriate cell configuration
        cellConfig = getCellConfigurationForSite(site, simParams, configurators);
        
        % Create sectors for this site
        sectorsCreated = createSectorsForSite(site, cellConfig, cellIdx, simParams);
        
        % Add sectors to cells array
        for i = 1:length(sectorsCreated)
            if isempty(cells)
                cells = sectorsCreated(i);
            else
                cells(end+1) = sectorsCreated(i);
            end
        end
        
        cellIdx = cellIdx + length(sectorsCreated);
    end
    
    fprintf('Configured %d cells for %s scenario\n', length(cells), simParams.deploymentScenario);
end

function configurators = getCellConfigurators()
% Registry of cell configuration functions by site type
    
    configurators = struct();
    configurators.indoor_trxp = @getIndoorCellConfig;
    configurators.macro = @getMacroCellConfig;
    configurators.micro = @getllConfig;
    configurators.rural_macro = @getRuralMacroCellConfig;
    configurators.urban_macro = @getUrbanMacroCellConfig;
end

function cellConfig = getCellConfigurationForSite(site, simParams, configurators)
% Get appropriate cell configuration based on site type and scenario
    
    if isfield(site, 'type') && isfield(configurators, site.type)
        cellConfig = configurators.(site.type)(simParams, site);
    else
        % Fallback to scenario-based configuration
        switch simParams.deploymentScenario
            case 'indoor_hotspot'
                cellConfig = configurators.indoor_trxp(simParams, site);
            case 'dense_urban'
                cellConfig = configurators.macro(simParams, site);
            case 'rural'
                cellConfig = configurators.rural_macro(simParams, site);
            case 'urban_macro'
                cellConfig = configurators.urban_macro(simParams, site);
            otherwise
                cellConfig = configurators.macro(simParams, site);
        end
    end
end

function sectors = createSectorsForSite(site, cellConfig, startCellIdx, simParams)
% Create sectors for a given site
    
    numSectors = determineNumSectors(site, simParams);
    sectors = [];
    
    for sectorIdx = 1:numSectors
        azimuth = calculateSectorAzimuth(sectorIdx, numSectors);
        
        newCell = createCellStruct(...
            startCellIdx + sectorIdx - 1, ...
            site, ...
            sectorIdx, ...
            azimuth, ...
            cellConfig, ...
            numSectors == 1 ...
        );
        
        % Append to sectors array
        if isempty(sectors)
            sectors = newCell;
        else
            sectors(end+1) = newCell;
        end
    end
end

function numSectors = determineNumSectors(site, simParams)
% Determine number of sectors based on site type and scenario
    
    if strcmp(simParams.deploymentScenario, 'indoor_hotspot')
        numSectors = 1; % Indoor TRxPs are omnidirectional
    elseif isfield(site, 'type') && strcmp(site.type, 'micro')
        numSectors = 1; % Micro cells are typically omnidirectional
    else
        numSectors = getFieldOrDefault(simParams, 'numSectors', 3);
    end
end

function azimuth = calculateSectorAzimuth(sectorIdx, numSectors)
% Calculate sector azimuth angle
    azimuth = (sectorIdx - 1) * (360 / numSectors);
end

function cell = createCellStruct(cellId, site, sectorId, azimuth, config, isOmnidirectional)
% Create standardized cell structure
    
    cell = struct(...
        'id', cellId, ...
        'siteId', site.id, ...
        'sectorId', sectorId, ...
        'azimuth', azimuth, ...
        'x', site.x, ...
        'y', site.y, ...
        'frequency', config.frequency, ...
        'antennaHeight', config.antennaHeight, ...
        'txPower', config.initialTxPower, ...
        'minTxPower', config.minTxPower, ...
        'maxTxPower', config.maxTxPower, ...
        'cellRadius', config.cellRadius, ...
        'cpuUsage', 0, ...
        'prbUsage', 0, ...
        'energyConsumption', config.basePower, ...
        'baseEnergyConsumption', config.basePower, ...
        'idleEnergyConsumption', config.idlePower, ...
        'maxCapacity', config.maxCapacity, ...
        'currentLoad', 0, ...
        'connectedUEs', [], ...
        'ttt', getFieldOrDefault(config, 'ttt', 8), ...
        'a3Offset', getFieldOrDefault(config, 'a3Offset', 8), ...
        'isOmnidirectional', isOmnidirectional, ...
        'siteType', site.type ...
    );
end

% Cell Configuration Functions

function cellConfig = getIndoorCellConfig(simParams, site)
% Indoor hotspot cell configuration
    
    cellConfig = struct();
    cellConfig.frequency = simParams.carrierFrequency;
    cellConfig.antennaHeight = getFieldOrDefault(simParams, 'antennaHeight', 3);
    cellConfig.initialTxPower = 23;
    cellConfig.minTxPower = getFieldOrDefault(simParams, 'minTxPower', 20);
    cellConfig.maxTxPower = getFieldOrDefault(simParams, 'maxTxPower', 30);
    cellConfig.cellRadius = getFieldOrDefault(simParams, 'cellRadius', 50);
    cellConfig.basePower = getFieldOrDefault(simParams, 'basePower', 400);
    cellConfig.idlePower = getFieldOrDefault(simParams, 'idlePower', 100);
    cellConfig.maxCapacity = 50;
    cellConfig.ttt = 4; % Shorter TTT for indoor
    cellConfig.a3Offset = 6;
end

function cellConfig = getMacroCellConfig(simParams, site)
% Dense urban macro cell configuration
    
    cellConfig = struct();
    cellConfig.frequency = simParams.carrierFrequency;
    cellConfig.antennaHeight = getFieldOrDefault(simParams, 'antennaHeight', 25);
    cellConfig.initialTxPower = 43;
    cellConfig.minTxPower = getFieldOrDefault(simParams, 'minTxPower', 30);
    cellConfig.maxTxPower = getFieldOrDefault(simParams, 'maxTxPower', 46);
    cellConfig.cellRadius = getFieldOrDefault(simParams, 'cellRadius', 200);
    cellConfig.basePower = getFieldOrDefault(simParams, 'basePower', 800);
    cellConfig.idlePower = getFieldOrDefault(simParams, 'idlePower', 200);
    cellConfig.maxCapacity = 200;
    cellConfig.ttt = 8;
    cellConfig.a3Offset = 8;
end

function cellConfig = getllConfig(simParams, site)
% Micro cell configuration for dense urban
    
    cellConfig = struct();
    cellConfig.frequency = simParams.carrierFrequency;
    cellConfig.antennaHeight = 10;
    cellConfig.initialTxPower = 30;
    cellConfig.minTxPower = 20;
    cellConfig.maxTxPower = 38;
    cellConfig.cellRadius = 50;
    cellConfig.basePower = 200;
    cellConfig.idlePower = 50;
    cellConfig.maxCapacity = 100;
    cellConfig.ttt = 6; % Shorter for micro cells
    cellConfig.a3Offset = 6;
end

function cellConfig = getRuralMacroCellConfig(simParams, site)
% Rural macro cell configuration
    
    cellConfig = struct();
    cellConfig.frequency = simParams.carrierFrequency;
    cellConfig.antennaHeight = getFieldOrDefault(simParams, 'antennaHeight', 35);
    cellConfig.initialTxPower = 46; % Higher power for rural coverage
    cellConfig.minTxPower = getFieldOrDefault(simParams, 'minTxPower', 35);
    cellConfig.maxTxPower = getFieldOrDefault(simParams, 'maxTxPower', 49);
    cellConfig.cellRadius = getFieldOrDefault(simParams, 'cellRadius', 1000);
    cellConfig.basePower = getFieldOrDefault(simParams, 'basePower', 1200);
    cellConfig.idlePower = getFieldOrDefault(simParams, 'idlePower', 300);
    cellConfig.maxCapacity = 150; % Lower user density in rural
    cellConfig.ttt = 12; % Longer TTT for high mobility
    cellConfig.a3Offset = 10;
end

function cellConfig = getUrbanMacroCellConfig(simParams, site)
% Urban macro cell configuration
    
    cellConfig = struct();
    cellConfig.frequency = simParams.carrierFrequency;
    cellConfig.antennaHeight = getFieldOrDefault(simParams, 'antennaHeight', 25);
    cellConfig.initialTxPower = 43;
    cellConfig.minTxPower = getFieldOrDefault(simParams, 'minTxPower', 30);
    cellConfig.maxTxPower = getFieldOrDefault(simParams, 'maxTxPower', 46);
    cellConfig.cellRadius = getFieldOrDefault(simParams, 'cellRadius', 300);
    cellConfig.basePower = getFieldOrDefault(simParams, 'basePower', 1000);
    cellConfig.idlePower = getFieldOrDefault(simParams, 'idlePower', 250);
    cellConfig.maxCapacity = 250;
    cellConfig.ttt = 8;
    cellConfig.a3Offset = 8;
end

function value = getFieldOrDefault(structure, fieldName, defaultValue)
% Helper function to safely get field value with default fallback
    if isfield(structure, fieldName)
        value = structure.(fieldName);
    else
        value = defaultValue;
    end
end