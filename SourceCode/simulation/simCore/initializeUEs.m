function ues = initializeUEs(simParams, sites, seed)
% Initialize UEs based on 3GPP scenario requirements
% Refactored for better maintainability and extensibility

    ueRng = RandStream('mt19937ar', 'Seed', seed + 2000);
    prevStream = RandStream.setGlobalStream(ueRng);

    % Get UE initializers registry
    initializers = getUEInitializers();
    
    if isfield(initializers, simParams.deploymentScenario)
        ues = initializers.(simParams.deploymentScenario)(simParams, sites, seed);
    else
        warning('Unknown deployment scenario: %s. Using default initialization.', simParams.deploymentScenario);
        ues = initializers.default(simParams, sites, seed);
    end

    RandStream.setGlobalStream(prevStream);
    fprintf('Initialized %d UEs for %s scenario\n', length(ues), simParams.deploymentScenario);
end

function initializers = getUEInitializers()
% Registry of scenario-specific UE initializers
    
    initializers = struct();
    initializers.indoor_hotspot = @initializeIndoorHotspotUEs;
    initializers.dense_urban = @initializeDenseUrbanUEs;
    initializers.rural = @initializeRuralUEs;
    initializers.urban_macro = @initializeUrbanMacroUEs;
    initializers.default = @initializeDefaultUEs;
end

function ues = initializeIndoorHotspotUEs(simParams, sites, seed)
% Initialize UEs for indoor hotspot scenario
    
    numUEs = simParams.numUEs;
    mobilityConfig = getIndoorMobilityConfig();
    spatialConfig = getIndoorSpatialConfig();
    
    ues = [];
    for ueIdx = 1:numUEs
        position = generateIndoorPosition(spatialConfig, sites);
        mobilityParams = selectMobilityPattern(mobilityConfig);
        
        newUE = createUEStruct(ueIdx, position.x, position.y, ...
                              mobilityParams.velocity, mobilityParams.direction, ...
                              mobilityParams.pattern, simParams, seed);
        
        if isempty(ues)
            ues = newUE;
        else
            ues(end+1) = newUE;
        end
    end
end

function ues = initializeDenseUrbanUEs(simParams, sites, seed)
% Initialize UEs for dense urban scenario
    
    numUEs = simParams.numUEs;
    indoorRatio = getFieldOrDefault(simParams, 'indoorRatio', 0.8);
    
    indoorUEs = round(numUEs * indoorRatio);
    outdoorUEs = numUEs - indoorUEs;
    
    ues = [];
    ueIdx = 1;
    
    % Indoor UEs
    indoorConfig = getDenseUrbanIndoorConfig(simParams);
    for i = 1:indoorUEs
        position = generateUrbanPosition(sites, indoorConfig.positioning);
        mobilityParams = selectMobilityPattern(indoorConfig.mobility);
        
        newUE = createUEStruct(ueIdx, position.x, position.y, ...
                              mobilityParams.velocity, mobilityParams.direction, ...
                              mobilityParams.pattern, simParams, seed);
        
        if isempty(ues)
            ues = newUE;
        else
            ues(end+1) = newUE;
        end
        ueIdx = ueIdx + 1;
    end
    
    % Outdoor UEs
    outdoorConfig = getDenseUrbanOutdoorConfig(simParams);
    for i = 1:outdoorUEs
        position = generateUrbanPosition(sites, outdoorConfig.positioning);
        mobilityParams = selectMobilityPattern(outdoorConfig.mobility);
        
        newUE = createUEStruct(ueIdx, position.x, position.y, ...
                              mobilityParams.velocity, mobilityParams.direction, ...
                              mobilityParams.pattern, simParams, seed);
        
        if isempty(ues)
            ues = newUE;
        else
            ues(end+1) = newUE;
        end
        ueIdx = ueIdx + 1;
    end
end

function ues = initializeRuralUEs(simParams, sites, seed)
% Initialize UEs for rural scenario
    
    numUEs = simParams.numUEs;
    mobilityConfig = getRuralMobilityConfig(simParams);
    spatialConfig = getRuralSpatialConfig(simParams);
    
    ues = [];
    for ueIdx = 1:numUEs
        position = generateRuralPosition(sites, spatialConfig);
        mobilityParams = selectMobilityPattern(mobilityConfig);
        
        newUE = createUEStruct(ueIdx, position.x, position.y, ...
                              mobilityParams.velocity, mobilityParams.direction, ...
                              mobilityParams.pattern, simParams, seed);
        
        if isempty(ues)
            ues = newUE;
        else
            ues(end+1) = newUE;
        end
    end
end

function ues = initializeUrbanMacroUEs(simParams, sites, seed)
% Initialize UEs for urban macro scenario
    
    numUEs = simParams.numUEs;
    mobilityConfig = getUrbanMacroMobilityConfig(simParams);
    spatialConfig = getUrbanMacroSpatialConfig(simParams);
    
    ues = [];
    for ueIdx = 1:numUEs
        position = generateUrbanMacroPosition(sites, spatialConfig);
        mobilityParams = selectMobilityPattern(mobilityConfig);
        
        newUE = createUEStruct(ueIdx, position.x, position.y, ...
                              mobilityParams.velocity, mobilityParams.direction, ...
                              mobilityParams.pattern, simParams, seed);
        
        if isempty(ues)
            ues = newUE;
        else
            ues(end+1) = newUE;
        end
    end
end

function ues = initializeDefaultUEs(simParams, sites, seed)
% Initialize UEs for default scenario
    
    numUEs = simParams.numUEs;
    mobilityConfig = getDefaultMobilityConfig(simParams);
    spatialConfig = getDefaultSpatialConfig(simParams);
    
    ues = [];
    for ueIdx = 1:numUEs
        position = generateDefaultPosition(sites, spatialConfig);
        mobilityParams = selectMobilityPattern(mobilityConfig);
        
        newUE = createUEStruct(ueIdx, position.x, position.y, ...
                              mobilityParams.velocity, mobilityParams.direction, ...
                              mobilityParams.pattern, simParams, seed);
        
        if isempty(ues)
            ues = newUE;
        else
            ues(end+1) = newUE;
        end
    end
end

% Configuration Functions

function config = getIndoorMobilityConfig()
% Indoor hotspot mobility configuration
    
    config = struct();
    config.patterns = {'stationary', 'slow_walk', 'normal_walk'};
    config.velocities = [0, 0.5, 1.5]; % m/s
    config.weights = [0.4, 0.4, 0.2]; % More stationary users indoors
end

function config = getIndoorSpatialConfig()
% Indoor spatial distribution configuration
    
    config = struct();
    config.bounds = struct('minX', 10, 'maxX', 110, 'minY', 5, 'maxY', 45);
    config.avoidanceRadius = 5; % Minimum distance from sites
    config.maxAttempts = 100;
end

function config = getDenseUrbanIndoorConfig(simParams)
% Dense urban indoor UE configuration
    
    config = struct();
    config.mobility = struct();
    config.mobility.patterns = {'indoor_pedestrian'};
    config.mobility.velocities = [simParams.ueSpeed / 3.6]; % Convert km/h to m/s
    config.mobility.weights = [1.0];
    
    config.positioning = struct();
    config.positioning.maxDistance = 30; % Close to sites for indoor
    config.positioning.distribution = 'normal';
end

function config = getDenseUrbanOutdoorConfig(simParams)
% Dense urban outdoor UE configuration
    
    config = struct();
    config.mobility = struct();
    config.mobility.patterns = {'outdoor_vehicle'};
    config.mobility.velocities = [getFieldOrDefault(simParams, 'outdoorSpeed', 30) / 3.6];
    config.mobility.weights = [1.0];
    
    config.positioning = struct();
    config.positioning.minDistance = 50;
    config.positioning.maxDistance = 150;
    config.positioning.distribution = 'uniform';
end

function config = getRuralMobilityConfig(simParams)
% Rural mobility configuration
    
    config = struct();
    config.patterns = {'stationary', 'pedestrian', 'slow_vehicle', 'fast_vehicle'};
    config.velocities = [0, 1.0, simParams.ueSpeed/3.6, simParams.ueSpeed/3.6]; % m/s
    config.weights = [0.1, 0.4, 0.3, 0.2]; % Mixed mobility in rural
end

function config = getRuralSpatialConfig(simParams)
% Rural spatial distribution configuration
    
    config = struct();
    config.maxRadius = simParams.isd * 3; % Large coverage area
    config.distribution = 'clustered_uniform'; % Some clustering around sites
    config.clusterProbability = 0.6;
    config.clusterRadius = 200;
end

function config = getUrbanMacroMobilityConfig(simParams)
% Urban macro mobility configuration
    
    config = struct();
    config.patterns = {'pedestrian', 'slow_vehicle', 'vehicle'};
    config.velocities = [1.5, simParams.ueSpeed/3.6, simParams.ueSpeed/3.6]; % m/s
    config.weights = [0.6, 0.2, 0.2]; % Mostly pedestrians in urban macro
end

function config = getUrbanMacroSpatialConfig(simParams)
% Urban macro spatial configuration
    
    config = struct();
    config.maxRadius = simParams.cellRadius * 1.5;
    config.distribution = 'mixed';
    config.indoorRatio = getFieldOrDefault(simParams, 'indoorRatio', 0.8);
end

function config = getDefaultMobilityConfig(simParams)
% Default mobility configuration
    
    config = struct();
    config.patterns = {'stationary', 'pedestrian', 'slow_vehicle', 'fast_vehicle', 'vehicle'};
    config.velocities = [0, 1.5, 5.0, 15.0, 10.0]; % m/s
    config.weights = [0.2, 0.2, 0.2, 0.2, 0.2]; % Uniform distribution
end

function config = getDefaultSpatialConfig(simParams)
% Default spatial configuration
    
    config = struct();
    config.maxRadius = simParams.isd * sqrt(simParams.numSites) / (2 * pi);
    config.distribution = 'uniform';
end

% Position Generation Functions

function position = generateIndoorPosition(spatialConfig, sites)
% Generate UE position for indoor scenario
    
    bounds = spatialConfig.bounds;
    avoidanceRadius = spatialConfig.avoidanceRadius;
    maxAttempts = spatialConfig.maxAttempts;
    
    validPosition = false;
    attempts = 0;
    
    while ~validPosition && attempts < maxAttempts
        x = bounds.minX + rand() * (bounds.maxX - bounds.minX);
        y = bounds.minY + rand() * (bounds.maxY - bounds.minY);
        
        % Check minimum distance from sites
        validPosition = true;
        for siteIdx = 1:length(sites)
            distance = sqrt((x - sites(siteIdx).x)^2 + (y - sites(siteIdx).y)^2);
            if distance < avoidanceRadius
                validPosition = false;
                break;
            end
        end
        attempts = attempts + 1;
    end
    
    % Fallback to safe position if needed
    if ~validPosition
        x = (bounds.minX + bounds.maxX) / 2;
        y = (bounds.minY + bounds.maxY) / 2;
    end
    
    position = struct('x', x, 'y', y);
end

function position = generateUrbanPosition(sites, posConfig)
% Generate UE position for urban scenarios
    
    siteIdx = randi(length(sites));
    site = sites(siteIdx);
    
    angle = rand() * 2 * pi;
    
    if strcmp(posConfig.distribution, 'normal')
        distance = abs(randn()) * posConfig.maxDistance;
    else
        minDist = getFieldOrDefault(posConfig, 'minDistance', 0);
        maxDist = posConfig.maxDistance;
        distance = minDist + rand() * (maxDist - minDist);
    end
    
    x = site.x + distance * cos(angle);
    y = site.y + distance * sin(angle);
    
    position = struct('x', x, 'y', y);
end

function position = generateRuralPosition(sites, spatialConfig)
% Generate UE position for rural scenario
    
    if rand() < spatialConfig.clusterProbability
        % Clustered around a site
        siteIdx = randi(length(sites));
        site = sites(siteIdx);
        
        angle = rand() * 2 * pi;
        distance = rand() * spatialConfig.clusterRadius;
        
        x = site.x + distance * cos(angle);
        y = site.y + distance * sin(angle);
    else
        % Uniform distribution in coverage area
        angle = rand() * 2 * pi;
        radius = spatialConfig.maxRadius * sqrt(rand());
        
        x = radius * cos(angle);
        y = radius * sin(angle);
    end
    
    position = struct('x', x, 'y', y);
end

function position = generateUrbanMacroPosition(sites, spatialConfig)
% Generate UE position for urban macro scenario
    
    if rand() < spatialConfig.indoorRatio
        % Indoor positioning (closer to sites)
        siteIdx = randi(length(sites));
        site = sites(siteIdx);
        
        angle = rand() * 2 * pi;
        distance = abs(randn()) * (spatialConfig.maxRadius * 0.3);
        
        x = site.x + distance * cos(angle);
        y = site.y + distance * sin(angle);
    else
        % Outdoor positioning
        angle = rand() * 2 * pi;
        radius = spatialConfig.maxRadius * sqrt(rand());
        
        x = radius * cos(angle);
        y = radius * sin(angle);
    end
    
    position = struct('x', x, 'y', y);
end

function position = generateDefaultPosition(sites, spatialConfig)
% Generate UE position for default scenario
    
    angle = rand() * 2 * pi;
    radius = spatialConfig.maxRadius * sqrt(rand());
    
    x = radius * cos(angle);
    y = radius * sin(angle);
    
    position = struct('x', x, 'y', y);
end

function mobilityParams = selectMobilityPattern(mobilityConfig)
% Select mobility pattern based on weights
    
    randVal = rand();
    cumWeights = cumsum(mobilityConfig.weights);
    patternIdx = find(randVal <= cumWeights, 1);
    
    mobilityParams = struct();
    mobilityParams.pattern = mobilityConfig.patterns{patternIdx};
    mobilityParams.velocity = mobilityConfig.velocities(patternIdx);
    mobilityParams.direction = rand() * 2 * pi;
end

function ue = createUEStruct(ueId, x, y, velocity, direction, mobilityPattern, simParams, seed)
% Create standardized UE structure with enhanced fields
    
    ue = struct(...
        'id', ueId, ...
        'x', x, ...
        'y', y, ...
        'velocity', velocity, ...
        'direction', direction, ...
        'mobilityPattern', mobilityPattern, ...
        'servingCell', NaN, ...
        'rsrp', NaN, ...
        'rsrq', NaN, ...
        'sinr', NaN, ...
        'neighborMeasurements', [], ...
        'hoTimer', 0, ...
        'stepCounter', 0, ...
        'lastDirectionChange', 0, ...
        'pauseTimer', 0, ...
        'connectionTimer', 0, ...
        'disconnectionTimer', 0, ...
        'lastServingRsrp', NaN, ...
        'trafficDemand', 0, ...
        'qosLatency', 0, ...
        'sessionActive', false, ...
        'dropCount', 0, ...
        'rngS', seed + ueId * 100, ...
        'deploymentScenario', simParams.deploymentScenario, ...
        'handoverHistory', struct('ueId', {}, 'cellSource', {}, 'cellTarget', {}, ...
                     'rsrpSource', {}, 'rsrpTarget', {}, ...
                     'rsrqSource', {}, 'rsrqTarget', {}, ...
                     'sinrSource', {}, 'sinrTarget', {}, ...
                     'a3Offset', {}, 'ttt', {}, ...
                     'hoSuccess', {}, 'timestamp', {}) ...
    );
end

function value = getFieldOrDefault(structure, fieldName, defaultValue)
% Helper function to safely get field value with default fallback
    if isfield(structure, fieldName)
        value = structure.(fieldName);
    else
        value = defaultValue;
    end
end