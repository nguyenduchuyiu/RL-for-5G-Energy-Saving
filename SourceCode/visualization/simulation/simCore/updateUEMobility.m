function ues = updateUEMobility(ues, timeStep, currentTime, seed, simParams)
% Update UE mobility with scenario-specific patterns
% Refactored for better maintainability and extensibility

    for ueIdx = 1:length(ues)
        ue = ues(ueIdx);
        
        % Set up mobility-specific RNG
        mobilityRng = RandStream('mt19937ar', 'Seed', ue.rngS + floor(currentTime * 1000));
        prevStream = RandStream.setGlobalStream(mobilityRng);
        
        ue.stepCounter = ue.stepCounter + 1;
        
        % Update UE position based on mobility pattern
        ue = updateUEPosition(ue, timeStep, currentTime);
        
        % Enforce scenario-specific boundaries
        if exist('simParams', 'var') && isfield(simParams, 'deploymentScenario')
            ue = enforceScenarioBounds(ue, simParams);
        end
        
        RandStream.setGlobalStream(prevStream);
        ues(ueIdx) = ue;
    end
end

function ue = updateUEPosition(ue, timeStep, currentTime)
% Update UE position based on its mobility pattern
    
    % Get mobility handler
    mobilityHandlers = getMobilityHandlers();
    
    if isfield(mobilityHandlers, ue.mobilityPattern)
        ue = mobilityHandlers.(ue.mobilityPattern)(ue, timeStep, currentTime);
    else
        warning('Unknown mobility pattern: %s. Using pedestrian model.', ue.mobilityPattern);
        ue = mobilityHandlers.pedestrian(ue, timeStep, currentTime);
    end
end

function handlers = getMobilityHandlers()
% Registry of mobility pattern handlers
    
    handlers = struct();
    handlers.stationary = @handleStationaryMobility;
    handlers.pedestrian = @handlePedestrianMobility;
    handlers.slow_walk = @handleSlowWalkMobility;
    handlers.normal_walk = @handleNormalWalkMobility;
    handlers.fast_walk = @handleFastWalkMobility;
    handlers.slow_vehicle = @handleSlowVehicleMobility;
    handlers.fast_vehicle = @handleFastVehicleMobility;
    handlers.indoor_pedestrian = @handleIndoorPedestrianMobility;
    handlers.indoor_mobile = @handleIndoorMobileMobility;
    handlers.outdoor_vehicle = @handleOutdoorVehicleMobility;
    handlers.vehicle = @handleVehicleMobility;
end

% Mobility Pattern Handlers

function ue = handleStationaryMobility(ue, timeStep, currentTime)
% Stationary UE with minimal random movement
    
    if rand() < 0.05 % 5% chance of small movement
        ue.x = ue.x + (rand() - 0.5) * 2; % ±1m
        ue.y = ue.y + (rand() - 0.5) * 2;
    end
end

function ue = handlePedestrianMobility(ue, timeStep, currentTime)
% Standard pedestrian mobility with pause periods
    
    distance = ue.velocity * timeStep;
    
    if ue.pauseTimer > 0
        ue.pauseTimer = ue.pauseTimer - timeStep;
    elseif rand() < 0.1 % 10% chance to pause
        ue.pauseTimer = 5 + rand() * 10; % Pause 5-15 seconds
    elseif rand() < 0.3 % 30% chance to change direction
        ue.direction = ue.direction + (rand() - 0.5) * pi;
        ue = moveUE(ue, distance);
    else
        ue = moveUE(ue, distance);
    end
end

function ue = handleFastVehicleMobility(ue, timeStep, currentTime)
% Fast vehicle mobility pattern
    
    distance = ue.velocity * timeStep;
    
    % Infrequent direction changes for highways
    if currentTime - ue.lastDirectionChange > 40 + rand() * 20
        ue.direction = ue.direction + (rand() - 0.5) * pi/4;
        ue.lastDirectionChange = currentTime;
    end
    
    ue = moveUE(ue, distance);
end

function ue = handleIndoorPedestrianMobility(ue, timeStep, currentTime)
% Indoor pedestrian with pause and direction changes
    
    distance = ue.velocity * timeStep;
    
    if ue.pauseTimer > 0
        ue.pauseTimer = ue.pauseTimer - timeStep;
    elseif rand() < 0.15 % 15% chance to pause
        ue.pauseTimer = 2 + rand() * 8; % Pause 2-10 seconds
    elseif rand() < 0.4 % 40% chance to change direction
        ue.direction = ue.direction + (rand() - 0.5) * pi; % ±90°
        ue = moveUE(ue, distance);
    else
        ue = moveUE(ue, distance);
    end
end

function ue = handleIndoorMobileMobility(ue, timeStep, currentTime)
% General indoor mobile pattern
    
    distance = ue.velocity * timeStep;
    
    if rand() < 0.2 % 20% chance to change direction
        ue.direction = ue.direction + (rand() - 0.5) * pi;
    end
    
    ue = moveUE(ue, distance);
end

function ue = handleOutdoorVehicleMobility(ue, timeStep, currentTime)
% Outdoor vehicle mobility for urban scenarios
    
    distance = ue.velocity * timeStep;
    
    if currentTime - ue.lastDirectionChange > 30 + rand() * 20
        ue.direction = ue.direction + (rand() - 0.5) * pi/6;
        ue.lastDirectionChange = currentTime;
    end
    
    ue = moveUE(ue, distance);
end

function ue = handleVehicleMobility(ue, timeStep, currentTime)
% Generic vehicle mobility pattern
    
    distance = ue.velocity * timeStep;
    
    if currentTime - ue.lastDirectionChange > 25 + rand() * 15
        ue.direction = ue.direction + (rand() - 0.5) * pi/3;
        ue.lastDirectionChange = currentTime;
    end
    
    ue = moveUE(ue, distance);
end

function ue = moveUE(ue, distance)
% Move UE in its current direction by given distance
    
    ue.x = ue.x + distance * cos(ue.direction);
    ue.y = ue.y + distance * sin(ue.direction);
end

function ue = enforceScenarioBounds(ue, simParams)
% Enforce boundaries based on deployment scenario
    
    boundaryHandlers = getBoundaryHandlers();
    
    if isfield(boundaryHandlers, simParams.deploymentScenario)
        ue = boundaryHandlers.(simParams.deploymentScenario)(ue, simParams);
    else
        ue = boundaryHandlers.default(ue, simParams);
    end
    
    % Normalize direction to [0, 2π]
    ue.direction = mod(ue.direction, 2*pi);
end

function handlers = getBoundaryHandlers()
% Registry of scenario-specific boundary handlers
    
    handlers = struct();
    handlers.indoor_hotspot = @enforceIndoorBounds;
    handlers.dense_urban = @enforceUrbanBounds;
    handlers.rural = @enforceRuralBounds;
    handlers.urban_macro = @enforceUrbanMacroBounds;
    handlers.default = @enforceDefaultBounds;
end

function ue = enforceIndoorBounds(ue, simParams)
% Indoor building bounds (120m x 50m office)
    
    bounds = struct('minX', 5, 'maxX', 115, 'minY', 5, 'maxY', 45);
    
    % Bounce off walls
    if ue.x <= bounds.minX
        ue.x = bounds.minX + 1;
        ue.direction = pi - ue.direction; % Reflect horizontally
    elseif ue.x >= bounds.maxX
        ue.x = bounds.maxX - 1;
        ue.direction = pi - ue.direction;
    end
    
    if ue.y <= bounds.minY
        ue.y = bounds.minY + 1;
        ue.direction = -ue.direction; % Reflect vertically
    elseif ue.y >= bounds.maxY
        ue.y = bounds.maxY - 1;
        ue.direction = -ue.direction;
    end
end

function ue = enforceUrbanBounds(ue, simParams)
% Urban area bounds
    
    maxRadius = getFieldOrDefault(simParams, 'maxRadius', 500);
    distance = sqrt(ue.x^2 + ue.y^2);
    
    if distance > maxRadius
        % Reflect back towards center
        angle = atan2(ue.y, ue.x);
        ue.x = (maxRadius - 10) * cos(angle);
        ue.y = (maxRadius - 10) * sin(angle);
        ue.direction = angle + pi + (rand() - 0.5) * pi/2;
    end
end

function ue = enforceRuralBounds(ue, simParams)
% Rural area bounds (larger coverage area)
    
    maxRadius = getFieldOrDefault(simParams, 'maxRadius', 2000); % Larger rural coverage
    distance = sqrt(ue.x^2 + ue.y^2);
    
    if distance > maxRadius
        angle = atan2(ue.y, ue.x);
        ue.x = (maxRadius - 50) * cos(angle);
        ue.y = (maxRadius - 50) * sin(angle);
        ue.direction = angle + pi + (rand() - 0.5) * pi/4;
    end
end

function ue = enforceUrbanMacroBounds(ue, simParams)
% Urban macro bounds
    
    maxRadius = getFieldOrDefault(simParams, 'maxRadius', 800);
    distance = sqrt(ue.x^2 + ue.y^2);
    
    if distance > maxRadius
        angle = atan2(ue.y, ue.x);
        ue.x = (maxRadius - 20) * cos(angle);
        ue.y = (maxRadius - 20) * sin(angle);
        ue.direction = angle + pi + (rand() - 0.5) * pi/3;
    end
end

function ue = enforceDefaultBounds(ue, simParams)
% Default bounds for unknown scenarios
    
    maxBound = 1000;
    if abs(ue.x) > maxBound || abs(ue.y) > maxBound
        ue.x = min(max(ue.x, -maxBound), maxBound);
        ue.y = min(max(ue.y, -maxBound), maxBound);
        ue.direction = ue.direction + pi + (rand() - 0.5) * pi/4;
    end
end

function value = getFieldOrDefault(structure, fieldName, defaultValue)
% Helper function to safely get field value with default fallback
    if isfield(structure, fieldName)
        value = structure.(fieldName);
    else
        value = defaultValue;
    end
end

function ue = handleSlowWalkMobility(ue, timeStep, currentTime)
% Slow walking pattern for indoor scenarios
    
    distance = ue.velocity * timeStep;
    
    if rand() < 0.3 % 30% chance to change direction
        ue.direction = ue.direction + (rand() - 0.5) * pi/2; % ±45°
    end
    
    ue = moveUE(ue, distance);
end

function ue = handleNormalWalkMobility(ue, timeStep, currentTime)
% Normal walking pattern
    
    distance = ue.velocity * timeStep;
    
    if rand() < 0.4 % 40% chance to change direction
        ue.direction = ue.direction + (rand() - 0.5) * pi/2; % ±45°
    end
    
    ue = moveUE(ue, distance);
end

function ue = handleFastWalkMobility(ue, timeStep, currentTime)
% Fast walking/jogging pattern
    
    distance = ue.velocity * timeStep;
    
    if rand() < 0.2 % 20% chance to change direction
        ue.direction = ue.direction + (rand() - 0.5) * pi/4; % ±22.5°
    end
    
    ue = moveUE(ue, distance);
end

function ue = handleSlowVehicleMobility(ue, timeStep, currentTime)
% Slow vehicle mobility pattern
    
    distance = ue.velocity * timeStep;
    
    % Change direction less frequently than pedestrians
    if currentTime - ue.lastDirectionChange > 20 + rand() * 30
        ue.direction = ue.direction + (rand() - 0.5) * pi/2;
        ue.lastDirectionChange = currentTime;
    end

    ue = moveUE(ue, distance);
end