function simParams = loadScenarioConfig(scenarioInput)
% Load scenario configuration from JSON files
% Refactored for better maintainability and extensibility

    baseDir = fileparts(mfilename('fullpath'));
    scenariosDir = fullfile(baseDir, 'scenarios');
    
    % Get scenario name mappings
    nameMap = getScenarioNameMappings();
    
    % Resolve JSON file path
    jsonPath = resolveScenarioPath(scenarioInput, scenariosDir, nameMap);
    
    % Load and parse configuration
    simParams = loadAndParseConfig(jsonPath);
    
    % Validate configuration
    simParams = validateAndEnhanceConfig(simParams);
    
    fprintf('Loaded scenario: %s\n', simParams.name);
end

function nameMap = getScenarioNameMappings()
% Registry of scenario name to JSON file mappings
    
    nameMap = containers.Map();
    
    % 3GPP Standard scenarios
    nameMap('indoor_hotspot') = 'indoor_hotspot.json';
    nameMap('dense_urban') = 'dense_urban.json';
    nameMap('rural') = 'rural.json';
    nameMap('urban_macro') = 'urban_macro.json';
end

function jsonPath = resolveScenarioPath(scenarioInput, scenariosDir, nameMap)
% Resolve scenario input to actual JSON file path
    
    if isfile(scenarioInput)
        % Direct file path provided
        jsonPath = scenarioInput;
    elseif nameMap.isKey(scenarioInput)
        % Known scenario name
        jsonPath = fullfile(scenariosDir, nameMap(scenarioInput));
    else
        % Try appending .json extension
        candidate = fullfile(scenariosDir, [scenarioInput '.json']);
        if isfile(candidate)
            jsonPath = candidate;
        else
            availableScenarios = strjoin(keys(nameMap), ', ');
            error('ScenarioConfig:UnknownScenario', ...
                  'Unknown scenario: %s\nAvailable scenarios: %s\nOr provide a valid JSON file path', ...
                  scenarioInput, availableScenarios);
        end
    end
    
    % Verify file exists
    if ~isfile(jsonPath)
        error('ScenarioConfig:FileNotFound', 'JSON scenario file not found: %s', jsonPath);
    end
end

function simParams = loadAndParseConfig(jsonPath)
% Load and parse JSON configuration file
    
    try
        raw = fileread(jsonPath);
        cfg = jsondecode(raw);
        simParams = convertJSONToParams(cfg);
    catch ME
        if strcmp(ME.identifier, 'MATLAB:jsondecode:InvalidJSON')
            error('ScenarioConfig:InvalidJSON', 'Invalid JSON format in file: %s', jsonPath);
        elseif strcmp(ME.identifier, 'MATLAB:fileread:cannotOpenFile')
            error('ScenarioConfig:CannotRead', 'Cannot read file: %s', jsonPath);
        else
            error('ScenarioConfig:ParseError', 'Failed to parse JSON file %s: %s', jsonPath, ME.message);
        end
    end
end

function simParams = convertJSONToParams(cfg)
% Convert JSON configuration to simulation parameters structure
    
    simParams = struct();
    
    % Basic scenario information
    simParams.name = getFieldOrDefault(cfg, 'name', 'Unnamed Scenario');
    simParams.description = getFieldOrDefault(cfg, 'description', 'No description provided');
    simParams.deploymentScenario = getFieldOrDefault(cfg, 'deploymentScenario', 'custom');
    
    % Network topology parameters
    simParams = addNetworkTopologyParams(simParams, cfg);
    
    % RF parameters
    simParams = addRFParams(simParams, cfg);
    
    % User parameters
    simParams = addUserParams(simParams, cfg);
    
    % Power parameters
    simParams = addPowerParams(simParams, cfg);
    
    % Simulation parameters
    simParams = addSimulationParams(simParams, cfg);
    
    % Threshold parameters
    simParams = addThresholdParams(simParams, cfg);
    
    % Traffic parameters
    simParams = addTrafficParams(simParams, cfg);
    
    % Scenario-specific optional parameters
    simParams = addScenarioSpecificParams(simParams, cfg);
end

function simParams = addNetworkTopologyParams(simParams, cfg)
% Add network topology related parameters
    
    simParams.numSites = getFieldOrDefault(cfg, 'numSites', 7);
    simParams.numSectors = getFieldOrDefault(cfg, 'numSectors', 3);
    simParams.isd = getFieldOrDefault(cfg, 'isd', 200);
    simParams.antennaHeight = getFieldOrDefault(cfg, 'antennaHeight', 25);
    simParams.cellRadius = getFieldOrDefault(cfg, 'cellRadius', 200);
end

function simParams = addRFParams(simParams, cfg)
% Add RF related parameters
    
    simParams.carrierFrequency = getFieldOrDefault(cfg, 'carrierFrequency', 3.5e9);
    simParams.systemBandwidth = getFieldOrDefault(cfg, 'systemBandwidth', 100e6);
end

function simParams = addUserParams(simParams, cfg)
% Add user related parameters
    
    simParams.numUEs = getFieldOrDefault(cfg, 'numUEs', 210);
    simParams.ueSpeed = getFieldOrDefault(cfg, 'ueSpeed', 3);
    simParams.indoorRatio = getFieldOrDefault(cfg, 'indoorRatio', 0.8);
    simParams.outdoorSpeed = getFieldOrDefault(cfg, 'outdoorSpeed', 30);
end

function simParams = addPowerParams(simParams, cfg)
% Add power related parameters
    
    simParams.minTxPower = getFieldOrDefault(cfg, 'minTxPower', 30);
    simParams.maxTxPower = getFieldOrDefault(cfg, 'maxTxPower', 46);
    simParams.basePower = getFieldOrDefault(cfg, 'basePower', 800);
    simParams.idlePower = getFieldOrDefault(cfg, 'idlePower', 200);
end

function simParams = addSimulationParams(simParams, cfg)
% Add simulation control parameters
    
    simParams.simTime = getFieldOrDefault(cfg, 'simTime', 600);
    simParams.timeStep = getFieldOrDefault(cfg, 'timeStep', 1);
end

function simParams = addThresholdParams(simParams, cfg)
% Add threshold parameters for KPIs and measurements
    
    simParams.rsrpServingThreshold = getFieldOrDefault(cfg, 'rsrpServingThreshold', -110);
    simParams.rsrpTargetThreshold = getFieldOrDefault(cfg, 'rsrpTargetThreshold', -100);
    simParams.rsrpMeasurementThreshold = getFieldOrDefault(cfg, 'rsrpMeasurementThreshold', -115);
    simParams.dropCallThreshold = getFieldOrDefault(cfg, 'dropCallThreshold', 1);
    simParams.latencyThreshold = getFieldOrDefault(cfg, 'latencyThreshold', 50);
    simParams.cpuThreshold = getFieldOrDefault(cfg, 'cpuThreshold', 80);
    simParams.prbThreshold = getFieldOrDefault(cfg, 'prbThreshold', 80);
end

function simParams = addTrafficParams(simParams, cfg)
% Add traffic generation parameters
    
    simParams.trafficLambda = getFieldOrDefault(cfg, 'trafficLambda', 30);
    simParams.peakHourMultiplier = getFieldOrDefault(cfg, 'peakHourMultiplier', 1.5);
end

function simParams = addScenarioSpecificParams(simParams, cfg)
% Add scenario-specific optional parameters
    
    % Optional layout parameters
    if isfield(cfg, 'layout')
        simParams.layout = cfg.layout;
    end
    
    % User distribution patterns
    if isfield(cfg, 'userDistribution')
        simParams.userDistribution = cfg.userDistribution;
    end
    
    % Additional mobility parameters
    if isfield(cfg, 'mobilityModel')
        simParams.mobilityModel = cfg.mobilityModel;
    end
    
    % Coverage area parameters
    if isfield(cfg, 'maxRadius')
        simParams.maxRadius = cfg.maxRadius;
    end
end

function simParams = validateAndEnhanceConfig(simParams)
% Validate configuration and add derived parameters
    
    % Validate basic parameters
    validateBasicParams(simParams);
    
    % Add scenario-specific validations
    validateScenarioSpecificParams(simParams);
    
    % Add derived parameters
    simParams = addDerivedParams(simParams);
    
    % Set logging configuration
    simParams = configureLogging(simParams);
end

function validateBasicParams(simParams)
% Validate basic simulation parameters
    
    % Check required fields
    requiredFields = {'deploymentScenario', 'numSites', 'numUEs', 'simTime'};
    for i = 1:length(requiredFields)
        field = requiredFields{i};
        if ~isfield(simParams, field)
            error('ScenarioConfig:MissingField', 'Required field missing: %s', field);
        end
    end
    
    % Validate ranges
    if simParams.numSites <= 0
        error('ScenarioConfig:InvalidValue', 'numSites must be positive');
    end
    
    if simParams.numUEs <= 0
        error('ScenarioConfig:InvalidValue', 'numUEs must be positive');
    end
    
    if simParams.simTime <= 0
        error('ScenarioConfig:InvalidValue', 'simTime must be positive');
    end
    
    if simParams.timeStep <= 0 || simParams.timeStep > simParams.simTime
        error('ScenarioConfig:InvalidValue', 'timeStep must be positive and less than simTime');
    end
end

function validateScenarioSpecificParams(simParams)
% Validate scenario-specific parameters
    
    switch simParams.deploymentScenario
        case 'indoor_hotspot'
            validateIndoorParams(simParams);
        case 'dense_urban'
            validateDenseUrbanParams(simParams);
        case 'rural'
            validateRuralParams(simParams);
        case 'urban_macro'
            validateUrbanMacroParams(simParams);
        otherwise
            warning('ScenarioConfig:UnknownScenario', ...
                   'Unknown deployment scenario: %s', simParams.deploymentScenario);
    end
end

function validateIndoorParams(simParams)
% Validate indoor hotspot specific parameters
    
    if simParams.carrierFrequency > 10e9
        warning('ScenarioConfig:HighFrequency', ...
               'High frequency (%.1f GHz) for indoor scenario may cause coverage issues', ...
               simParams.carrierFrequency/1e9);
    end
    
    if simParams.numSites > 20
        warning('ScenarioConfig:ManySites', ...
               'Large number of sites (%d) for indoor scenario', simParams.numSites);
    end
end

function validateDenseUrbanParams(simParams)
% Validate dense urban specific parameters
end

function validateRuralParams(simParams)
% Validate rural specific parameters
    
    if simParams.isd < 1000
        warning('ScenarioConfig:SmallISD', ...
               'Small ISD (%.0fm) for rural scenario', simParams.isd);
    end
    
    if simParams.ueSpeed < 30
        warning('ScenarioConfig:LowSpeed', ...
               'Low UE speed (%.0f km/h) for rural scenario', simParams.ueSpeed);
    end
end

function validateUrbanMacroParams(simParams)
% Validate urban macro specific parameters
    
    if simParams.cellRadius < 200
        warning('ScenarioConfig:SmallRadius', ...
               'Small cell radius (%.0fm) for urban macro', simParams.cellRadius);
    end
end

function simParams = addDerivedParams(simParams)
% Add derived parameters based on scenario
    
    % Calculate total simulation steps
    simParams.totalSteps = ceil(simParams.simTime / simParams.timeStep);
    
    % Set scenario-specific defaults if not specified
    switch simParams.deploymentScenario
        case 'indoor_hotspot'
            simParams.maxRadius = getFieldOrDefault(simParams, 'maxRadius', 100);
        case 'dense_urban'
            simParams.maxRadius = getFieldOrDefault(simParams, 'maxRadius', 500);
        case 'rural'
            simParams.maxRadius = getFieldOrDefault(simParams, 'maxRadius', 2000);
        case 'urban_macro'
            simParams.maxRadius = getFieldOrDefault(simParams, 'maxRadius', 800);
    end
    
    
    simParams.expectedCells = simParams.numSites * simParams.numSectors;
end

function simParams = configureLogging(simParams)
% Configure logging parameters
    
    % Generate log file name based on scenario and timestamp
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    logFileName = sprintf('%s_%s.log', simParams.deploymentScenario, timestamp);
    
    simParams.logFile = logFileName;
    simParams.enableLogging = true;
    simParams.logLevel = getFieldOrDefault(simParams, 'logLevel', 'INFO');
end

function value = getFieldOrDefault(structure, fieldName, defaultValue)
% Helper function to safely get field value with default fallback
    if isfield(structure, fieldName)
        value = structure.(fieldName);
    else
        value = defaultValue;
    end
end