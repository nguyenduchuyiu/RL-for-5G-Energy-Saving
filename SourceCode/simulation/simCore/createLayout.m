function sites = createLayout(simParams, seed)
% Create site layout based on 3GPP scenario requirements
% Refactored for better maintainability and extensibility

    siteRng = RandStream('mt19937ar', 'Seed', seed + 1000);
    prevStream = RandStream.setGlobalStream(siteRng);

    % Use scenario-specific layout creators
    layoutCreators = getLayoutCreators();
    
    if isfield(layoutCreators, simParams.deploymentScenario)
        sites = layoutCreators.(simParams.deploymentScenario)(simParams, seed);
    else
        warning('Unknown deployment scenario: %s. Using default hexagonal layout.', simParams.deploymentScenario);
        sites = createHexLayout(simParams.numSites, simParams.isd, seed);
    end

    RandStream.setGlobalStream(prevStream);
    fprintf('Created %d sites for %s scenario\n', length(sites), simParams.deploymentScenario);
end

function creators = getLayoutCreators()
% Registry of scenario-specific layout creators
    creators = struct();
    creators.indoor_hotspot = @createIndoorLayout;
    creators.dense_urban = @createDenseUrbanLayout;
    creators.rural = @createRuralLayout;
    creators.urban_macro = @createUrbanMacroLayout;
end

function sites = createIndoorLayout(simParams, seed)
% Create indoor office layout with TRxPs in 120m x 50m area
    
    config = getScenarioConfig('indoor_hotspot');
    sites = createGridLayout(simParams.numSites, config.dimensions, config.gridSize, 'indoor_trxp');
end

function sites = createDenseUrbanLayout(simParams, seed)
% Create dense urban layout with macro and micro sites
    
    % Create macro sites in hexagonal pattern
    macroSites = createHexLayout(simParams.numSites, simParams.isd, seed);
    
    sites = macroSites;
end

function sites = createRuralLayout(simParams, seed)
% Create rural layout with widely spaced macro sites
    
    % Rural uses hexagonal grid with larger ISD
    sites = createHexLayout(simParams.numSites, simParams.isd, seed);
    
    % Adjust site types for rural scenario
    for i = 1:length(sites)
        sites(i).type = 'rural_macro';
    end
end

function sites = createUrbanMacroLayout(simParams, seed)
% Create urban macro layout with hexagonal grid
    
    sites = createHexLayout(simParams.numSites, simParams.isd, seed);
    
    % Set site type for urban macro
    for i = 1:length(sites)
        sites(i).type = 'urban_macro';
    end
end

function sites = createGridLayout(numSites, dimensions, gridSize, siteType)
% Generic grid layout creator
    
    floorWidth = dimensions.width;
    floorHeight = dimensions.height;
    cols = gridSize.cols;
    rows = gridSize.rows;
    
    xSpacing = floorWidth / (cols + 1);
    ySpacing = floorHeight / (rows + 1);
    
    % Initialize as empty struct array
    sites = struct('id', {}, 'x', {}, 'y', {}, 'type', {});
    siteIdx = 1;
    
    for row = 1:rows
        for col = 1:cols
            if siteIdx <= numSites
                sites(siteIdx) = struct(...
                    'id', siteIdx, ...
                    'x', col * xSpacing, ...
                    'y', row * ySpacing, ...
                    'type', siteType ...
                );
                siteIdx = siteIdx + 1;
            end
        end
    end
end

function sites = createHexLayout(numSites, isd, seed)
% Create hexagonal layout pattern
    
    rng(seed + 1000, 'twister');
    
    % Initialize as empty struct array with proper fields
    sites = struct('id', {}, 'x', {}, 'y', {}, 'type', {});
    
    % Central site
    sites(1) = struct('id', 1, 'x', 0, 'y', 0, 'type', 'macro');
    
    if numSites == 1
        return;
    end
    
    % Create concentric hexagonal rings
    siteIdx = 2;
    ring = 1;
    maxRings = 5; % Prevent infinite loops
    
    while siteIdx <= numSites && ring <= maxRings
        ringSites = createHexRing(ring, isd, siteIdx);
        
        for i = 1:length(ringSites)
            if siteIdx <= numSites
                sites(siteIdx) = ringSites(i);
                siteIdx = siteIdx + 1;
            end
        end
        ring = ring + 1;
    end
    
    % Fill remaining sites randomly if needed
    sites = fillRemainingSites(sites, numSites, siteIdx, isd);
end

function ringSites = createHexRing(ring, isd, startIdx)
% Create sites in a hexagonal ring
    
    % Initialize as empty struct array
    ringSites = struct('id', {}, 'x', {}, 'y', {}, 'type', {});
    siteIdx = startIdx;
    ringIdx = 1;
    
    for side = 0:5
        for pos = 0:(ring-1)
            angle = side * pi/3;
            x = isd * ring * cos(angle) + pos * isd * cos(angle + pi/3);
            y = isd * ring * sin(angle) + pos * isd * sin(angle + pi/3);
            
            ringSites(ringIdx) = struct('id', siteIdx, 'x', x, 'y', y, 'type', 'macro');
            siteIdx = siteIdx + 1;
            ringIdx = ringIdx + 1;
        end
    end
end

function sites = fillRemainingSites(sites, numSites, currentIdx, isd)
% Fill remaining sites with random positions
    
    while currentIdx <= numSites
        angle = rand() * 2 * pi;
        distance = isd + rand() * (isd * 2);
        x = distance * cos(angle);
        y = distance * sin(angle);
        
        sites(currentIdx) = struct('id', currentIdx, 'x', x, 'y', y, 'type', 'macro');
        currentIdx = currentIdx + 1;
    end
end

function config = getScenarioConfig(scenarioType)
% Get scenario-specific configuration parameters
    
    configs = struct();
    
    % Indoor hotspot configuration
    configs.indoor_hotspot = struct(...
        'dimensions', struct('width', 120, 'height', 50), ...
        'gridSize', struct('cols', 4, 'rows', 3) ...
    );
    
    % Micro site placement configuration
    configs.micro_placement = struct(...
        'minDistance', 20, ...
        'maxDistance', 100 ...
    );
    
    if isfield(configs, scenarioType)
        config = configs.(scenarioType);
    else
        error('Unknown scenario type: %s', scenarioType);
    end
end