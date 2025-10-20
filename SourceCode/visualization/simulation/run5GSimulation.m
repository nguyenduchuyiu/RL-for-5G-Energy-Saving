function simResults = run5GSimulation(scenarioInput, seed)
% Entry point to run the 5G energy saving simulation with 3GPP scenarios
% Usage:
%   run5GSimulation()                                -> default indoor_hotspot
%   run5GSimulation('indoor_hotspot')               -> Indoor hotspot scenario
%   run5GSimulation('dense_urban')                  -> Dense urban scenario
%   run5GSimulation('scenarios/custom.json')        -> Custom JSON file
%   run5GSimulation(..., 123)                      -> Custom RNG seed

    if nargin < 1 || isempty(scenarioInput)
        scenarioInput = 'indoor_hotspot';
    end
    if nargin < 2 || isempty(seed)
        seed = 42;
    end

    rng(seed, 'twister');

    % Load scenario config
    simParams = loadScenarioConfig(scenarioInput);

    % Create logs directory if it doesn't exist
    if ~exist('logs', 'dir')
        mkdir('logs');
    end
    
    timestamp = datestr(datetime('now'), 'yyyymmdd_HHMMSS');
    
    % Update log file paths
    simParams.logFile = sprintf('logs/%s_energy_saving.log', timestamp);
    simParams.ueLogFile = sprintf('logs/%s_ue.log', timestamp);
    simParams.cellLogFile = sprintf('logs/%s_cell.log', timestamp);
    simParams.agentLogFile = sprintf('logs/%s_agent.log', timestamp);
    simParams.handoverLogFile = sprintf('logs/%s_handover.log', timestamp);

    try
        fid = fopen(simParams.logFile, 'w');
        if fid == -1
            error('Could not create log file: %s', simParams.logFile);
        end
        fprintf(fid, 'Simulation started: %s\n', datestr(now));
        fclose(fid);
        
        try
            ESAgent = ESInterface('n_cells', simParams.numSites * simParams.numSectors, ...
                'max_time', simParams.simTime / simParams.timeStep, 'num_ue', simParams.numUEs);
        catch
            fprintf('Failed to initialize ESAgent with GPU, terminating simulation\n');
            simResults = [];
            return;
        end

        simResults = simulate5GNetwork(simParams, ESAgent, seed);
        % plotResults(simResults);
        
    catch ME
        fprintf('Simulation failed: %s\n', ME.message);
        rethrow(ME);
    end
end











