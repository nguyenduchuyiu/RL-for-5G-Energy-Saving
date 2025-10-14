function main_run_scenarios()
% Runs all scenarios and outputs energies.txt

    % Define test suite (matching opts.txt format)
    suite = [
        struct('name','indoor_hotspot', 'seed',42)  % kWh
        struct('name', 'dense_urban', 'seed', 42)  % kWh
        struct('name', 'rural', 'seed', 42)  % kWh
        struct('name', 'urban_macro', 'seed', 42)  % kWh
    ];
    
    % Add simulation path if needed
    if exist('simulation', 'dir')
        addpath('simulation');
    end
    
    % Run benchmark suite
    results = runBenchmarkSuite(suite);
    
    fprintf('energies.txt generated with %d values\n', length(results.energies));
    
    % Display energy values for verification
    fprintf('\nEnergy values written to energies.txt:\n');
    for i = 1:length(results.energies)
        fprintf('  Scenario %d (%s): %.6f kWh\n', i, suite(i).name, results.energies(i));
    end
        
end