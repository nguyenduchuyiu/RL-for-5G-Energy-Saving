function plotResults(simResults)
% Simple plotting helper. Customize further as needed.
    if isempty(simResults) || ~isfield(simResults, 'energyMetrics')
        return;
    end

    times = [simResults.energyMetrics.time];
    energy = [simResults.energyMetrics.totalEnergy]; % kWh
    active = [simResults.energyMetrics.activeCells];

    figure;
    subplot(2,1,1);
    plot(times, energy);
    ylabel('Total Energy (kWh)');
    title('Energy over time');

    subplot(2,1,2);
    plot(times, active);
    ylabel('Active Cells');
    xlabel('Time (s)');
    title('Active cells over time');
end
