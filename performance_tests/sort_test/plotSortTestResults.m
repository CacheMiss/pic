function plotSortTestResults
    x = csvread('sortTimes.txt', 1, 0);
    
    f = figure;
    plot(x(:, 1), x(:, 2), '-', ...
         x(:,1), x(:,3), '-.', ...
         x(:,1), x(:,5), '--');
    legend('Integer Keys', 'Custom operator<', 'Two Integer Sorts');
    xlabel('Number of Particles');
    ylabel('Time to Sort (ms)');
    saveSameSize(f, 'format', '-dpdfwrite', 'file', 'sortBenchmark');
    
    f = figure;
    plot(x(:, 1), x(:, 5), '-', ...
         x(:,1), x(:,4), '-.', ...
         x(:,1), x(:,7), ':');
    legend('GPU Sort', 'CPU Sort(with copy)', 'CPU Sort(without copy)');
    xlabel('Number of Particles');
    ylabel('Time to Sort (ms)');
    saveSameSize(f, 'format', '-dpdfwrite', 'file', 'gpuCpuSortComparison');
    
    f = figure;
    plot(x(:, 1), x(:, 6), '-', ...
         x(:,1), x(:,7), '-.', ...
         x(:,1), x(:,8), ':');
    legend('Copy To Host', 'CPU Sort', 'Copy To Device');
    xlabel('Number of Particles');
    ylabel('Time to Sort (ms)');
    saveSameSize(f, 'format', '-dpdfwrite', 'file', 'cpuSortTimeBreakdown');
    
    f = figure;
    plot(x(:, 1), x(:, 5), '-', ...
         x(:,1), x(:,7), '-.');
    legend('GPU Sort', 'CPU Sort');
    xlabel('Number of Particles');
    ylabel('Time to Sort (ms)');
    saveSameSize(f, 'format', '-dpdfwrite', 'file', 'gpuCpuSortNoCopy');
end