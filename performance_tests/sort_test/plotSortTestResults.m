function plotSortTestResults
    figure;
    x = csvread('sortTimes.txt', 1, 0);
    plot(x(:, 1), x(:, 2) / 10^3, '-', x(:,1), x(:,3) / 10 ^ 3, '-.');
    legend('Integer Keys', 'Custom operator<');
    xlabel('Number of Particles');
    ylabel('Time to Sort (ms)');
end