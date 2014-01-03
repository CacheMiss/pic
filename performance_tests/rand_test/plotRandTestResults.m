function plotRandTestResults
    figure;
    x = csvread('randTimes.txt', 1, 0);
    plot(x(:, 1), x(:, 2) / 10^6);
    xlabel('Sim Time');
    ylabel('Seconds');
end
