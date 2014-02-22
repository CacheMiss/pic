function plotPotentPerformance()
   f = figure;
   x = csvread('potentPerformance.csv', 1, 0);
   widthList = [32 64 128 256 512 1024];
   %  1 = Height  2500
   %  2 = Height  5000
   %  3 = Height  7500
   %  4 = Height 10000
   %  5 = Height 22500
   %  6 = Height 25000
   %  7 = Height 27500
   %  8 = Height 20000
   %  9 = Height 32500
   % 10 = Height 35000
   % 11 = Height 37500
   % 12 = Height 30000
   plot(widthList, x(:, 2), ':', ...
        widthList, x(:, 4), '--', ...
        widthList, x(:, 8), '-.', ...
        widthList, x(:, 12), '-' ...
        );
   xlabel('Grid Width');
   ylabel('Milliseconds');
   legend( ...
       'Grid Height = 5000', ...
       'Grid Height = 10000', ...
       'Grid Height = 20000', ...
       'Grid Height = 30000' ...
       );
   axis([32 1024 0 100]);
   saveSameSize(f, 'format', '-dpdfwrite', 'file', 'potent2Performance');
end