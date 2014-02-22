
function ret = plotPerformance(dir1, dir2, tag1, tag2, varargin)

   file1 = strcat(dir1, '/performance.csv');
   file2 = strcat(dir2, '/performance.csv');
   
   optArgs = parseArgs(varargin);
   
   data1 = loadPerformance(file1, optArgs);
   data2 = loadPerformance(file2, optArgs);
   
   figure;
   plot(data1.simTime, data1.iterationTime, ...
        data2.simTime, data2.iterationTime);
   legend(tag1, tag2);
   title('Iteration Time Comparison');
   xlabel('Sim Time');
   ylabel('Time (ms)');
   
   [d1Subset d2Subset] = perfIntersection(data1, data2);
   
   figure;
   plot(d1Subset.simTime, d2Subset.iterationTime ./d1Subset.iterationTime);
   title(strcat([tag1 ' speedup over time']));
   xlabel('Sim Time');
   ylabel('Speedup Factor');
   
%    figure;
%    plot(d1Subset.simTime, d2Subset.injectTime ./d1Subset.injectTime);
%    title(strcat([tag1 ' inject speedup over time']));
%    xlabel('Sim Time');
%    ylabel('Speedup Factor');
   
   figure;
   plot(d1Subset.simTime, d2Subset.densTime ./d1Subset.densTime);
   title(strcat([tag1 ' dens speedup over time']));
   xlabel('Sim Time');
   ylabel('Speedup Factor');
   
   figure;
   plot(d1Subset.simTime, d2Subset.fieldTime ./d1Subset.fieldTime);
   title(strcat([tag1 ' field speedup over time']));
   xlabel('Sim Time');
   ylabel('Speedup Factor');
   
   figure;
   plot(d1Subset.simTime, d2Subset.moveTime ./d1Subset.moveTime);
   title(strcat([tag1 ' move speedup over time']));
   xlabel('Sim Time');
   ylabel('Speedup Factor');
   
   plotRunHistograph(d1Subset, tag1);
   plotRunHistograph(d2Subset, tag2);
end

function plotRunHistograph(data, tag)
   iterationTimes = ...
       data.injectTime + ...
       data.densTime + ...
       data.potentTime + ...
       data.fieldTime + ...
       data.moveTime;
   field = data.fieldTime ./ iterationTimes;
   move = data.moveTime ./ iterationTimes + field;
   dens = data.densTime ./ iterationTimes + move;
   potent = data.potentTime ./ iterationTimes + dens;
   inject = ones(1, size(dens,1));
   
   injectColor = [1,0,0];
   densColor = [0.2,0.8,0];
   potentColor = [0,0.2,0.8];
   moveColor = [0.8, 0.8, 0.8];
   fieldColor = [0.2, 0.2, 0.2];
   
   figure;
   area(data.simTime, inject, 'FaceColor', injectColor);
   hold on;
   area(data.simTime, potent, 'FaceColor', potentColor);
   area(data.simTime, dens, 'FaceColor', densColor);
   area(data.simTime, move, 'FaceColor', moveColor);
   area(data.simTime, field, 'FaceColor', fieldColor);
   title(strcat([tag ' Runtime']));
   xlabel('Sim Time');
   ylabel('Runtime Division');
   legend('Inject', 'Potent', 'Dens', 'Move', 'Field');
   hold off;
end

function [d1Subset d2Subset] = perfIntersection(data1, data2)
   [C, ia, ib] = intersect(data1.simTime, data2.simTime);
   d1Subset = trimPerformanceStruct(data1, ia);
   d2Subset = trimPerformanceStruct(data2, ib);
end

function ret = trimPerformanceStruct(d, logical)
   ret.iteration = d.iteration(logical);
   ret.simTime = d.simTime(logical);
   ret.numEleHot = d.numEleHot(logical);
   ret.numEleCold = d.numEleCold(logical);
   ret.numIonHot = d.numIonHot(logical);
   ret.numIonCold = d.numIonCold(logical);
   ret.iterationTime = d.iterationTime(logical);
   ret.injectTime = d.injectTime(logical);
   ret.densTime = d.densTime(logical);
   ret.potentTime = d.potentTime(logical);
   ret.fieldTime = d.fieldTime(logical);
   ret.moveTime = d.moveTime(logical);
end

function ret = loadPerformance(fileName, optArgs)
   %labels = ['Iteration Num', 'Sim Time', 'Num Ele Hot', 'Num Ele Cold' \
   %          'Num Ion Hot', 'Num Ion Cold', 'Iteration Time (ms)',
   %          'Inject Time', 'Dens Time', 'Potent2 Time',
   %          'Field Time', 'Movep Time'];
   
   d = load(fileName);
   
   if optArgs.maxTime ~= inf
       logical = d(:,2) < optArgs.maxTime;
       d = d(logical, :);
   end
   if optArgs.limitExeTime ~= inf
       logical = d(:,7) < optArgs.limitExeTime;
       d = d(logical, :);
   end
   
   ret.iteration = d(:,1);
   ret.simTime = d(:,2);
   ret.numEleHot = d(:,3);
   ret.numEleCold = d(:,4);
   ret.numIonHot = d(:,5);
   ret.numIonCold = d(:,6);
   ret.iterationTime = d(:,7);
   ret.injectTime = d(:,8);
   ret.densTime = d(:,9);
   ret.potentTime = d(:,10);
   ret.fieldTime = d(:,11);
   ret.moveTime = d(:,12);   
end

function ret = smoothLine(line, window)
   halfWin = window / 2;
   lineSize = size(line);
   lineSize = lineSize(1);
   ret = zeros(lineSize, 1);
   for i=1 : lineSize
       if i < halfWin+1
           ret(i) = mean(line(1:i+halfWin));
       elseif i < lineSize-halfWin
           ret(i) = mean(line(i-halfWin:i+halfWin));
       else
           ret(i) = mean(line(i-halfWin:end));
       end
   end
end

function ret = parseArgs(args)
    ret = struct( ...
        'limitExeTime', inf, ...
        'maxTime', inf ...
        );
    if ~isempty(args)
        i = 1;
        while i <= length(args)
            % Use this to filter iterations that took way too long
            if strcmp(args{i}, 'limitExeTime')
                ret.limitExeTime = args{i+1};
                i = i + 1;
            elseif strcmp(args{i}, 'maxTime')
                ret.maxTime = args{i+1};
                i = i + 1;
            else
                error('Invalid option!');
            end
            i = i + 1;
        end
    end
end
