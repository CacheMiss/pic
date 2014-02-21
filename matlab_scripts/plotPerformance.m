
function ret = plotPerformance(fName, varargin)

   d = load(fName);
   
   optArgs = parseArgs(varargin);
   
   if optArgs.maxTime ~= inf
       logical = d(:,2) < optArgs.maxTime;
       d = d(logical, :);
   end
   if optArgs.limitExeTime ~= inf
       logical = d(:,7) < optArgs.limitExeTime;
       d = d(logical, :);
   end

   %labels = ['Iteration Num', 'Sim Time', 'Num Ele Hot', 'Num Ele Cold' \
   %          'Num Ion Hot', 'Num Ion Cold', 'Iteration Time (ms)',
   %          'Inject Time', 'Dens Time', 'Potent2 Time',
   %          'Field Time', 'Movep Time'];

   figure;
   plot(d(:,2), smoothLine(d(:,7),256));
   xlabel('Sim Time');
   ylabel('Iteration Time (ms)');
   print('-dpng', 'it_time_vs_sim_time');
   totalParticles = d(:,3) + d(:,4) + d(:,5) + d(:,6);
   figure;
   plot(totalParticles, smoothLine(d(:,7),256));
   %plot(totalParticles, smoothn(d(:,7)));
   xlabel('Num Particles');
   ylabel('Iteration Time (ms)');
   print('-dpng', 'particles_vs_it_time');
   figure;
   plot(d(:,2), totalParticles);
   xlabel('Sim Time (s)');
   ylabel('Num Particles');
   print('-dpng', 'particles_vs_sim_time');
   figure;
   plot(d(:,2), smoothLine(d(:,9),16));
   xlabel('Sim Time');
   ylabel('Dens Time (ms)');
   print('-dpng', 'dens_time_vs_sim_time');
   figure;
   plot(d(:,2), smoothLine(d(:,12),16));
   xlabel('Sim Time');
   ylabel('Movep Time (ms)');
   print('-dpng', 'movep_time_vs_sim_time');

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