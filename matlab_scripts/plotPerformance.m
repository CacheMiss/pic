%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2014, Stephen C. Sewell
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
% 1. Redistributions of source code must retain the above copyright notice, this
%    list of conditions and the following disclaimer.
% 
% 2. Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the documentation
%    and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
% ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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