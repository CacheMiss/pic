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
function plotEarlyHoles()
    holeBeg = 2953;
    holeEnd = 3481;
    plotRange(holeBeg, holeEnd);
    
    holeBeg = 4078;
    holeEnd = 5158;
    plotRange(holeBeg, holeEnd);
    
    holeBeg = 5158;
    holeEnd = 5811;
    plotRange(holeBeg, holeEnd);

    % Plot velocities in regions without holes
    plotRange(6500, 7300);
    
    % Plot across many holes
    plotRange(2953, 6174);
end

function plotRange(start, stop)
    logTime = '051400';
    fileName = strcat('ele_', logTime);
    templatePlot(fileName, start, stop);
    fileName = strcat('ion_', logTime);
    templatePlot(fileName, start, stop);
end

function templatePlot(fileName, start, stop)
    width = 20;
    plotVxVyRange(fileName, ...
                  512/2-width, 512/2+width, ...
                  start, stop, ...
                  'noCull', ...
                  'numBins', 50, ...
                  'plotVyY' ...
                  );
    % Find all windows of type figure, which have an empty FileName attribute.
    allPlots = findall(0, 'Type', 'figure', 'FileName', []);
    % Close.
    delete(allPlots);
    fclose('all');
end