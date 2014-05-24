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
function plotIonHoles()
    holePeak = 5950;
    holeBeg = 5532;
    holeEnd = 6082;
    plotHole(holePeak, holeBeg, holeEnd);

    holePeak = 7618;
    holeBeg = 7363;
    holeEnd = 7907;
    plotHole(holePeak, holeBeg, holeEnd);
    
    % Plot velocities in regions without holes
    templatePlot(6379, 7328);
    
    templatePlot(8455, 8782);
    templatePlot(9204, 9650);
end

function plotHole(holePeak, holeBeg, holeEnd)
    templatePlot(holePeak, holePeak+10);
    templatePlot(holePeak, holePeak+20);
    templatePlot(holePeak-10, holePeak);
    templatePlot(holePeak-20, holePeak);
    templatePlot(holeBeg, holeEnd);
end

function templatePlot(start, stop)
    plotVxVyRange('ele_160000', ...
                  512/2-50, 512/2+50, ...
                  start, stop, ...
                  'hotOnly', 'noCull', ...
                  'numBins', 50, ...
                  'vxMin', -5, ...
                  'vxMax', 5, ...
                  'vyMin', -5, ...
                  'vyMax', 5, ...
                  'plotVxVy', ...
                  'plotVxLine', 0 ...
                  );
    % Find all windows of type figure, which have an empty FileName attribute.
    allPlots = findall(0, 'Type', 'figure', 'FileName', []);
    % Close.
    delete(allPlots);
    fclose('all');
end