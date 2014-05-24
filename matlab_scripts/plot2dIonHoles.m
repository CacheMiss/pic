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
function plot2dIonHoles()
    holeBeg = 5532;
    holeEnd = 6082;
    p1 = templatePlot(holeBeg, holeEnd);

    % Plot region between holes
    p2 = templatePlot(6379, 7328);
    
    holeBeg = 7363;
    holeEnd = 7907;
    p3 = templatePlot(holeBeg, holeEnd);
    
    % Plot region above holes
    p4 = templatePlot(8455, 8782);
    
    p1 = normalizeDataXY(p1.hot.vxLineData);
    p2 = normalizeDataXY(p2.hot.vxLineData);
    p3 = normalizeDataXY(p3.hot.vxLineData);
    p4 = normalizeDataXY(p4.hot.vxLineData);
    
    f = figure;
    plot(p1.x, p1.y, '-', ...
         p2.x, p2.y, '--', ...
         p3.x, p3.y, '-.', ...
         p4.x, p4.y, ':');
    ylabel('Normalized Particle Quantity');
    xlabel('vy');
    legend('Lower Hole', ...
           'Between Holes', ...
           'Upper Hole', ...
           'Above Holes');
    title('VY where VX = 0');
    print(f, '-dpng', 'vy_for_vx0');
end

function ret = normalizeDataXY(d)
    ret.x = d.x;
    ret.y = normalizeData(d.y);
end

function ret = normalizeData(d)
    top = max(d);
    bottom = min(d);
    difference = top - bottom;
    scaleFactor = 1 / difference;
    ret = (d - bottom) * scaleFactor;
end

function ret = templatePlot(start, stop)
    ret = plotVxVyRange('ele_160000', ...
                        512/2-50, 512/2+50, ...
                        start, stop, ...
                        'hotOnly', 'noCull', ...
                        'numBins', 50, ...
                        'vxMin', -5, ...
                        'vxMax', 5, ...
                        'vyMin', -5, ...
                        'vyMax', 5, ...
                        'plotVxLine', 0, ...
                        'getVxLineData' ...
                        );
    fclose('all');
end