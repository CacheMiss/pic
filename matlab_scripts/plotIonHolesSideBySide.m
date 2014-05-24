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
function plotIonHolesSideBySide()
    f = figure;
    
    numSideBySide = 4;
    holeBeg = 5532;
    holeEnd = 6082;
    templatePlot(holeBeg, holeEnd, numSideBySide, 4);
    
    templatePlot(6379, 7328, numSideBySide, 3);

    holeBeg = 7363;
    holeEnd = 7907;
    templatePlot(holeBeg, holeEnd, numSideBySide, 2);
    
    templatePlot(8455, 8782, numSideBySide, 1);
    
    annotation('textbox', [0 0.9 1 0.1], ...
    'String', 'Hot Electrons: Time = 160000', ...
    'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center')
    
    print(f, '-dpng', 'sideBySideVxVy');
end

function templatePlot(start, stop, numSideBySide, plotNumber)
    subplotTitle = strcat('y=', num2str(start), '-', num2str(stop));
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
                  'subplotArgs', [1 numSideBySide, plotNumber], ...
                  'subplotTitle', subplotTitle ...
                  );
    fclose('all');
end