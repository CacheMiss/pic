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
function plotEarlySideBySide()
    % Plot across many holes
    plotRange(2953, 6174);
end

function plotRange(start, stop)
    f = figure;
    
    logTime = '051400';
    eleName = strcat('ele_', logTime);
    ionName = strcat('ion_', logTime);
    templatePlot(eleName, start, stop, 1, true, 'Ele Hot');
    templatePlot(eleName, start, stop, 2, false, 'Ele Cold');
    templatePlot(ionName, start, stop, 3, true, 'Ion Hot');
    templatePlot(ionName, start, stop, 4, false, 'Ion Cold');
    
    annotation('textbox', [0 0.9 1 0.1], ...
    'String',     strcat('Vy vs. Y Time=', logTime, ...
    'Y=', num2str(start), '-', num2str(stop)), ...
    'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center');
    
    print(f, '-dpng', strcat('sideBySideVyY_', logTime));
end

function templatePlot(fileName, start, stop, plotNum, isHot, subplotTitle)
    width = 20;
    subplotArgs = [1, 4, plotNum];
    if isHot
        selectorArg = 'hotOnly';
    else
        selectorArg = 'coldOnly';
    end
    plotVxVyRange(fileName, ...
                  512/2-width, 512/2+width, ...
                  start, stop, ...
                  'noCull', ...
                  'numBins', 50, ...
                  'plotVyY', ...
                  'noColorbar', ...
                  selectorArg, ...
                  'subplotArgs', subplotArgs, ...
                  'subplotTitle', subplotTitle ...
                  );
    fclose('all');
end