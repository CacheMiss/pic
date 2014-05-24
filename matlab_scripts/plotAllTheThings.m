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
function plotAllTheThings(startIdx, endIdx, stepSize, gridWidth)
    for i=startIdx:stepSize:endIdx
        idxStr = sprintf('%06d', i);
        eleName = strcat('ele_', idxStr);
        ionName = strcat('ion_', idxStr);
        rhoName = strcat('rho_', idxStr);
        rhoeName = strcat('rhoe_', idxStr);
        rhoiName = strcat('rhoi_', idxStr);
        phiName = strcat('phi_', idxStr);
        phiAvgName = strcat('phiAvg_', idxStr);
        
        partStepSize = 512;
        
        plotPart(eleName, partStepSize);
        plotPart(ionName, partStepSize);
        %plotVxRange(eleName, gridWidth/2, 20, partStepSize, partStepSize);
        %plotVxRange(ionName, gridWidth/2, 20, partStepSize, partStepSize);
        %plotVyRange(eleName, gridWidth/2, 20, partStepSize, partStepSize);
        %plotVyRange(ionName, gridWidth/2, 20, partStepSize, partStepSize);
        plotVxVy(eleName, 1, 1);
        plotVxVy(ionName, 1, 1);
        %plotVxVyRange(eleName, gridWidth/2-32, gridWidth+32, 0, 3000);
        %plotVxVyRange(ionName, gridWidth/2-32, gridWidth+32, 0, 3000);
        plotRho(idxStr, gridWidth/2);
        contourPhi(rhoName, 4, 32);
        plotPhiAll(rhoiName, 8, 64);
        contourRho(rhoiName, 4, 32);
        %contourRhoBottom(rhoiName);
        plotPhi(phiName, gridWidth/2);
        plotPhi(phiAvgName, gridWidth/2);
        plotPhiAll(phiName, 8, 64);
        contourPhi(phiName, 4, 32);
        contourPhi(phiAvgName, 4, 32);
        
        % Find all windows of type figure, which have an empty FileName attribute.
        allPlots = findall(0, 'Type', 'figure', 'FileName', []);
        % Close.
        delete(allPlots);
        fclose('all');
    end
end