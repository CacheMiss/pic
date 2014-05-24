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
function plotLetterFig3
    data = loadParticles('ele_160000');
    data = data(1);
    f = figure;
    
    subplotWidth = 0.22;
    subplotHeight = 0.88;
    
    xLimits = [512/2-50 512/2+50];
    vxMin = -5;
    vxMax = 5;
    vyMin = -5;
    vyMax = 5;
    numBins = 50;
    numContourLevels = 20;
    
    kernelSize = 3;
    kernel = ones(kernelSize,kernelSize) / kernelSize^2; % NxN mean kernel
    
    nextWidth = 0.07;
    subplot('Position', [nextWidth 0.05 subplotWidth subplotHeight]);
    yLimits = [5532 6082];
    lowerHole = filterParticles(data, xLimits, yLimits);
    [xEdges, yEdges, bins] = ...
           vxDerivative(lowerHole, vxMin, vxMax, vyMin, vyMax, numBins, kernel);
    h = contourf(xEdges(2:end-2), yEdges(1:end-1), bins(1:end-1, 1:end-1), numContourLevels);
    %set(h, 'edgecolor', 'none');
    axis([xEdges(2) xEdges(end-2) yEdges(1) yEdges(end-1)]);
    view(2);
    xlabel('vx');
    ylabel('vy');
    title('Lower Hole');
    colorbar;
    caxis([-0.5 0.5]);
    clear lowerHole;
    
    nextWidth = nextWidth + subplotWidth + 0.01;
    subplot('Position', [nextWidth 0.05 subplotWidth subplotHeight]);
    yLimits = [6379 7328];
    betweenHoles = filterParticles(data, xLimits, yLimits);
    [xEdges, yEdges, bins] = ...
           vxDerivative(betweenHoles, vxMin, vxMax, vyMin, vyMax, numBins, kernel);
    h = contourf(xEdges(2:end-2), yEdges(1:end-1), bins(1:end-1, 1:end-1), numContourLevels);
    %set(h, 'edgecolor', 'none');
    axis([xEdges(2) xEdges(end-2) yEdges(1) yEdges(end-1)]);
    view(2);
    xlabel('vx');
    set(gca, 'YTick', [])
    title('Between Holes');
    colorbar;
    caxis([-0.5 0.5]);
    clear betweenHoles;
    
    nextWidth = nextWidth + subplotWidth + 0.01;
    subplot('Position', [nextWidth 0.05 subplotWidth subplotHeight]);
    yLimits = [7363 7907];
    upperHole = filterParticles(data, xLimits, yLimits);
    [xEdges, yEdges, bins] = ...
           vxDerivative(upperHole, vxMin, vxMax, vyMin, vyMax, numBins, kernel);
    h = contourf(xEdges(2:end-2), yEdges(1:end-1), bins(1:end-1, 1:end-1), numContourLevels);
    %set(h, 'edgecolor', 'none');
    axis([xEdges(2) xEdges(end-2) yEdges(1) yEdges(end-1)]);
    view(2);
    xlabel('vx');
    set(gca, 'YTick', [])
    title('Upper Hole');
    colorbar;
    caxis([-0.5 0.5]);
    clear upperHole;
    
    nextWidth = nextWidth + subplotWidth + 0.01;
    subplot('Position', [nextWidth 0.05 subplotWidth subplotHeight]);
    yLimits = [6379 7328];
    aboveHoles = filterParticles(data, xLimits, yLimits);
    [xEdges, yEdges, bins] = ...
           vxDerivative(aboveHoles, vxMin, vxMax, vyMin, vyMax, numBins, kernel);
    h = contourf(xEdges(2:end-2), yEdges(1:end-1), bins(1:end-1, 1:end-1), numContourLevels);
    %set(h, 'edgecolor', 'none');
    axis([xEdges(2) xEdges(end-2) yEdges(1) yEdges(end-1)]);
    view(2);
    xlabel('vx');
    set(gca, 'YTick', [])
    title('Above Holes');
    colorbar;
    caxis([-0.5 0.5]);
    clear aboveHoles;
    
    saveSameSize(f, 'format', '-dpdfwrite', 'file', 'figure3');
end

function [xEdges, yEdges, bins] = vxDerivative(p, vxMin, vxMax, vyMin, vyMax, numBins, kernel)
   xEdges = linspace(vxMin, vxMax, numBins);
   yEdges = linspace(vyMin, vyMax, numBins);
   % Bin all the values
   tmpBins = hist2(p.vx, p.vy, xEdges, yEdges);
   % Smooth the bins with an average filter
   tmpBins = conv2(tmpBins, kernel, 'same');
   
   % Calculate the derivative
   bins = (tmpBins(:, 3:end) - tmpBins(:, 1:end-2)) ./ tmpBins(:, 2:end-1) / 2 ;
   % Remove Infinities and Nans. These are from cells with a value of 0
   bins(isinf(bins)) = 0;
   bins(isnan(bins)) = 0;
end

function ret = filterParticlesWithLogical(p, logical)
   ret.x = p.x(logical);
   ret.y = p.y(logical);
   ret.vx = p.vx(logical);
   ret.vy = p.vy(logical);
   ret.vz = p.vz(logical);
end

function ret = filterParticles(p, xLimits, yLimits)
   logArray = p.y >= yLimits(1) & ...
              p.y <= yLimits(2) & ...
              p.x >= xLimits(1) & ...
              p.x <= xLimits(2);
   ret = filterParticlesWithLogical(p, logArray);
end