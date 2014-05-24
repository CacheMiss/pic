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
function plotLetterFig2
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
    
    kernelSize = 3;
    kernel = ones(kernelSize,kernelSize) / kernelSize^2; % NxN mean kernel
    
    nextWidth = 0.07;
    subplot('Position', [nextWidth 0.05 subplotWidth subplotHeight]);
    yLimits = [5532 6082];
    lowerHole = filterParticles(data, xLimits, yLimits);
    xEdges = linspace(vxMin, vxMax, numBins);
    yEdges = linspace(vyMin, vyMax, numBins);
    bins = hist2(lowerHole.vx, lowerHole.vy, xEdges, yEdges);
    bins = conv2(bins, kernel, 'same');
    contourf(xEdges, yEdges, bins);
    xlabel('vx');
    ylabel('vy');
    %axis([vxMin vxMax vyMin vyMax]);
    title('Lower Hole');
    clear lowerHole;
    
    nextWidth = nextWidth + subplotWidth + 0.01;
    subplot('Position', [nextWidth 0.05 subplotWidth subplotHeight]);
    yLimits = [6379 7328];
    betweenHoles = filterParticles(data, xLimits, yLimits);
    xEdges = linspace(vxMin, vxMax, numBins);
    yEdges = linspace(vyMin, vyMax, numBins);
    bins = hist2(betweenHoles.vx, betweenHoles.vy, xEdges, yEdges);
    bins = conv2(bins, kernel, 'same');
    contourf(xEdges, yEdges, bins);
    xlabel('vx');
    set(gca, 'YTick', [])
    %axis([vxMin vxMax vyMin vyMax]);
    title('Between Holes');
    clear betweenHoles;
    
    nextWidth = nextWidth + subplotWidth + 0.01;
    subplot('Position', [nextWidth 0.05 subplotWidth subplotHeight]);
    yLimits = [7363 7907];
    upperHole = filterParticles(data, xLimits, yLimits);
    xEdges = linspace(vxMin, vxMax, numBins);
    yEdges = linspace(vyMin, vyMax, numBins);
    bins = hist2(upperHole.vx, upperHole.vy, xEdges, yEdges);
    bins = conv2(bins, kernel, 'same');
    contourf(xEdges, yEdges, bins);
    xlabel('vx');
    set(gca, 'YTick', [])
    %axis([vxMin vxMax vyMin vyMax]);
    title('Upper Hole');
    clear upperHole;
    
    nextWidth = nextWidth + subplotWidth + 0.01;
    subplot('Position', [nextWidth 0.05 subplotWidth subplotHeight]);
    yLimits = [6379 7328];
    aboveHoles = filterParticles(data, xLimits, yLimits);
    xEdges = linspace(vxMin, vxMax, numBins);
    yEdges = linspace(vyMin, vyMax, numBins);
    bins = hist2(aboveHoles.vx, aboveHoles.vy, xEdges, yEdges);
    bins = conv2(bins, kernel, 'same');
    contourf(xEdges, yEdges, bins);
    xlabel('vx');
    set(gca, 'YTick', [])
    %axis([vxMin vxMax vyMin vyMax]);
    title('Above Holes');
    clear aboveHoles;
    
    saveSameSize(f, 'format', '-dpdfwrite', 'file', 'figure2');
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