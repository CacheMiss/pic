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