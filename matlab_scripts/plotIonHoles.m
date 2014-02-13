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