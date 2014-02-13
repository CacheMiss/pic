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