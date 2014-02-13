function ret = plotVxVyRange(fName, xMin, xMax, yMin, yMax, varargin)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end
   
   optArgs = parseArgs(varargin);
   maxPlottableParticles = optArgs.maxPoints;
   
   [hotP coldP] = loadParticles(fName, ...
       'enableCulling', false, ...
       'loadHot', optArgs.plotHot, ...
       'loadCold', optArgs.plotCold);
   hotP = filterParticles(hotP, xMin, xMax, yMin, yMax);
   coldP = filterParticles(coldP, xMin, xMax, yMin, yMax);
   numHotCulled = size(hotP.x, 1);
   numColdCulled = size(coldP.x, 1);
   
   if optArgs.enableCulling
       if optArgs.plotHot && numHotCulled > maxPlottableParticles
           sliceHot = floor(numHotCulled / maxPlottableParticles);
           hotP = particleCull(hotP, sliceHot);
           numHotCulled = maxPlottableParticles;
           fprintf('Limiting number of hot particles plotted to %d\n', ...
               numHotCulled);
       end
       if optArgs.plotCold && numColdCulled > maxPlottableParticles
           sliceCold = floor(numColdCulled / maxPlottableParticles);
           coldP = particleCull(coldP, sliceCold);
           numColdCulled = maxPlottableParticles;
           fprintf('Limiting number of cold particles plotted to %d\n', ...
               numColdCulled);
       end
   end
   
   % Randomize these vectors for faster plotting
   if optArgs.plotHot
       hotP = particleMix(hotP);
   end
   if optArgs.plotCold
       coldP = particleMix(coldP);
   end
   
   if optArgs.plotHot && size(hotP.x, 1) ~= 0 && size(hotP.x, 2) ~= 0
       ret.hot = plotParticles(fName, hotP, 'hot', xMin, xMax, yMin, yMax, optArgs);
   end
   
   if optArgs.plotCold && size(coldP.x, 1) ~= 0 && size(coldP.x, 2) ~= 0
       ret.cold = plotParticles(fName, coldP, 'cold', xMin, xMax, yMin, yMax, optArgs);
   end
end

function ret = plotParticles(fName, p, descriptor, xMin, xMax, yMin, yMax, optArgs)
    ret = struct();
    stdDevMultiplier = 3;
    numBins = optArgs.numBins;
    kernelSize = 3;
    kernel = ones(kernelSize,kernelSize) / kernelSize^2; % NxN mean kernel

    if size(p.x, 1) ~= 0
       stdVx = std(p.vx);
       meanVx = mean(p.vx);
       vxMax = meanVx + stdVx * stdDevMultiplier;
       vxMin = meanVx - stdVx * stdDevMultiplier;
       stdVy = std(p.vy);
       meandVy = mean(p.vy);
       vyMax = meandVy + stdVy * stdDevMultiplier;
       vyMin = meandVy - stdVy * stdDevMultiplier;
   else
       vxMax = 1;
       vxMin = 0;
       vyMax = 1;
       vyMin = 0;
    end
   
    if optArgs.vxMax ~= Inf
       vxMax = optArgs.vxMax;
    end
    if optArgs.vxMin ~= Inf
       vxMin = optArgs.vxMin;
    end
    if optArgs.vyMax ~= Inf
       vyMax = optArgs.vyMax;
    end
    if optArgs.vyMin ~= Inf
       vyMin = optArgs.vyMin;
    end

    if optArgs.plotVxY
       % Vx vs. Y
       figureOrSubplot(optArgs.useSubplot, optArgs.subplotArgs);
       xEdges = linspace(vxMin, vxMax, numBins);
       yEdges = linspace(yMin, yMax, numBins);
       bins = hist2(p.vx, p.y, xEdges, yEdges);
       bins = conv2(bins, kernel, 'same');
       if optArgs.logScale
           bins = log(bins);
       end
       contourf(xEdges, yEdges, bins);
       if optArgs.showColorbar
           colorbar;
       end
       xlabel('vx');
       ylabel('y');
       axis([vxMin vxMax yMin yMax]);
       if ~optArgs.useSubplot
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' ' descriptor ' ' fields{2} ' Vx x=' num2str(xMin) '-'  ...
               num2str(xMax) ' y=' num2str(yMin) '-' num2str(yMax)]));
           outFile = strcat(fields{1}, '_', descriptor, '_vx_vs_y_x', num2str(xMin), ...
               '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
               '_', fields{2});
           print('-dpng', outFile);
       elseif~ strcmp(optArgs.subplotTitle, '')
           title(optArgs.subplotTitle);
       end
   end

   if optArgs.plotVyY
       % Vy vs. Y
       figureOrSubplot(optArgs.useSubplot, optArgs.subplotArgs);
       xEdges = linspace(vyMin, vyMax, numBins);
       yEdges = linspace(yMin, yMax, numBins);
       bins = hist2(p.vy, p.y, xEdges, yEdges);
       bins = conv2(bins, kernel, 'same');
       if optArgs.logScale
           bins = log(bins);
       end
       contourf(xEdges, yEdges, bins);
       if optArgs.showColorbar
           colorbar;
       end
       xlabel('vy');
       ylabel('y');
       axis([vyMin vyMax yMin yMax]);
       if ~optArgs.useSubplot
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' ' descriptor ' ' fields{2} ' Vy x=' num2str(xMin) '-'  ...
               num2str(xMax) ' y=' num2str(yMin) '-' num2str(yMax)]));
           outFile = strcat(fields{1}, '_', descriptor, '_vy_vs_y_x', num2str(xMin), ...
               '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
               '_', fields{2});
           print('-dpng', outFile);
       elseif~ strcmp(optArgs.subplotTitle, '')
           title(optArgs.subplotTitle);
       end
   end

   if optArgs.plotVxVy
       % Vx vs. Vy
       figureOrSubplot(optArgs.useSubplot, optArgs.subplotArgs);
       xEdges = linspace(vxMin, vxMax, numBins);
       yEdges = linspace(vyMin, vyMax, numBins);
       bins = hist2(p.vx, p.vy, xEdges, yEdges);
       bins = conv2(bins, kernel, 'same');
       if optArgs.logScale
           bins = log(bins);
       end
       contourf(xEdges, yEdges, bins);
       if optArgs.showColorbar
           colorbar;
       end
       xlabel('vx');
       ylabel('vy');
       axis([vxMin vxMax vyMin vyMax]);
       if ~optArgs.useSubplot
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' ' descriptor ' ' fields{2} ' Vy vs Vx x=' ...
               num2str(xMin) '-'  num2str(xMax) ' y=' num2str(yMin) '-' ...
               num2str(yMax)]));
           outFile = strcat(fields{1}, '_', descriptor, '_vy_vs_vx_x', num2str(xMin), ...
               '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
               '_', fields{2});
           print('-dpng', outFile);
       elseif~ strcmp(optArgs.subplotTitle, '')
           title(optArgs.subplotTitle);
       end
   end

   if optArgs.plotVxLine ~= Inf
       % Line plot vx ~= 0 vy
       if ~optArgs.getVxLineData
           figureOrSubplot(optArgs.useSubplot, optArgs.subplotArgs);
       end
       xEdges = linspace(vxMin, vxMax, numBins);
       yEdges = linspace(vyMin, vyMax, numBins);
       bins = hist2(p.vx, p.vy, xEdges, yEdges);
       bins = conv2(bins, kernel, 'same');
       if optArgs.logScale
           bins = log(bins);
       end
       xValues = yEdges;
       yValues = getAndAverageBins(bins, xEdges, optArgs.plotVxLine);
       if ~optArgs.getVxLineData
           plot(xValues, yValues);
           if optArgs.logScale
               ylabel('numParticles');
           else
               ylabel('numParticles');
           end
           xlabel('vy');
           if ~optArgs.useSubplot
               fields = strsplit(fName, '_');
               title(strcat([fields{1} ' ' descriptor ' ' fields{2} ' Vy vs density Vx=' ...
                   num2str(optArgs.plotVxLine) ' x=' ...
                   num2str(xMin) '-'  num2str(xMax) ' y=' num2str(yMin) '-' ...
                   num2str(yMax)]));
               outFile = strcat(fields{1}, '_', descriptor, '_vy_vs_density_x', num2str(xMin), ...
                   '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
                   '_', fields{2});
               print('-dpng', outFile);
           elseif~ strcmp(optArgs.subplotTitle, '')
               title(optArgs.subplotTitle);
           end
       else
           ret.vxLineData.x = xValues;
           ret.vxLineData.y = yValues;
       end
   end
   
   if optArgs.plotDerivative
       figureOrSubplot(optArgs.useSubplot, optArgs.subplotArgs);
       [xEdges, yEdges, bins] = ...
           vxDerivative(p, vxMin, vxMax, vyMin, vyMax, numBins, kernel);
       if optArgs.logScale
           bins = log(bins);
       end
       surf(xEdges(2:end-2), yEdges(1:end-1), bins(1:end-1, 1:end-1));
       if optArgs.showColorbar
           colorbar;
           if size(optArgs.derivativeRange, 1) ~= 0
               caxis(optArgs.derivativeRange);
           end
       end
       xlabel('vx');
       ylabel('vy');
       %axis([vxMin vxMax vyMin vyMax]);
       axis([xEdges(2) xEdges(end-2) yEdges(1) yEdges(end-1)]);
       if ~optArgs.useSubplot
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' ' descriptor ' ' fields{2} ' \partialF/\partialV_{x} x=' ...
               num2str(xMin) '-'  num2str(xMax) ' y=' num2str(yMin) '-' ...
               num2str(yMax)]));
           outFile = strcat(fields{1}, '_', descriptor, '_dVx_x', num2str(xMin), ...
               '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
               '_', fields{2});
           print('-dpng', outFile);
       elseif~ strcmp(optArgs.subplotTitle, '')
           title(optArgs.subplotTitle);
       end
   end
end

function ret = filterParticlesWithLogical(p, logical)
   ret.x = p.x(logical);
   ret.y = p.y(logical);
   ret.vx = p.vx(logical);
   ret.vy = p.vy(logical);
   ret.vz = p.vz(logical);
end

function ret = filterParticles(p, xMin, xMax, yMin, yMax)
   logArray = p.y >= yMin & ...
              p.y <= yMax & ...
              p.x >= xMin & ...
              p.x <= xMax;
   ret = filterParticlesWithLogical(p, logArray);
end

function ret = figureOrSubplot(useSubplot, subplotArgs)
   if ~useSubplot
       ret = figure;
   else
       ret = subplot(subplotArgs(1), subplotArgs(2), subplotArgs(3));
   end
end

function ret = getAndAverageBins(binList, xEdges, xValue)
   first = 1;
   second = 1;
   if xEdges(first) == xValue
       ret = binList(:,first);
       return
   end
   
   for i = 2:size(xEdges, 2)
       if xEdges(i) < xValue
           first = i;
       else
           second = i;
           break;
       end
   end
   ret = binList(:,first) + binList(:,second);
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

function ret = clearPlotFlags(options)
    options.plotVxY = false;
    options.plotVyY = false;
    options.plotVxVy = false;
    ret = options;
end

function ret = parseArgs(args)
    ret = struct( ...
        'enableCulling', true, ...
        'maxPoints', 64000, ...
        'plotHot', true, ...
        'plotCold', true, ...
        'numBins', 100, ...
        'logScale', false, ...
        'plotVxY', true, ...
        'plotVyY', true, ...
        'plotVxVy', true, ...
        'vxMax', Inf, ...
        'vxMin', Inf, ...
        'vyMax', Inf, ...
        'vyMin', Inf, ...
        'getVxLineData', false, ...
        'plotVxLine', Inf, ...
        'getGxLineData', false, ...
        'useSubplot', false, ...
        'subplotArgs', [], ...
        'subplotTitle', '', ...
        'showColorbar', true, ...
        'plotDerivative', false, ...
        'derivativeRange', [] ...
        );
    if ~isempty(args)
        i = 1;
        firstClear = false;
        while i <= length(args)
            if strcmp(args{i}, 'noCull')
                ret.enableCulling = false;
            elseif strcmp(args{i}, 'maxPoints')
                ret.maxPoints = args{i+1};
                i = i + 1;
            elseif strcmp(args{i}, 'hotOnly')
                ret.plotHot = true;
                ret.plotCold = false;
            elseif strcmp(args{i}, 'coldOnly')
                ret.plotHot = false;
                ret.plotCold = true;
            % The number of bins to use when constructing
            % the contour plots. The actual numbere of bins
            % used is numBins^2
            elseif strcmp(args{i}, 'numBins')
                ret.numBins = args{i+1};
                i = i + 1;
            elseif strcmp(args{i}, 'logScale')
                ret.logScale = true;
            elseif strcmp(args{i}, 'vxMax')
                ret.vxMax = args{i+1};
                i = i + 1;
            elseif strcmp(args{i}, 'vxMin')
                ret.vxMin = args{i+1};
                i = i + 1;
            elseif strcmp(args{i}, 'vyMax')
                ret.vyMax = args{i+1};
                i = i + 1;
            elseif strcmp(args{i}, 'vyMin')
                ret.vyMin = args{i+1};
                i = i + 1;
            elseif strcmp(args{i}, 'plotVxY')
                if ~firstClear
                    ret = clearPlotFlags(ret);
                    firstClear = true;
                end
                ret.plotVxY = true;
            elseif strcmp(args{i}, 'plotVyY')
                if ~firstClear
                    ret = clearPlotFlags(ret);
                    firstClear = true;
                end
                ret.plotVyY = true;
            elseif strcmp(args{i}, 'plotVxVy')
                if ~firstClear
                    ret = clearPlotFlags(ret);
                    firstClear = true;
                end
                ret.plotVxVy = true;
            elseif strcmp(args{i}, 'plotVxLine')
                if ~firstClear
                    ret = clearPlotFlags(ret);
                    firstClear = true;
                end
                ret.plotVxLine = args{i+1};
                i = i + 1;
            elseif strcmp(args{i}, 'plotDerivative')
                if ~firstClear
                    ret = clearPlotFlags(ret);
                    firstClear = true;
                end
                ret.plotDerivative = true;
            elseif strcmp(args{i}, 'derivativeRange')
                ret.derivativeRange = args{i+1};
                i = i + 1;
            elseif strcmp(args{i}, 'getVxLineData')
                ret.getVxLineData = true;
            elseif strcmp(args{i}, 'subplotArgs')
                ret.useSubplot = true;
                ret.subplotArgs = args{i+1};
                i = i + 1;
            elseif strcmp(args{i}, 'subplotTitle')
                ret.subplotTitle = args{i+1};
                i = i + 1;
            elseif strcmp(args{i}, 'noColorbar')
                ret.showColorbar = false;
            else
                error('Invalid option!');
            end
            i = i + 1;
        end
    end
end
