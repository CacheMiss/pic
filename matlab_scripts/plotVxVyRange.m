function ret = plotVxVyRange(fName, xMin, xMax, yMin, yMax, varargin)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end
   
   optArgs = parseArgs(varargin);

   sizeOfFloat = 4;
   maxPlottableParticles = optArgs.maxPoints;
   sliceHot = 0;
   sliceCold = 0;

   numParticles = fread(f, 1, 'int32');
   numHot = fread(f, 1, 'int32');
   numHotCulled = floor(numHot / sliceHot);
   numCold = fread(f, 1, 'int32');
   numColdCulled = floor(numCold / sliceCold);
   dataStart = ftell(f);
   
   if optArgs.plotHot
       hotP = particleNull(numHot);
       nextSpace = 1;
       for i=1:numHot
          tmpX = fread(f, 1, 'float'); % Read x
          tmpY = fread(f, 1, 'float'); % Read y
          tmpVx = fread(f, 1, 'float'); % Read vx
          tmpVy = fread(f, 1, 'float'); % Read vy
          tmpVz = fread(f, 1, 'float'); % Read vz
          if (xMin <= tmpX) && (tmpX < xMax) && (yMin <= tmpY) && (tmpY < yMax)
              hotP.x(nextSpace) = tmpX;
              hotP.y(nextSpace) = tmpY;
              hotP.vx(nextSpace) = tmpVx;
              hotP.vy(nextSpace) = tmpVy;
              hotP.vz(nextSpace) = tmpVz;
              nextSpace = nextSpace + 1;
          end
       end
       numHotCulled = nextSpace - 1;
       hotP = particleTrim(hotP, numHotCulled);
   else
       hotP = particleNull(0);
   end
   
   if optArgs.plotCold
       fseek(f, dataStart, 'bof');
       fseek(f, sizeOfFloat * 5 * numHot, 'cof');
       coldP = particleNull(numCold);
       nextSpace = 1;
       for i=1:numCold
           tmpX = fread(f, 1, 'float'); % Read x
           tmpY = fread(f, 1, 'float'); % Read y
           tmpVx = fread(f, 1, 'float'); % Read vx
           tmpVy = fread(f, 1, 'float'); % Read vy
           tmpVz = fread(f, 1, 'float'); % Read vz
           if (xMin <= tmpX) && (tmpX < xMax) && (yMin <= tmpY) && (tmpY < yMax)
               coldP.x(nextSpace) = tmpX;
               coldP.y(nextSpace) = tmpY;
               coldP.vx(nextSpace) = tmpVx;
               coldP.vy(nextSpace) = tmpVy;
               coldP.vz(nextSpace) = tmpVz;
               nextSpace = nextSpace + 1;
           end
       end
       numColdCulled = nextSpace - 1;
       coldP = particleTrim(coldP, numColdCulled);
   else
       coldP = particleNull(0);
   end
   
   fclose(f);
   
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
   
   stdDevMultiplier = 3;
   
   if optArgs.plotHot && numHotCulled ~= 0
       stdHotVx = std(hotP.vx);
       meanHotVx = mean(hotP.vx);
       vxMaxHot = meanHotVx + stdHotVx * stdDevMultiplier;
       vxMinHot = meanHotVx - stdHotVx * stdDevMultiplier;
       stdHotVy = std(hotP.vy);
       meanHotVy = mean(hotP.vy);
       vyMaxHot = meanHotVy + stdHotVy * stdDevMultiplier;
       vyMinHot = meanHotVy - stdHotVy * stdDevMultiplier;
   else
       stdHotVx = 1;
       meanHotVx = 1;
       vxMaxHot = 1;
       vxMinHot = 0;
       stdHotVy = 1;
       meanHotVy = 1;
       vyMaxHot = 1;
       vyMinHot = 0;
   end
   
   if optArgs.plotCold && numColdCulled ~= 0
       stdColdVx = std(coldP.vx);
       meanColdVx = mean(coldP.vx);
       vxMaxCold = meanColdVx + stdColdVx * stdDevMultiplier;
       vxMinCold = meanColdVx - stdColdVx * stdDevMultiplier;
       stdColdVy = std(coldP.vy);
       meanColdVy = mean(coldP.vy);
       vyMaxCold = meanColdVy + stdColdVy * stdDevMultiplier;
       vyMinCold = meanColdVy - stdColdVy * stdDevMultiplier;
   else
       stdColdVx = 1;
       meanColdVx = 1;
       vxMaxCold = 1;
       vxMinCold = 0;
       stdColdVy = 1;
       meanColdVy = 1;
       vyMaxCold = 1;
       vyMinCold = 0;
   end
   
   if optArgs.vxMax ~= Inf
       vxMaxHot = optArgs.vxMax;
       vxMaxCold = optArgs.vxMax;
   end
   if optArgs.vxMin ~= Inf
       vxMinHot = optArgs.vxMin;
       vxMinCold = optArgs.vxMin;
   end
   if optArgs.vyMax ~= Inf
       vyMaxHot = optArgs.vyMax;
       vyMaxCold = optArgs.vyMax;
   end
   if optArgs.vyMin ~= Inf
       vyMinHot = optArgs.vyMin;
       vyMinCold = optArgs.vyMin;
   end
   
   % Randomize these vectors for faster plotting
   if optArgs.plotHot
       hotP = particleMix(hotP);
   end
   if optArgs.plotCold
       coldP = particleMix(coldP);
   end
   
   numBins = optArgs.numBins;
   kernelSize = 3;
   kernel = ones(kernelSize,kernelSize) / kernelSize^2; % NxN mean kernel
   
   if optArgs.plotHot && size(hotP.x, 1) ~= 0
       if optArgs.plotVxY
           % Hot Vx vs. Y
           figure;
           %scatter(hotP.vx, hotP.y, 0.4);
           xEdges = linspace(vxMinHot, vxMaxHot, numBins);
           yEdges = linspace(yMin, yMax, numBins);
           bins = hist2(hotP.vx, hotP.y, xEdges, yEdges);
           bins = conv2(bins, kernel, 'same');
           if optArgs.logScale
               bins = log(bins);
           end
           contourf(xEdges, yEdges, bins);
           colorbar;
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' hot ' fields{2} ' Vx x=' num2str(xMin) '-'  ...
               num2str(xMax) ' y=' num2str(yMin) '-' num2str(yMax)]));
           xlabel('vx');
           ylabel('y');
           axis([vxMinHot vxMaxHot yMin yMax]);
           fields = strsplit(fName, '_');
           outFile = strcat(fields{1}, '_hot_vx_vs_y_x', num2str(xMin), ...
               '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
               '_', fields{2});
           print('-dpng', outFile);
       end
       
       if optArgs.plotVyY
           % Hot Vy vs. Y
           figure;
           %scatter(hotP.vy, hotP.y, 0.4);
           xEdges = linspace(vyMinHot, vyMaxHot, numBins);
           yEdges = linspace(yMin, yMax, numBins);
           bins = hist2(hotP.vy, hotP.y, xEdges, yEdges);
           bins = conv2(bins, kernel, 'same');
           if optArgs.logScale
               bins = log(bins);
           end
           contourf(xEdges, yEdges, bins);
           colorbar;
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' hot ' fields{2} ' Vy x=' num2str(xMin) '-'  ...
               num2str(xMax) ' y=' num2str(yMin) '-' num2str(yMax)]));
           xlabel('vy');
           ylabel('y');
           axis([vyMinHot vyMaxHot yMin yMax]);
           fields = strsplit(fName, '_');
           outFile = strcat(fields{1}, '_hot_vy_vs_y_x', num2str(xMin), ...
               '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
               '_', fields{2});
           print('-dpng', outFile);
       end
       
       if optArgs.plotVxVy
           % Hot Vx vs. Vy
           %figure;
           %scatter(hotP.vx, hotP.vy, 0.4);
           figure;
           xEdges = linspace(vxMinHot, vxMaxHot, numBins);
           yEdges = linspace(vyMinHot, vyMaxHot, numBins);
           bins = hist2(hotP.vx, hotP.vy, xEdges, yEdges);
           bins = conv2(bins, kernel, 'same');
           if optArgs.logScale
               bins = log(bins);
           end
           contourf(xEdges, yEdges, bins);
           colorbar;
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' hot ' fields{2} ' Vy vs Vx x=' ...
               num2str(xMin) '-'  num2str(xMax) ' y=' num2str(yMin) '-' ...
               num2str(yMax)]));
           xlabel('vx');
           ylabel('vy');
           axis([vxMinHot vxMaxHot vyMinHot vyMaxHot]);
           fields = strsplit(fName, '_');
           outFile = strcat(fields{1}, '_hot_vy_vs_vx_x', num2str(xMin), ...
               '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
               '_', fields{2});
           print('-dpng', outFile);
       end
       
       if optArgs.plotVxLine ~= Inf
           figure;
           xEdges = linspace(vxMinHot, vxMaxHot, numBins);
           yEdges = linspace(vyMinHot, vyMaxHot, numBins);
           bins = hist2(hotP.vx, hotP.vy, xEdges, yEdges);
           bins = conv2(bins, kernel, 'same');
           if optArgs.logScale
               bins = log(bins);
           end
           xValues = yEdges;
           yValues = getAndAverageBins(bins, xEdges, optArgs.plotVxLine);
           plot(xValues, yValues);
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' hot ' fields{2} ' Vy vs density Vx=' ...
               num2str(optArgs.plotVxLine) ' x=' ...
               num2str(xMin) '-'  num2str(xMax) ' y=' num2str(yMin) '-' ...
               num2str(yMax)]));
           if optArgs.logScale
               ylabel('numParticles');
           else
               ylabel('numParticles');
           end
           xlabel('vy');
           fields = strsplit(fName, '_');
           outFile = strcat(fields{1}, '_hot_vy_vs_density_x', num2str(xMin), ...
               '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
               '_', fields{2});
           print('-dpng', outFile);
       end
   end
   
   if optArgs.plotCold && size(coldP.x, 1) ~= 0
       if optArgs.plotVxY
           % Cold Vx vs. Y
           figure;
           %scatter(coldP.vx, coldP.y, 0.4);
           xEdges = linspace(vxMinCold, vxMaxCold, numBins);
           yEdges = linspace(yMin, yMax, numBins);
           bins = hist2(coldP.vx, coldP.y, xEdges, yEdges);
           bins = conv2(bins, kernel, 'same');
           if optArgs.logScale
               bins = log(bins);
           end
           contourf(xEdges, yEdges, bins);
           colorbar;
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' cold ' fields{2} ' Vx x=' num2str(xMin) '-'  ...
               num2str(xMax) ' y=' num2str(yMin) '-' num2str(yMax)]));
           xlabel('vx');
           ylabel('y');
           axis([vxMinCold vxMaxCold yMin yMax]);
           fields = strsplit(fName, '_');
           outFile = strcat(fields{1}, '_cold_vx_vs_y_x', num2str(xMin), ...
               '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
               '_', fields{2});
           print('-dpng', outFile);
       end
       
       if optArgs.plotVyY
           % Cold Vy vs. Y
           figure;
           %scatter(coldP.vy, coldP.y, 0.4);
           xEdges = linspace(vyMinCold, vyMaxCold, numBins);
           yEdges = linspace(yMin, yMax, numBins);
           bins = hist2(coldP.vy, coldP.y, xEdges, yEdges);
           bins = conv2(bins, kernel, 'same');
           if optArgs.logScale
               bins = log(bins);
           end
           contourf(xEdges, yEdges, bins);
           colorbar;
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' cold ' fields{2} ' Vy x=' num2str(xMin) '-'  ...
               num2str(xMax) ' y=' num2str(yMin) '-' num2str(yMax)]));
           title(strcat([fields{1} ' cold ' fields{2} ' Vy']));
           xlabel('vy');
           ylabel('y');
           axis([vyMinCold vyMaxCold yMin yMax]);
           fields = strsplit(fName, '_');
           outFile = strcat(fields{1}, '_cold_vy_vs_y_x', num2str(xMin), ...
               '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
               '_', fields{2});
           print('-dpng', outFile);
       end
     
       if optArgs.plotVxVy
           % Cold Vx vs. Vy
           figure;
           %scatter(coldP.vx, coldP.vy, 0.4);
           xEdges = linspace(vxMinCold, vxMaxCold, numBins);
           yEdges = linspace(vyMinCold, vyMaxCold, numBins);
           bins = hist2(coldP.vx, coldP.vy, xEdges, yEdges);
           bins = conv2(bins, kernel, 'same');
           if optArgs.logScale
               bins = log(bins);
           end
           contourf(xEdges, yEdges, bins);
           colorbar;
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' cold ' fields{2} ' Vy vs Vx x=' ...
               num2str(xMin) '-'  num2str(xMax) ' y=' num2str(yMin) '-' ...
               num2str(yMax)]));
           title(strcat([fields{1} ' cold ' fields{2} ' Vy']));
           xlabel('vx');
           ylabel('vy');
           axis([vxMinCold vxMaxCold vyMinCold vyMaxCold]);
           fields = strsplit(fName, '_');
           outFile = strcat(fields{1}, '_cold_vy_vs_vx_x', num2str(xMin), ...
               '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
               '_', fields{2});
           print('-dpng', outFile);
       end
       
       if optArgs.plotVxLine ~= Inf
           figure;
           xEdges = linspace(vxMinCold, vxMaxCold, numBins);
           yEdges = linspace(vyMinCold, vyMaxCold, numBins);
           bins = hist2(coldP.vx, coldP.vy, xEdges, yEdges);
           bins = conv2(bins, kernel, 'same');
           if optArgs.logScale
               bins = log(bins);
           end
           xValues = yEdges;
           yValues = getAndAverageBins(bins, xEdges, optArgs.plotVxLine);
           plot(xValues, yValues);
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' cold ' fields{2} ' Vy vs density Vx=' ...
               num2str(optArgs.plotVxLine) ' x=' ...
               num2str(xMin) '-'  num2str(xMax) ' y=' num2str(yMin) '-' ...
               num2str(yMax)]));
           if optArgs.logScale
               ylabel('numParticles');
           else
               ylabel('numParticles');
           end
           xlabel('vy');
           fields = strsplit(fName, '_');
           outFile = strcat(fields{1}, '_cold_vy_vs_density_x', num2str(xMin), ...
               '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
               '_', fields{2});
           print('-dpng', outFile);
       end
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

function ret = clearPlotFlags(options)
    options.plotVxY = false;
    options.plotVyY = false;
    options.plotVxVy = false;
    ret = options;
end

function ret = parseArgs(args)
    ret = struct( ...
        'enableCulling', true, ...
        'maxPoints', 4000, ...
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
        'plotVxLine', Inf ...
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
            else
                error('Invalid option!');
            end
            i = i + 1;
        end
    end
end