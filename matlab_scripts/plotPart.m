function ret = plotPart(fName, varargin)

   optArgs = parseArgs(varargin);
   
   fNameTokens = strsplit(fName, '_');
   
   massRatio = 400;
   if strcmp(fNameTokens{1}, 'ele')
       massRatio = 1;
   end
   
   [hotP coldP] = loadParticles(fName, ...
       'maxParticles', optArgs.maxParticles, ...
       'enableCulling', optArgs.enableCulling, ...
       'loadHot', optArgs.plotHot, ...
       'loadCold', optArgs.plotCold ...
       );
   hotP = particleMix(hotP);
   coldP = particleMix(coldP);
   energyHot = massRatio * (hotP.vx.^2 + hotP.vy.^2 + hotP.vz.^2) / 2;
   energyCold = massRatio * (coldP.vx.^2 + coldP.vy.^2 + coldP.vz.^2) / 2;
   
   dotScale = 20;
   stdDevMultiplier = 3;
   numBins = optArgs.numBins;
   
   if size(hotP.x, 1) ~= 0 && optArgs.plotHot
       stdHotVx = std(hotP.vx);
       meanHotVx = mean(hotP.vx);
       vxMaxHot = meanHotVx + stdHotVx * stdDevMultiplier;
       vxMinHot = meanHotVx - stdHotVx * stdDevMultiplier;

       stdHotVy = std(hotP.vy);
       meanHotVy = mean(hotP.vy);
       vyMaxHot = meanHotVy + stdHotVy * stdDevMultiplier;
       vyMinHot = meanHotVy - stdHotVy * stdDevMultiplier;
   
       xMax = 2^nextpow2(max(hotP.x));
       % yMax = 2^nextpow2(max(hotP.vx));
       yMax = max(hotP.y);
       
       % Plot Hot Particle Positions
       if optArgs.plotLoc
           figure;
           scatter(hotP.x, hotP.y, energyHot/norm(energyHot)*dotScale, energyHot);
           colorbar;
           titleStr = strcat([fNameTokens{1}, ' hot ', fNameTokens{2}]);
           title(titleStr);
           xlabel('x');
           ylabel('y');
           caxis([0 mean(energyHot) + 3 * std(energyHot)]);
           axis([0 xMax 0 yMax]);
           outName = strcat(fNameTokens{1}, '_hot_', fNameTokens{2});
           print('-dpng', outName);
       end

       if optArgs.plotVxY
           % Plot Hot Particle Vx vs. Y
           figure;
           xEdges = linspace(vxMinHot, vxMaxHot, numBins);
           yEdges = linspace(0, max(hotP.y), numBins);
           bins = hist2(hotP.vx, hotP.y, xEdges, yEdges);
           if optArgs.logScale
               bins = log(bins);
           end
           contourf(xEdges, yEdges, bins);
           colorbar;
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' hot ' fields{2} ' Vx']));
           xlabel('vx');
           ylabel('y');
           axis([vxMinHot vxMaxHot 0 max(hotP.y)]);
           fields = strsplit(fName, '_');
           outFile = strcat(fields{1}, '_hot_vx_vs_y_', fields{2});
           print('-dpng', outFile);
       end

       if optArgs.plotVyY
           % Plot Hot Particle Vy vs. Y
           figure;
           xEdges = linspace(vyMinHot, vyMaxHot, numBins);
           yEdges = linspace(0, max(hotP.y), numBins);
           bins = hist2(hotP.vx, hotP.y, xEdges, yEdges);
           if optArgs.logScale
               bins = log(bins);
           end
           contourf(xEdges, yEdges, bins);
           colorbar;
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' hot ' fields{2} ' Vy']));
           xlabel('vy');
           ylabel('y');
           axis([vyMinHot vyMaxHot 0 max(hotP.y)]);
           fields = strsplit(fName, '_');
           outFile = strcat(fields{1}, '_hot_vy_vs_y_', fields{2});
           print('-dpng', outFile);
       end

       if optArgs.plotVxVy
           % Plot Hot Particle Vy vs. Vx
           figure;
           xEdges = linspace(vxMinHot, vxMaxHot, numBins);
           yEdges = linspace(vyMinHot, vyMaxHot, numBins);
           bins = hist2(hotP.vx, hotP.vy, xEdges, yEdges);
           if optArgs.logScale
               bins = log(bins);
           end
           contourf(xEdges, yEdges, bins);
           colorbar;
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' hot ' fields{2} ' Vy vs Vx']));
           xlabel('vx');
           ylabel('vy');
           axis([vxMinHot vxMaxHot vyMinHot vyMaxHot]);
           fields = strsplit(fName, '_');
           outFile = strcat(fields{1}, '_hot_vy_vs_vx_', fields{2});
           print('-dpng', outFile);
       end
   end
   
   if size(coldP.x, 1) ~= 0 && optArgs.plotCold
       stdColdVx = std(coldP.vx);
       meanColdVx = mean(coldP.vx);
       vxMaxCold = meanColdVx + stdColdVx * stdDevMultiplier;
       vxMinCold = meanColdVx - stdColdVx * stdDevMultiplier;

       stdColdVy = std(coldP.vy);
       meanColdVy = mean(coldP.vy);
       vyMaxCold = meanColdVy + stdColdVy * stdDevMultiplier;
       vyMinCold = meanColdVy - stdColdVy * stdDevMultiplier;

       xMax = 2^nextpow2(max(coldP.x));
       % yMax = max(yMax, 2^nextpow2(max(hotP.vx)));
       yMax = max(coldP.y);
       
       if optArgs.plotLoc
           % Plot Cold Particle Positions
           figure;
           scatter(coldP.x, coldP.y, energyCold/norm(energyCold)*dotScale, energyCold);
           colorbar;
           titleStr = strcat([fNameTokens{1}, ' cold ', fNameTokens{2}]);
           title(titleStr);
           xlabel('x');
           ylabel('y');
           caxis([0 mean(energyCold) + 3 * std(energyCold)]);
           axis([0 xMax 0 yMax]);
           outName = strcat(fNameTokens{1}, '_cold_', fNameTokens{2});
           print('-dpng', outName);
       end
       
       if optArgs.plotVxY
           % Plot Cold Particle Vx vs. Y
           figure;
           xEdges = linspace(vxMinCold, vxMaxCold, numBins);
           yEdges = linspace(0, max(coldP.y), numBins);
           bins = hist2(coldP.vx, coldP.y, xEdges, yEdges);
           if optArgs.logScale
               bins = log(bins);
           end
           contourf(xEdges, yEdges, bins);
           colorbar;
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' cold ' fields{2} ' Vx']));
           xlabel('vx');
           ylabel('y');
           axis([vxMinCold vxMaxCold 0 max(coldP.y)]);
           fields = strsplit(fName, '_');
           outFile = strcat(fields{1}, '_cold_vx_vs_y_', fields{2});
           print('-dpng', outFile);
       end

       if optArgs.plotVyY
           % Plot Cold Particle Vy vs. Y
           figure;
           xEdges = linspace(vyMinCold, vyMaxCold, numBins);
           yEdges = linspace(0, max(coldP.y), numBins);
           bins = hist2(coldP.vx, coldP.y, xEdges, yEdges);
           if optArgs.logScale
               bins = log(bins);
           end
           contourf(xEdges, yEdges, bins);
           colorbar;
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' cold ' fields{2} ' Vy']));
           xlabel('vy');
           ylabel('y');
           axis([vyMinCold vyMaxCold 0 max(coldP.y)]);
           fields = strsplit(fName, '_');
           outFile = strcat(fields{1}, '_cold_vy_vs_y_', fields{2});
           print('-dpng', outFile);
       end

       if optArgs.plotVxVy
           % Plot Cold Particle Vy vs. Vx
           figure;
           xEdges = linspace(vxMinCold, vxMaxCold, numBins);
           yEdges = linspace(vyMinCold, vyMaxCold, numBins);
           bins = hist2(coldP.vx, coldP.vy, xEdges, yEdges);
           if optArgs.logScale
               bins = log(bins);
           end
           contourf(xEdges, yEdges, bins);
           colorbar;
           fields = strsplit(fName, '_');
           title(strcat([fields{1} ' cold ' fields{2} ' Vy vs Vx']));
           xlabel('vx');
           ylabel('vy');
           axis([vxMinCold vxMaxCold vyMinCold vyMaxCold]);
           fields = strsplit(fName, '_');
           outFile = strcat(fields{1}, '_cold_vy_vs_vx_', fields{2});
           print('-dpng', outFile);
       end
   end

end

function ret = clearPlotFlags(options)
    options.plotLoc = false;
    options.plotVxY = false;
    options.plotVyY = false;
    options.plotVxVy = false;
    ret = options;
end

function ret = parseArgs(args)
    ret = struct( ...
        'enableCulling', true, ...
        'maxParticles', 64000, ...
        'plotHot', true, ...
        'plotCold', true, ...
        'plotLoc', true, ...
        'plotVxY', true, ...
        'plotVyY', true, ...
        'plotVxVy', true, ...
        'numBins', 100, ...
        'logScale', true ...
        );
    if ~isempty(args)
        i = 1;
        firstClear = false;
        while i <= length(args)
            if strcmp(args{i}, 'noCull')
                ret.enableCulling = false;
            elseif strcmp(args{i}, 'maxParticles')
                ret.maxParticles = args{i+1};
                i = i + 1;
            elseif strcmp(args{i}, 'hotOnly')
                ret.plotHot = true;
                ret.plotCold = false;
            elseif strcmp(args{i}, 'coldOnly')
                ret.plotHot = false;
                ret.plotCold = true;
            elseif strcmp(args{i}, 'plotLoc')
                if ~firstClear
                    ret = clearPlotFlags(ret);
                    firstClear = true;
                end
                ret.plotLoc = true;
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
            % The number of bins to use when constructing
            % the contour plots. The actual numbere of bins
            % used is numBins^2
            elseif strcmp(args{i}, 'numBins')
                ret.numBins = args{i+1};
                i = i + 1;
            elseif strcmp(args{i}, 'logScale')
                ret.logScale = true;
            elseif strcmp(args{i}, 'noLogScale')
                ret.logScale = false;
            else
                error('Invalid option!');
            end
            i = i + 1;
        end
    end
end
