function ret = plotPart(fName, varargin)

   if isempty(varargin)
       % I still need initial values, even if I don't have arguments
       optArgs = parseArgs; 
   else
       optArgs = parseArgs(varargin);
   end
   
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
   stdDevMultiplier = 6;
   
   if ~ isempty(hotP) && optArgs.plotHot
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
       figure;
       scatter(hotP.x, hotP.y, energyHot/norm(energyHot)*dotScale, energyHot);
       colorbar;
       titleStr = strcat([fNameTokens{1}, ' hot ', fNameTokens{2}]);
       title(titleStr);
       xlabel('x');
       ylabel('y');
       axis([0 xMax 0 yMax]);
       outName = strcat(fNameTokens{1}, '_hot_', fNameTokens{2});
       print('-dpng', outName);
       
       % Plot Hot Particle Vx vs. Y
       figure;
       scatter(hotP.vx,hotP.y, 0.4)
       fields = strsplit(fName, '_');
       title(strcat([fields{1} ' hot ' fields{2} ' Vx']));
       xlabel('vx');
       ylabel('y');
       axis([vxMinHot vxMaxHot 0 max(hotP.y)]);
       fields = strsplit(fName, '_');
       outFile = strcat(fields{1}, '_hot_vx_vs_y_', fields{2});
       print('-dpng', outFile);

       % Plot Hot Particle Vy vs. Y
       figure;
       scatter(hotP.vy,hotP.y, 0.4)
       fields = strsplit(fName, '_');
       title(strcat([fields{1} ' hot ' fields{2} ' Vy']));
       xlabel('vy');
       ylabel('y');
       axis([vyMinHot vyMaxHot 0 max(hotP.y)]);
       fields = strsplit(fName, '_');
       outFile = strcat(fields{1}, '_hot_vy_vs_y_', fields{2});
       print('-dpng', outFile);

       % Plot Hot Particle Vy vs. Vx
       figure;
       scatter(hotP.vx, hotP.vy, 0.4)
       fields = strsplit(fName, '_');
       title(strcat([fields{1} ' hot ' fields{2} ' Vy vs Vx']));
       xlabel('vx');
       ylabel('vy');
       axis([vxMinHot vxMaxHot vyMinHot vyMaxHot]);
       fields = strsplit(fName, '_');
       outFile = strcat(fields{1}, '_hot_vy_vs_vx_', fields{2});
       print('-dpng', outFile);
   end
   
   if ~ isempty(coldP) && optArgs.plotCold
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
       
       % Plot Cold Particle Positions
       figure;
       scatter(coldP.x, coldP.y, energyCold/norm(energyCold)*dotScale, energyCold);
       colorbar;
       titleStr = strcat([fNameTokens{1}, ' cold ', fNameTokens{2}]);
       title(titleStr);
       xlabel('x');
       ylabel('y');
       axis([0 xMax 0 yMax]);
       outName = strcat(fNameTokens{1}, '_cold_', fNameTokens{2});
       print('-dpng', outName);
       
       % Plot Cold Particle Vx vs. Y
       figure;
       scatter(coldP.vx, coldP.y, 0.4)
       fields = strsplit(fName, '_');
       title(strcat([fields{1} ' cold ' fields{2} ' Vx']));
       xlabel('vx');
       ylabel('y');
       axis([vxMinCold vxMaxCold 0 max(coldP.y)]);
       fields = strsplit(fName, '_');
       outFile = strcat(fields{1}, '_cold_vx_vs_y_', fields{2});
       print('-dpng', outFile);

       % Plot Cold Particle Vy vs. Y
       figure;
       scatter(coldP.vy, coldP.y, 0.4)
       fields = strsplit(fName, '_');
       title(strcat([fields{1} ' cold ' fields{2} ' Vy']));
       xlabel('vy');
       ylabel('y');
       axis([vyMinCold vyMaxCold 0 max(coldP.y)]);
       fields = strsplit(fName, '_');
       outFile = strcat(fields{1}, '_cold_vy_vs_y_', fields{2});
       print('-dpng', outFile);

       % Plot Cold Particle Vy vs. Vx
       figure;
       scatter(coldP.vx, coldP.vy, 0.4)
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

function ret = parseArgs(args)
    ret = struct( ...
        'enableCulling', true, ...
        'maxParticles', 64000, ...
        'plotHot', true, ...
        'plotCold', true ...
        );
    if ~isempty(args)
        i = 1;
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
            else
                error('Invalid option!');
            end
            i = i + 1;
        end
    end
end
