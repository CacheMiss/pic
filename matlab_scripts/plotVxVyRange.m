function ret = plotVxVy(fName, xMin, xMax, yMin, yMax, varargin)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end
   
   if isempty(varargin)
       % I still need initial values, even if I don't have arguments
       optArgs = parseArgs; 
   else
       optArgs = parseArgs(varargin);
   end

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
   fclose(f);
   
   if optArgs.enableCulling
       if numHotCulled > maxPlottableParticles
           sliceHot = floor(numHotCulled / maxPlottableParticles);
           hotP = particleCull(hotP, sliceHot);
           numHotCulled = maxPlottableParticles;
           fprintf('Limiting number of hot particles plotted to %d\n', ...
               numHotCulled);
       end
       if numColdCulled > maxPlottableParticles
           sliceCold = floor(numColdCulled / maxPlottableParticles);
           coldP = particleCull(coldP, sliceCold);
           numColdCulled = maxPlottableParticles;
           fprintf('Limiting number of cold particles plotted to %d\n', ...
               numColdCulled);
       end
   end
   
   stdDevMultiplier = 6;
   
   if numHotCulled ~= 0
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
   
   if numColdCulled ~= 0
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
   
   % Randomize these vectors for faster plotting
   hotP = particleMix(hotP);
   coldP = particleMix(coldP);
   
   % Cold Vx vs. Vy
   figure;
   scatter(coldP.vx, coldP.y, 0.4)
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
   
   % Hot Vx vs. Vy
   figure;
   scatter(hotP.vx, hotP.y, 0.4)
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
   
   % Cold Vy vs. Y
   figure;
   scatter(coldP.vy, coldP.y, 0.4)
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
   
   % Hot Vy vs. Y
   figure;
   scatter(hotP.vy, hotP.y, 0.4)
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
   
   % Cold Vx vs. Vy
   figure;
   scatter(coldP.vx, coldP.vy, 0.4)
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
   
   % Hot Vx vs. Vy
   figure;
   scatter(hotP.vx, hotP.vy, 0.4)
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

function ret = parseArgs(varargin)
    ret = struct( ...
        'enableCulling', true, ...
        'maxPoints', 4000 ...
        );
    if ~isempty(varargin)
        for i = 1:length(varargin)
            if strcmp(varargin{i}(1), 'noCull')
                ret.enableCulling = false;
            elseif strcmp(varargin{i}(1), 'maxPoints')
                ret.maxPoints = varargin{i}{2};
            else
                error('Invalid option!');
            end
        end
    end
end