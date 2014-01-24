function ret = plotVxVy(fName, xMin, xMax, yMin, yMax)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end
   
   sizeOfFloat = 4;
   maxPlottableParticles = 4000;
   sliceHot = 0;
   sliceCold = 0;

   numParticles = fread(f, 1, 'int32');
   numHot = fread(f, 1, 'int32');
   numHotCulled = floor(numHot / sliceHot);
   numCold = fread(f, 1, 'int32');
   numColdCulled = floor(numCold / sliceCold);
   dataStart = ftell(f);
   
   hotP = zeros(4, numHot);
   nextSpace = 1;
   for i=1:numHot
      tmpX = fread(f, 1, 'float'); % Skip x
      tmpY = fread(f, 1, 'float'); % Read y
      tmpVx = fread(f, 1, 'float'); % Read vx
      tmpVy = fread(f, 1, 'float'); % Read vy
      tmpVz = fread(f, 1, 'float'); % Read vz
      if (xMin <= tmpX) && (tmpX < xMax) && (yMin <= tmpY) && (tmpY < yMax)
          hotP(1,i) = tmpY;
          hotP(2,i) = tmpVx;
          hotP(3,i) = tmpVy;
          hotP(4,i) = tmpVz;
          nextSpace = nextSpace + 1;
      end
   end
   numHotCulled = nextSpace - 1;
   hotP = hotP(:,1:numHotCulled);
   fseek(f, dataStart, 'bof');
   fseek(f, sizeOfFloat * 5 * numHot, 'cof');
   coldP = zeros(4, numCold);
   nextSpace = 1;
   for i=1:numCold
       tmpX = fread(f, 1, 'float'); % Skip x
       tmpY = fread(f, 1, 'float'); % Read y
       tmpVx = fread(f, 1, 'float'); % Read vx
       tmpVy = fread(f, 1, 'float'); % Read vy
       tmpVz = fread(f, 1, 'float'); % Read vz
       if (xMin <= tmpX) && (tmpX < xMax) && (yMin <= tmpY) && (tmpY < yMax)
           coldP(1,i) = tmpY;
           coldP(2,i) = tmpVx;
           coldP(3,i) = tmpVy;
           coldP(4,i) = tmpVz;
           nextSpace = nextSpace + 1;
       end
   end
   numColdCulled = nextSpace - 1;
   coldP = coldP(:,1:numColdCulled);
   fclose(f);
   
   if numHotCulled > maxPlottableParticles
       sliceHot = floor(numHotCulled / maxPlottableParticles);
       hotP = hotP(:,1:sliceHot:numHotCulled);
       numHotCulled = maxPlottableParticles;
       fprintf('Limiting number of hot particles plotted to %d\n', ...
           numColdCulled);
   end
   if numColdCulled > maxPlottableParticles
       sliceCold = floor(numColdCulled / maxPlottableParticles);
       coldP = coldP(:,1:sliceCold:numColdCulled);
       numColdCulled = maxPlottableParticles;
       fprintf('Limiting number of cold particles plotted to %d\n', ...
           numColdCulled);
   end
   
   stdDevMultiplier = 6;
   
   if numHotCulled ~= 0
       stdHotVx = std(hotP(2,:));
       meanHotVx = mean(hotP(2,:));
       vxMaxHot = meanHotVx + stdHotVx * stdDevMultiplier;
       vxMinHot = meanHotVx - stdHotVx * stdDevMultiplier;
       stdHotVy = std(hotP(3,:));
       meanHotVy = mean(hotP(3,:));
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
       stdColdVx = std(coldP(2,:));
       meanColdVx = mean(coldP(2,:));
       vxMaxCold = meanColdVx + stdColdVx * stdDevMultiplier;
       vxMinCold = meanColdVx - stdColdVx * stdDevMultiplier;
       stdColdVy = std(coldP(3,:));
       meanColdVy = mean(coldP(3,:));
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
   
   figure;
   scatter(coldP(2,:), coldP(1,:), 0.4)
   fields = strsplit(fName, '_');
   title(strcat([fields{1} ' cold ' fields{2} ' Vx x=' num2str(xMin) '-'  ...
       num2str(xMax) ' y=' num2str(yMin) '-' num2str(yMax)]));
   xlabel('vx');
   ylabel('y');
   axis([vxMinCold vxMaxCold 0 yMax]);
   fields = strsplit(fName, '_');
   outFile = strcat(fields{1}, '_cold_vx_vs_y_x', num2str(xMin), ...
       '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
       '_', fields{2});
   print('-dpng', outFile);
   
   figure;
   scatter(hotP(2,:), hotP(1,:), 0.4)
   fields = strsplit(fName, '_');
   title(strcat([fields{1} ' hot ' fields{2} ' Vx x=' num2str(xMin) '-'  ...
       num2str(xMax) ' y=' num2str(yMin) '-' num2str(yMax)]));
   xlabel('vx');
   ylabel('y');
   axis([vxMinHot vxMaxHot 0 max(hotP(1,:))]);
   fields = strsplit(fName, '_');
   outFile = strcat(fields{1}, '_hot_vx_vs_y_x', num2str(xMin), ...
       '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
       '_', fields{2});
   print('-dpng', outFile);
   
   figure;
   scatter(coldP(3,:), coldP(1,:), 0.4)
   fields = strsplit(fName, '_');
   title(strcat([fields{1} ' cold ' fields{2} ' Vy x=' num2str(xMin) '-'  ...
       num2str(xMax) ' y=' num2str(yMin) '-' num2str(yMax)]));
   title(strcat([fields{1} ' cold ' fields{2} ' Vy']));
   xlabel('vy');
   ylabel('y');
   axis([vyMinCold vyMaxCold 0 yMax]);
   fields = strsplit(fName, '_');
   outFile = strcat(fields{1}, '_cold_vy_vs_y_x', num2str(xMin), ...
       '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
       '_', fields{2});
   print('-dpng', outFile);
   
   figure;
   scatter(hotP(3,:), hotP(1,:), 0.4)
   fields = strsplit(fName, '_');
   title(strcat([fields{1} ' hot ' fields{2} ' Vy x=' num2str(xMin) '-'  ...
       num2str(xMax) ' y=' num2str(yMin) '-' num2str(yMax)]));
   xlabel('vy');
   ylabel('y');
   axis([vyMinHot vyMaxHot 0 max(hotP(1,:))]);
   fields = strsplit(fName, '_');
   outFile = strcat(fields{1}, '_hot_vy_vs_y_x', num2str(xMin), ...
       '-', num2str(xMax), '_y', num2str(yMin), '-', num2str(yMax), ...
       '_', fields{2});
   print('-dpng', outFile);
   
   figure;
   scatter(coldP(2,:), coldP(3,:), 0.4)
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
      
   figure;
   scatter(hotP(2,:), hotP(3,:), 0.4)
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
