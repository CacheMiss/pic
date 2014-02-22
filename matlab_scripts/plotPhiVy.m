function ret = plotPhiVy(partFile, sliceHot, sliceCold, phiFile, phiMidpoint)

   f = fopen(partFile, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end
   
   sizeOfFloat = 4;
   maxPlottableParticles = 4000;

   numParticles = fread(f, 1, 'int32');
   numHot = fread(f, 1, 'int32');
   numHotCulled = floor(numHot / sliceHot);
   numCold = fread(f, 1, 'int32');
   numColdCulled = floor(numCold / sliceCold);
   dataStart = ftell(f);
   
   if numHotCulled > maxPlottableParticles
       numHotCulled = maxPlottableParticles;
       sliceHot = floor(numHot / maxPlottableParticles);
       fprintf('Limiting number of hot particles plotted to %d\n', ...
           numHotCulled);
   end
   if numColdCulled > maxPlottableParticles
       numColdCulled = maxPlottableParticles;
       sliceCold = floor(numCold / maxPlottableParticles);
       fprintf('Limiting number of cold particles plotted to %d\n', ...
           numColdCulled);
   end
   
   hotP = zeros(3, numHotCulled);
   coldP = zeros(3, numColdCulled);
   nextSpace = 1;
   for i=1:numHotCulled
      %tmpX = fread(f, 1, 'float'); % Skip x
      fseek(f, sizeOfFloat, 'cof'); % Skip x
      tmpY = fread(f, 1, 'float'); % Read y
      tmpVx = fread(f, 1, 'float'); % Read vx
      tmpVy = fread(f, 1, 'float'); % Read vy
      hotP(1,i) = tmpY;
      hotP(2,i) = tmpVx;
      hotP(3,i) = tmpVy;
      % Skip the  vz plus whatever else I need to reach
      % the next particle I care about
      skipBytes = sizeOfFloat + sizeOfFloat * 5 * (sliceHot-1);
      fseek(f, skipBytes, 'cof');
   end
   fseek(f, dataStart, 'bof');
   fseek(f, sizeOfFloat * 5 * numHot, 'cof');
   nextSpace = 1;
   for i=1:numColdCulled
       %tmpX = fread(f, 1, 'float'); % Skip x
       fseek(f, sizeOfFloat, 'cof'); % Skip x
       tmpY = fread(f, 1, 'float'); % Read y
       tmpVx = fread(f, 1, 'float'); % Read vx
       tmpVy = fread(f, 1, 'float'); % Read vy
       coldP(1,i) = tmpY;
       coldP(2,i) = tmpVx;
       coldP(3,i) = tmpVy;
       % Skip vz plus whatever else I need to reach
       % the next particle I care about
       skipBytes = sizeOfFloat + sizeOfFloat * 5 * (sliceCold-1);
       fseek(f, skipBytes, 'cof');
   end
   fclose(f);
   
   f = fopen(phiFile, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end
   
   numRows = fread(f, 1, 'int32');
   numColumns = fread(f, 1, 'int32');
   columnOrder = fread(f, 1, 'int32');
   phi = fread(f, [numRows,numColumns], 'float');
   phi = phi(:,phiMidpoint);
   phiMax = max(phi);
   phiMin = min(phi);
   
   yMax = max(hotP(2,:));
   
   stdDevMultiplier = 6;
   
   if numHotCulled > 0
       stdHotVx = std(hotP(2,:));
       meanHotVx = mean(hotP(2,:));
       vxMaxHot = meanHotVx + stdHotVx * stdDevMultiplier;
       vxMinHot = meanHotVx - stdHotVx * stdDevMultiplier;
       
       stdHotVy = std(hotP(3,:));
       meanHotVy = mean(hotP(3,:));
       vyMaxHot = meanHotVy + stdHotVy * stdDevMultiplier;
       vyMinHot = meanHotVy - stdHotVy * stdDevMultiplier;
   end
   
   if numColdCulled > 0
       stdColdVx = std(coldP(2,:));
       meanColdVx = mean(coldP(2,:));
       vxMaxCold = meanColdVx + stdColdVx * stdDevMultiplier;
       vxMinCold = meanColdVx - stdColdVx * stdDevMultiplier;
       
       stdColdVy = std(coldP(3,:));
       meanColdVy = mean(coldP(3,:));
       vyMaxCold = meanColdVy + stdColdVy * stdDevMultiplier;
       vyMinCold = meanColdVy - stdColdVy * stdDevMultiplier;
   end
  
   if numColdCulled > 0
       figure;
       scatter(coldP(3,:), coldP(1,:), 0.4)
       hold on; % preserve this plot
       phiScale = (vyMaxCold - vyMinCold) / (phiMax - phiMin);
       phiScaled = phi * phiScale;
       plot(phiScaled, 0:numRows-1);
       fields = strsplit(partFile, '_');
       title(strcat([fields{1} ' cold ' fields{2} ' Vy']));
       xlabel('vy');
       ylabel('y');
       axis([vyMinCold vyMaxCold 0 numRows-1]);
       outFile = strcat(fields{1}, '_cold_vy_vs_phi_', fields{2});
       print('-dpng', outFile);
   end
   
   if numHotCulled > 0
       figure;
       scatter(hotP(3,:), hotP(1,:), 0.4)
       hold on; % preserve this plot
       phiScale = (vyMaxHot - vyMinHot) / (phiMax - phiMin);
       phiScaled = phi * phiScale;
       plot(phiScaled, 0:numRows-1);
       fields = strsplit(partFile, '_');
       title(strcat([fields{1} ' hot ' fields{2} ' Vy']));
       xlabel('vy');
       ylabel('y');
       axis([vyMinHot vyMaxHot 0 numRows-1]);
       outFile = strcat(fields{1}, '_hot_vy_vs_phi_', fields{2});
       print('-dpng', outFile);
   end
end
