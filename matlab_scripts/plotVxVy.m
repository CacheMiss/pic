function ret = plotVxVy(fName, sliceHot, sliceCold)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end
   
   sizeOfFloat = 4;

   numParticles = fread(f, 1, 'int32');
   numHot = fread(f, 1, 'int32');
   numHotCulled = floor(numHot / sliceHot);
   numCold = fread(f, 1, 'int32');
   numColdCulled = floor(numCold / sliceCold);
   dataStart = ftell(f);
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
   yMax = max(hotP(2,:));
   
   stdDevMultiplier = 6;
   
   stdHotVx = std(hotP(2,:));
   meanHotVx = mean(hotP(2,:));
   vxMaxHot = meanHotVx + stdHotVx * stdDevMultiplier;
   vxMinHot = meanHotVx - stdHotVx * stdDevMultiplier;
   
   stdColdVx = std(coldP(2,:));
   meanColdVx = mean(coldP(2,:));
   vxMaxCold = meanColdVx + stdColdVx * stdDevMultiplier;
   vxMinCold = meanColdVx - stdColdVx * stdDevMultiplier;
   
   stdHotVy = std(hotP(3,:));
   meanHotVy = mean(hotP(3,:));
   vyMaxHot = meanHotVy + stdHotVy * stdDevMultiplier;
   vyMinHot = meanHotVy - stdHotVy * stdDevMultiplier;
   
   stdColdVy = std(coldP(3,:));
   meanColdVy = mean(coldP(3,:));
   vyMaxCold = meanColdVy + stdColdVy * stdDevMultiplier;
   vyMinCold = meanColdVy - stdColdVy * stdDevMultiplier;
   
   figure;
   scatter(coldP(2,:), coldP(1,:), 0.4)
   fields = strsplit(fName, '_');
   title(strcat([fields{1} ' cold ' fields{2} ' Vx']));
   xlabel('vx');
   ylabel('y');
   axis([vxMinCold vxMaxCold 0 max(coldP(1,:))]);
   fields = strsplit(fName, '_');
   outFile = strcat(fields{1}, '_cold_vx_vs_y_', fields{2});
   print('-dpng', outFile);
   
   figure;
   scatter(hotP(2,:), hotP(1,:), 0.4)
   fields = strsplit(fName, '_');
   title(strcat([fields{1} ' hot ' fields{2} ' Vx']));
   xlabel('vx');
   ylabel('y');
   axis([vxMinHot vxMaxHot 0 max(hotP(1,:))]);
   fields = strsplit(fName, '_');
   outFile = strcat(fields{1}, '_hot_vx_vs_y_', fields{2});
   print('-dpng', outFile);
   
   figure;
   scatter(coldP(3,:), coldP(1,:), 0.4)
   fields = strsplit(fName, '_');
   title(strcat([fields{1} ' cold ' fields{2} ' Vy']));
   xlabel('vy');
   ylabel('y');
   axis([vyMinCold vyMaxCold 0 max(coldP(1,:))]);
   fields = strsplit(fName, '_');
   outFile = strcat(fields{1}, '_cold_vy_vs_y_', fields{2});
   print('-dpng', outFile);
   
   figure;
   scatter(hotP(3,:), hotP(1,:), 0.4)
   fields = strsplit(fName, '_');
   title(strcat([fields{1} ' hot ' fields{2} ' Vy']));
   xlabel('vy');
   ylabel('y');
   axis([vyMinHot vyMaxHot 0 max(hotP(1,:))]);
   fields = strsplit(fName, '_');
   outFile = strcat(fields{1}, '_hot_vy_vs_y_', fields{2});
   print('-dpng', outFile);
   

end
