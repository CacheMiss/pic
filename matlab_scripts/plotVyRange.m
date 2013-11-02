function ret = plotVyRange(fName, midpoint, width, sliceHot, sliceCold)

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
   hotP = zeros(2, numHotCulled);
   coldP = zeros(2, numColdCulled);
   nextSpace = 1;
   for i=1:numHotCulled
      tmpX = fread(f, 1, 'float'); % Skip x
      tmpY = fread(f, 1, 'float'); % Read y
      fseek(f, sizeOfFloat, 'cof'); % Skip vx
      tmpVy = fread(f, 1, 'float'); % Read vy
      if tmpX >= midpoint - width && tmpX <= midpoint + width
          hotP(2,nextSpace) = tmpY;
          hotP(1,nextSpace) = tmpVy;
          nextSpace = nextSpace + 1;
      end
      % Skip the  remaining velocity plus whatever else I need to reach
      % the next particle I care about
      skipBytes = sizeOfFloat * 1 + sizeOfFloat * 5 * (sliceHot-1);
      fseek(f, skipBytes, 'cof');
   end
   fseek(f, dataStart, 'bof');
   fseek(f, sizeOfFloat * 5 * numHot, 'cof');
   hotP = hotP(:, 1:nextSpace);
   nextSpace = 1;
   for i=1:numColdCulled
       tmpX = fread(f, 1, 'float'); % Skip x
       tmpY = fread(f, 1, 'float'); % Read y
       fseek(f, sizeOfFloat, 'cof'); % Skip vx
       tmpVy = fread(f, 1, 'float'); % Read vy
       if tmpX >= midpoint - width && tmpX <= midpoint + width
           coldP(2,nextSpace) = tmpY;
           coldP(1,nextSpace) = tmpVy;
           nextSpace = nextSpace + 1;
       end
       % Skip the  remaining velocity plus whatever else I need to reach
       % the next particle I care about
       skipBytes = sizeOfFloat * 1 + sizeOfFloat * 5 * (sliceCold-1);
       fseek(f, skipBytes, 'cof');
   end
   coldP = coldP(:, 1:nextSpace);
   fclose(f);
   yMax = max(hotP(2,:));
   
   stdDevMultiplier = 6;
   
   stdHotX = std(hotP(1,:));
   meanHotX = mean(hotP(1,:));
   xMaxHot = meanHotX + stdHotX * stdDevMultiplier;
   xMinHot = meanHotX - stdHotX * stdDevMultiplier;
   
   stdColdX = std(coldP(1,:));
   meanColdX = mean(coldP(1,:));
   xMaxCold = meanColdX + stdColdX * stdDevMultiplier;
   xMinCold = meanColdX - stdColdX * stdDevMultiplier;
   
   figure;
   scatter(coldP(1,:), coldP(2,:), 0.4)
   fields = strsplit(fName, '_');
   windowBegin = int2str(midpoint-width);
   windowEnd = int2str(midpoint+width);
   title(strcat([fields{1} ' cold ' fields{2} ...
       ' Vy (x=' windowBegin ' to x=' windowEnd ')']));
   xlabel('vy');
   ylabel('y');
   %axis([-2.5 2.2 0 yMax]);
   %axis([xMinCold xMaxCold 0 yMax]);
   axis([xMinCold xMaxCold 0 max(coldP(2,:))]);
   fields = strsplit(fName, '_');
   outFile = strcat(fields{1}, '_cold_vy_vs_y_x', ...
       windowBegin, '-', windowEnd, '_', fields{2});
   print('-dpng', outFile);
   
   figure;
   scatter(hotP(1,:), hotP(2,:), 0.4)
   fields = strsplit(fName, '_');
   windowBegin = int2str(midpoint-width);
   windowEnd = int2str(midpoint+width);
   title(strcat([fields{1} ' hot ' fields{2} ...
       ' Vy (x=' windowBegin ' to x=' windowEnd ')']));
   xlabel('vy');
   ylabel('y');
   %axis([-2.5 2.2 0 yMax]);
   %axis([xMinCold xMaxCold 0 yMax]);
   axis([xMinHot xMaxHot 0 max(hotP(2,:))]);
   fields = strsplit(fName, '_');
   outFile = strcat(fields{1}, '_hot_vy_vs_y_x', ...
       windowBegin, '-', windowEnd, '_', fields{2});
   print('-dpng', outFile);

end
