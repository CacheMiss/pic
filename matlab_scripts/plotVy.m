function ret = plotPart(fName, sliceSize)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end

   numParticles = fread(f, 1, 'int32');
   numHot = fread(f, 1, 'int32');
   numCold = fread(f, 1, 'int32');
   finalHot = floor(numHot / sliceSize);
   finalCold = floor(numCold / sliceSize);
   hotP = zeros(2, finalHot);
   coldP = zeros(2, finalCold);
   for i=1:finalHot
       sizeOfFloat = 4;
       fseek(f, sizeOfFloat, 'cof'); % Skip x
       hotP(2,i) = fread(f, 1, 'float'); % Read y
       fseek(f, sizeOfFloat, 'cof'); % Skip vx
       hotP(1,i) = fread(f, 1, 'float'); % Read vy
       % Skip the  remaining velocity plus whatever else I need to reach
       % the next particle I care about
       skipBytes = sizeOfFloat * 1 + sizeOfFloat * 5 * (sliceSize-1);
       fseek(f, skipBytes, 'cof');
   end
   for i=1:finalCold
       sizeOfFloat = 4;
       fseek(f, sizeOfFloat, 'cof'); % Skip x
       coldP(2,i) = fread(f, 1, 'float'); % Read y
       fseek(f, sizeOfFloat, 'cof'); % Skip vx
       coldP(1,i) = fread(f, 1, 'float'); % Read vy
       % Skip the  remaining velocity plus whatever else I need to reach
       % the next particle I care about
       skipBytes = sizeOfFloat * 1 + sizeOfFloat * 5 * (sliceSize-1);
       fseek(f, skipBytes, 'cof');
   end
   fclose(f);
   yMax = 2^nextpow2(max(hotP(2,:)));
   yMax = max(yMax, 2^nextpow2(max(coldP(2,:))));
   
   stdDevMultiplier = 2;
   stdHotX = std(hotP(1,:));
   stdColdX = std(coldP(1,:));
   meanHotX = mean(hotP(1,:));
   meanColdX = mean(coldP(1,:));
   
   xMaxHot = meanHotX + stdHotX * stdDevMultiplier;
   xMaxCold = meanColdX + stdColdX * stdDevMultiplier;
   
   xMinHot = meanHotX - stdHotX * stdDevMultiplier;
   xMinCold = meanColdX - stdColdX * stdDevMultiplier;
   
   figure;
   scatter(hotP(1,:), hotP(2,:), 0.4)
   title('Hot');
   xlabel('vy');
   ylabel('y');
   %axis([0 max(hotP(1,:)) 0 yMax]);
   axis([xMinHot xMaxHot 0 yMax]);

   figure;
   scatter(coldP(1,:), coldP(2,:), 0.4)
   title('Cold');
   xlabel('vy');
   ylabel('v');
   axis([xMinCold xMaxHot 0 yMax]);

end
