function ret = plotPart(fName, titleStr, y)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end

   numParticles = fread(f, 1, 'int32');
   numHot = fread(f, 1, 'int32');
   numCold = fread(f, 1, 'int32');
   sizeOfFloat = 4;
   hotP = [];
   coldP = [];
   scratchPad = zeros(2, max([numHot numCold]));
   
   numParticles = 0;
   for i=1:numHot
       fseek(f, sizeOfFloat, 'cof'); % Skip x
       thisY = fread(f, 1, 'float'); % Read y
       row = fread(f, 2, 'float'); % Read vx and vy
       fseek(f, sizeOfFloat, 'cof'); % Skip vz
       if floor(thisY) == y
           numParticles = numParticles + 1;
           scratchPad(:, numParticles) = row;
       end
   end
   hotP = scratchPad(:, 1:numParticles+1);
   numParticles = 0;
   for i=1:numCold
       fseek(f, sizeOfFloat, 'cof'); % Skip x
       thisY = fread(f, 1, 'float'); % Read y
       row = fread(f, 2, 'float'); % Read vx and vy
       fseek(f, sizeOfFloat, 'cof'); % Skip vz
       if floor(thisY) == y
           numParticles = numParticles + 1;
           scratchPad(:, numParticles) = row;
       end
   end
   coldP = scratchPad(:, 1:numParticles+1);
   clear scratchPad;
   fclose(f);
   
   stdDevMultiplier = 2;
   stdHotX = std(hotP(1,:));
   stdHotY = std(hotP(2,:));
   stdColdX = std(coldP(1,:));
   stdColdY = std(coldP(2,:));
   meanHotX = mean(hotP(1,:));
   meanHotY = mean(hotP(2,:));
   meanColdX = mean(coldP(1,:));
   meanColdY = mean(coldP(2,:));
   
   xMaxHot = meanHotX + stdHotX * stdDevMultiplier;
   yMaxHot = meanHotY + stdHotY * stdDevMultiplier;
   xMaxCold = meanColdX + stdColdX * stdDevMultiplier;
   yMaxCold = meanColdY + stdColdY * stdDevMultiplier;
   
   xMinHot = meanHotX - stdHotX * stdDevMultiplier;
   yMinHot = meanHotY - stdHotY * stdDevMultiplier;
   xMinCold = meanColdX - stdColdX * stdDevMultiplier;
   yMinCold = meanColdY - stdColdY * stdDevMultiplier;
   
   figure;
   scatter(hotP(1,:), hotP(2,:), 4)
   title(strcat([titleStr ' Hot y=' int2str(y)]));
   xlabel('vx');
   ylabel('vy');
   axis([xMinHot xMaxHot yMinHot yMaxHot]);

   figure;
   scatter(coldP(1,:), coldP(2,:), 4)
   title(strcat([titleStr ' Cold y=' int2str(y)]));
   xlabel('vx');
   ylabel('vy');
   axis([xMinCold xMaxCold yMinCold yMaxCold]);

end
