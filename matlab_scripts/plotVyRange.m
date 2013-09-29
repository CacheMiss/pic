function ret = plotVyRange(fName, titleStr, midpoint, width)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end

   numParticles = fread(f, 1, 'int32');
   numHot = fread(f, 1, 'int32');
   numCold = fread(f, 1, 'int32');
   hotP = zeros(2, numHot);
   nextSpace = 1;
   for i=1:numHot
       sizeOfFloat = 4;
       tmpX = fread(f, 1, 'float'); % Skip x
       tmpY = fread(f, 1, 'float'); % Read y
       fseek(f, sizeOfFloat, 'cof'); % Skip vx
       tmpVy = fread(f, 1, 'float'); % Read vy
       if tmpX >= midpoint - width && tmpX <= midpoint + width
           hotP(2,nextSpace) = tmpY;
           hotP(1,nextSpace) = tmpVy;
           nextSpace = nextSpace + 1;
       end
       % Skip Vz
       fseek(f, sizeOfFloat, 'cof');
   end
   fclose(f);
   hotP = hotP(:, 1:nextSpace);
   yMax = max(hotP(2,:));
   
   stdDevMultiplier = 2;
   stdHotX = std(hotP(1,:));
   meanHotX = mean(hotP(1,:));
   
   xMaxHot = meanHotX + stdHotX * stdDevMultiplier;
   
   xMinHot = meanHotX - stdHotX * stdDevMultiplier;
   
   figure;
   scatter(hotP(1,:), hotP(2,:), 0.4)
   title(strcat([titleStr ' Hot']));
   xlabel('vy');
   ylabel('y');
   axis([xMinHot xMaxHot 0 yMax]);

end
