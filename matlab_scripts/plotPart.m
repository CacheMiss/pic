function ret = plotPart(fName, sliceSize)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end
   
   fNameTokens = strsplit(fName, '_');
   
   massRatio = 400;
   if strcmp(fNameTokens{1}, 'ele')
       massRatio = 1;
   end

   numParticles = fread(f, 1, 'int32');
   numHot = fread(f, 1, 'int32');
   numCold = fread(f, 1, 'int32');
   fprintf('Num hot = %d\nNum cold = %d\n', numHot, numCold);
   finalHot = floor(numHot / sliceSize);
   finalCold = floor(numCold / sliceSize);
   hotP = zeros(3, finalHot);
   coldP = zeros(3, finalCold);
   for i=1:finalHot
       hotP(1:2,i) = fread(f, 2, 'float');
       vel = fread(f, 3, 'float');
       % e = 1/2 * mv^2
       energy = massRatio * (vel(1)^2 + vel(2)^2 + vel(3)^2) / 2;
       hotP(3,i) = energy;
       sizeOfFloat = 4;
       % Skip the velocity plus whatever else I need to reach
       % the next particle I care about
       skipBytes = sizeOfFloat * 5 * (sliceSize-1);
       fseek(f, skipBytes, 'cof');
   end
   for i=1:finalCold
       coldP(1:2,i) = fread(f, 2, 'float');
       vel = fread(f, 3, 'float');
       % e = 1/2 * mv^2
       energy = massRatio * (vel(1)^2 + vel(2)^2 + vel(3)^2) / 2;
       coldP(3,i) = energy;
       sizeOfFloat = 4;
       % Skip the velocity plus whatever else I need to reach
       % the next particle I care about
       skipBytes = sizeOfFloat * 5 * (sliceSize-1);
       fseek(f, skipBytes, 'cof');
   end

   fclose(f);
   dotScale = 20;
   if ~ isempty(hotP)
       xMax = 2^nextpow2(max(hotP(1,:)));
       % yMax = 2^nextpow2(max(hotP(2,:)));
       yMax = max(hotP(2,:));
       
       figure;
       scatter(hotP(1,:), hotP(2,:), hotP(3,:)/norm(hotP(3,:))*dotScale, hotP(3,:));
       colorbar;
       titleStr = strcat([fNameTokens{1}, ' hot ', fNameTokens{2}]);
       title(titleStr);
       xlabel('x');
       ylabel('y');
       axis([0 xMax 0 yMax]);
       outName = strcat(fNameTokens{1}, '_hot_', fNameTokens{2});
       print('-dpng', outName);
   end
   
   if ~ isempty(coldP)
       xMax = max(xMax, 2^nextpow2(max(coldP(1,:))));
       % yMax = max(yMax, 2^nextpow2(max(coldP(2,:))));
       yMax = max(coldP(2,:));
       
       figure;
       scatter(coldP(1,:), coldP(2,:), coldP(3,:)/norm(coldP(3,:))*dotScale, coldP(3,:));
       colorbar;
       titleStr = strcat([fNameTokens{1}, ' cold ', fNameTokens{2}]);
       title(titleStr);
       xlabel('x');
       ylabel('y');
       axis([0 xMax 0 yMax]);
       outName = strcat(fNameTokens{1}, '_cold_', fNameTokens{2});
       print('-dpng', outName);
   end

end
