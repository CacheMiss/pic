function ret = plotPart(fName, sliceSize)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end
   
   fNameTokens = strsplit(fName, '_');

   numParticles = fread(f, 1, 'int32');
   numHot = fread(f, 1, 'int32');
   numCold = fread(f, 1, 'int32');
   fprintf('Num hot = %d\nNum cold = %d\n', numHot, numCold);
   finalHot = floor(numHot / sliceSize);
   finalCold = floor(numCold / sliceSize);
   hotP = zeros(2, finalHot);
   coldP = zeros(2, finalCold);
   for i=1:finalHot
       hotP(:,i) = fread(f, 2, 'float');
       sizeOfFloat = 4;
       % Skip the velocity plus whatever else I need to reach
       % the next particle I care about
       skipBytes = sizeOfFloat * 3 + sizeOfFloat * 5 * (sliceSize-1);
       fseek(f, skipBytes, 'cof');
   end
   for i=1:finalCold
       coldP(:,i) = fread(f, 2, 'float');
       sizeOfFloat = 4;
       % Skip the velocity plus whatever else I need to reach
       % the next particle I care about
       skipBytes = sizeOfFloat * 3 + sizeOfFloat * 5 * (sliceSize-1);
       fseek(f, skipBytes, 'cof');
   end

   fclose(f);
   
   if ~ isempty(hotP)
       xMax = 2^nextpow2(max(hotP(1,:)));
       % yMax = 2^nextpow2(max(hotP(2,:)));
       yMax = max(hotP(2,:));
       
       figure;
       scatter(hotP(1,:), hotP(2,:), 0.1)
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
       scatter(coldP(1,:), coldP(2,:), 0.1)
       titleStr = strcat([fNameTokens{1}, ' cold ', fNameTokens{2}]);
       title(titleStr);
       xlabel('x');
       ylabel('y');
       axis([0 xMax 0 yMax]);
       outName = strcat(fNameTokens{1}, '_cold_', fNameTokens{2});
       print('-dpng', outName);
   end

end
