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
   finalNum = floor(numParticles / sliceSize);
   particles = zeros(2, finalNum);
   for i=1:finalNum
       particles(:,i) = fread(f, 2, 'float');
       sizeOfFloat = 4;
       % Skip the velocity plus whatever else I need to reach
       % the next particle I care about
       skipBytes = sizeOfFloat * 3 + sizeOfFloat * 5 * (sliceSize-1);
       fseek(f, skipBytes, 'cof');
   end
   fclose(f);
   xMax = 2^nextpow2(max(particles(1,:)));
   yMax = 2^nextpow2(max(particles(2,:)));
   
   figure;
   scatter(particles(1,:), particles(2,:), 0.1) % o means no line between points
   xlabel('x');
   ylabel('y');
   axis([0 xMax 0 yMax]);

end
