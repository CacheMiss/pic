function ret = plotPart(fName, sliceSize)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end

   numParticles = fread(f, 1, 'int32');
   particles = fread(f, [6, numParticles], 'float');
   fclose(f);
   % Remove velocity to save memory
   particles(3:5,:) = [];
   xMax = 2^nextpow2(max(particles(1,:)));
   yMax = 2^nextpow2(max(particles(2,:)));
   
   particles = particles(:, 1:sliceSize:end);
   hotEnd =  find(particles(3,:), 1, 'last');
   coldBeg = hotEnd + 1;
   hotP = particles(1:2, 1:hotEnd);
   coldP = particles(1:2, coldBeg:end);
   particles = [];

   figure;
   scatter(hotP(1,:), hotP(2,:), 0.1) % o means no line between points
   title('Hot');
   xlabel('x');
   ylabel('y');
   axis([0 xMax 0 yMax]);

   figure;
   scatter(coldP(1,:), coldP(2,:), 0.1) % o means no line between points
   title('Cold');
   xlabel('x');
   ylabel('y');
   axis([0 xMax 0 yMax]);

end
