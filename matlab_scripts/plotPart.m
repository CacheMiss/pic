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
   hotP = fread(f, [5, numHot], 'float');
   % Remove velocity to save memory
   hotP(3:5,:) = [];
   coldP = fread(f, [5, numCold], 'float');
   % Remove velocity to save memory
   coldP(3:5,:) = [];
   fclose(f);
   xMax = 2^nextpow2(max(hotP(1,:)));
   xMax = max(xMax, 2^nextpow2(max(coldP(1,:))));
   yMax = 2^nextpow2(max(hotP(2,:)));
   yMax = max(yMax, 2^nextpow2(max(coldP(2,:))));
   
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
