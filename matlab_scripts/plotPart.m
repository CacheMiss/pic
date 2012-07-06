function ret = plotPart(fName)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   endif

   partStep = 1000;
   numParticles = fread(f, 1, 'int32');
   particles = fread(f, [numParticles,5], 'float');
   sparsePart = zeros(numParticles/partStep, 2);
   for i = 1:partStep:numParticles
      sparsePart(int32(i/partStep)+1,1) = particles(i,3);
      sparsePart(int32(i/partStep)+1,2) = particles(i,4);
   endfor

   figure;
   scatter(sparsePart(:,1), sparsePart(:,2), 0.0);
   title(fName);
   xlabel('x');
   ylabel('x-vel');

   fclose(f);

end
