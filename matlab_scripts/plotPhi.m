
function ret = plotPhi(fName)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end

   numRows = fread(f, 1, 'int32');
   numColumns = fread(f, 1, 'int32');
   columnOrder = fread(f, 1, 'int32');
   phi = fread(f, [numRows,numColumns], 'float');

   yValues = [0:numRows-1];
   figure;
   plot(yValues, phi(:,1));
   title(fName);
   xlabel('y');
   ylabel('phi');

   fclose(f);

end
