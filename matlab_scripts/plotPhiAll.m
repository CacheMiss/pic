
function ret = plotPhiAll(fName, titleStr, sliceX, sliceY)

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
   fclose(f);

   xValues = [0:numColumns-1];
   yValues = [0:numRows-1];

   phi = phi(1:sliceY:end, 1:sliceX:end);
   xValues = xValues(1:sliceX:end);
   yValues = yValues(1:sliceY:end);

   figure;
   surf(xValues, yValues, phi);
   colorbar;
   title(titleStr);
   axis([0 max(xValues) 0 max(yValues)]);

end
