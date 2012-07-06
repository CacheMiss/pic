
function ret = plotPhiAll(fName)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      %printf('Unable to open "%s"\n', fName);
      return;
   end

   numRows = fread(f, 1, 'int32');
   numColumns = fread(f, 1, 'int32');
   columnOrder = fread(f, 1, 'int32');
   phi = fread(f, [numRows,numColumns], 'float');

   xValues = [0:numColumns-1];
   yValues = [0:numRows-1];
   figure;
   surf(xValues, yValues, phi);

   fclose(f);

end
