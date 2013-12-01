
function ret = contourRhoBottom(fName)

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

   %xValues = [0:numColumns-1];
   %yValues = [0:numRows-1];
   xValues = [95:165];
   yValues = [0:4-1];
   
   phi = phi(1:4, 95:165);
   %phi = smoothn(phi);
   %phi = phi(1:sliceY:end, 1:sliceX:end);
   %xValues = xValues(1:sliceX:end);
   %yValues = yValues(1:sliceY:end);
   
   figure;
   contour(xValues, yValues, phi);
   colorbar;
   fields = strsplit(fName, '_');
   title(strcat([fields{1} ' ' fields{2}]));
   print('-dpng', strcat(fields{1}, '_contourBottom_', fields{2}));

end
