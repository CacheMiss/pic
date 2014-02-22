
function ret = contourPhi(fName, sliceX, sliceY)

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
   
   phi = phi(1:sliceY:end, 1:sliceX:end);
   xValues = 0:sliceX:numColumns-1;
   yValues = 0:sliceY:numRows-1;
   
   % Chop things off
   %phi = phi(5077:5117, :);
   %xValues = 1:numColumns;
   %yValues = 5077:5117;

%    sliceSize = 4;
%    phi = phi(1:sliceSize:end, 1:sliceSize:end);
%    xValues = xValues(1:sliceSize:end);
%    yValues = yValues(1:sliceSize:end);
   
   figure;
   contourf(xValues, yValues, phi);
   colorbar;
   caxis([-15 5]);
   fields = strsplit(fName, '_');
   title(strcat([fields{1} ' ' fields{2}]));
   print('-dpng', strcat(fields{1}, '_contour_', fields{2}));

end
