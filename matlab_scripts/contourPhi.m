
function ret = contourPhi(fName)

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
   
   % Chop things off
   %phi = phi(5077:5117, :);
   %xValues = 1:numColumns;
   %yValues = 5077:5117;

%    sliceSize = 4;
%    phi = phi(1:sliceSize:end, 1:sliceSize:end);
%    xValues = xValues(1:sliceSize:end);
%    yValues = yValues(1:sliceSize:end);
   
   figure;
   contour(xValues, yValues, phi);
   colorbar;
   fields = strsplit(fName, '_');
   title(strcat([fields{1} ' ' fields{2}]));
   print('-dpng', strcat(fields{1}, '_contour_', fields{2}));

end
