
function ret = bfield()

   f = fopen('bym.dat', 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', 'bym.dat');
      return;
   end

   height = fread(f, 1, 'int32');
   by = fread(f, height, 'float');

   yValues = 0:height-1;
   figure;
   plot(yValues, by);

   fclose(f);
   
   f = fopen('bxm.dat', 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', 'bxm.dat');
      return;
   end

   numColumns = fread(f, 1, 'int32');
   numRows = fread(f, 1, 'int32');
   bx = fread(f, [numRows,numColumns], 'float');
   
   bigBy = bx;
   for i = 1:numColumns
       bigBy(:,i) = by;
   end 
   
   %for i = 1:numColumns
   %    bx(:,i) = bx(:,i) ./ by;
   %end

   sliceSize = 20;
   xValues = 0:sliceSize:numColumns-1;
   yValues = 0:sliceSize:numRows-1;
   bx = bx(1:sliceSize:end, 1:sliceSize:end);
   bigBy = bigBy(1:sliceSize:end, 1:sliceSize:end);
   figure;
   %surf(xValues, yValues, bx);
   quiver(xValues, yValues, bx, bigBy);

   fclose(f);

end
