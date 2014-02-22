
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
   title('BYM vs Y');
   xlabel('Y');
   ylabel('BYM');
   axis([0 max(yValues) 0 max(by)]);
   print('-dpng', 'bym_vs_y');

   fclose(f);
   
   f = fopen('bxm.dat', 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', 'bxm.dat');
      return;
   end

   numColumns = fread(f, 1, 'int32');
   numRows = fread(f, 1, 'int32');
   bx = zeros(numRows, numColumns);
   for j=1:numRows
       bx(j,:) = fread(f, numColumns, 'float');
   end
   
   bigBy = bx;
   for i = 1:numColumns
       bigBy(:,i) = by;
   end 
   
   %for i = 1:numColumns
   %    bx(:,i) = bx(:,i) ./ by;
   %end

   sliceX = 8;
   sliceY = 64;
   
   xValues = 0:sliceX:numColumns-1;
   yValues = 0:sliceY:numRows-1;
   
   figure;
   bxSurf = bx(1:sliceY:numRows-1, 1:sliceX:numColumns-1);
   surf(xValues, yValues, bxSurf);
   title('bx');
   xlabel('X');
   ylabel('Y');
   zlabel('bz');
   colorbar;
   title('BX Surface');
   axis([0 max(xValues) 0 max(yValues)]);
   print('-dpng', 'bxm_surface');
   
   figure;
   contour(xValues, yValues, bxSurf);
   xlabel('X');
   ylabel('Y');
   zlabel('Z');
   colorbar;
   title('BX Contour');
   print('-dpng', 'bxm_contour');
   
   figure;
   plot(1:numColumns, bx(1,:));
   xlabel('X');
   ylabel('BX');
   title('BX vs. X for y=1');
   axis([0 numColumns min(bx(1,:)) max(bx(1,:))]);
   print('-dpng', 'bxm_vs_x_bottom');
   
   figure;
   plot(1:numColumns, bx(numRows,:));
   xlabel('X');
   ylabel('BX');
   title('BX vs. X for y=max');
   axis([0 numColumns min(bx(numRows,:)) max(bx(numRows,:))]);
   print('-dpng', 'bxm_vs_x_top');
   
   %bx = bx(1:sliceY:end, 1:sliceX:end);
   %bigBy = bigBy(1:sliceY:end, 1:sliceX:end);
   %figure;
   %quiver(xValues, yValues, bx, bigBy);

   fclose(f);

end
