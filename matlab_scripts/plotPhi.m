
function ret = plotPhi(fName, column)

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
   phi = phi(:,column);

   yValues = 0:numRows-1;
   figure;
   plot(yValues, phi);
   fields = strsplit(fName, '_');
   title(strcat([fields{1} ' ' fields{2} ' x=' int2str(column)]));
   xlabel('y');
   ylabel('phi');
   axis([0 max(yValues) min(phi) max(phi)])
   title(fName);
   print('-dpng', fName);

   fclose(f);

end
