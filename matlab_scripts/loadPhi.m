
function ret = plotPhiAll(fName, varargin)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end

   numRows = fread(f, 1, 'int32');
   numColumns = fread(f, 1, 'int32');
   columnOrder = fread(f, 1, 'int32');
   ret = fread(f, [numRows,numColumns], 'float');

   fclose(f);

end
