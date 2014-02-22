function ret = loadPic2dFile(fName)

   f = fopen(fName, 'rb');

   if f <= 0
      error('Unable to open "%s"\n', fName);
   end

   numRows = fread(f, 1, 'int32');
   numColumns = fread(f, 1, 'int32');
   % Skip the columnOrder
   fseek(f, 4, 'cof');
   ret = fread(f, [numRows,numColumns], 'float');
end