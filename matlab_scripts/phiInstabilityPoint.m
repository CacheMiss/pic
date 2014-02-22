
function ret = phiInstabilityPoint(x, y)
   % get file list
   fileList = ls('phi_*');
   tmp = size(fileList);
   numFiles = tmp(1);

   phi = zeros(numFiles, 1);
   indexList = zeros(numFiles, 2);
   for i = 1:numFiles
      tmp = textscan(fileList(i,:),'%s','delimiter','_');
      indexList(i, 1) = str2double(tmp{1}(2));
      f = fopen(fileList(i,:), 'rb');
      %numRows = fread(f, 1, 'int32');
      fseek(f, 4, 'cof');
      numColumns = fread(f, 1, 'int32');
      %columnOrder = fread(f, 1, 'int32');
      fseek(f, 4, 'cof');
      % Find the right phi value
      sizeOfFloat = 4;
      fseek(f, sizeOfFloat * ((y-1) * numColumns + (x-1)), 'cof');
      phi(i) = fread(f, 1, 'float');
      fclose(f);
   end
   
   info  = load('info');
   numLines = size(info,1);
   nextSlot = 1;
   i = 1;
   while(i <= numLines && nextSlot <= numFiles)
       if info(i,1) == indexList(nextSlot, 1)
           indexList(nextSlot, 2) = info(i, 2);
           nextSlot = nextSlot + 1;
       end
       i = i + 1;
   end

   clear allPhi;
   xValues = indexList(:,2);
   
   xValues = xValues(1:50);
   phi = phi(1:50);

   figure;
   plot(xValues, phi);
   xlabel('time (s)');
   ylabel('phi');
   axis([min(xValues) max(xValues) min(phi) max(phi)]);
   
   ret = 0;
end
