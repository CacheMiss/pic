
function ret = phiInstability(frameDivider)
   % get file list
   fileList = ls('phi*');
   tmp = size(fileList);
   numFiles = tmp(1);

   gridSize = getGridSize(fileList(1,:));
   xSize = gridSize(1);
   ySize = gridSize(2);
   phiOverTime = zeros(ySize, numFiles);
   for i = 1:numFiles
      phi = loadPhi(fileList(i,:));
      phiOverTime(:,i) = phi(:,xSize/2);
   end

   clear phi;
   xValues = 1:numFiles;
   yValues = 0:ySize-1;
   
   sliceSize = 4;
   xValues = xValues(1:sliceSize:end);
   yValues = yValues(1:sliceSize:end);
   phiOverTime = phiOverTime(1:sliceSize:end, 1:sliceSize:end);

   figure;
   surf(xValues, yValues, phiOverTime);
   xlabel('time');
   ylabel('height');
   zlabel('phi');
   
   ret = 0;
end
