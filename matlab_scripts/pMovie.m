
function ret = pmovie(fps)
   % get file list
   fileList = ls('phi*');
   tmp = size(fileList);
   numFiles = tmp(1);

   gridSize = getGridSize(fileList(1,:));
   xSize = gridSize(1);
   ySize = gridSize(2);
   phi = zeros(numFiles, ySize, xSize); 
   maxPhi = 0;
   minPhi = 0;
   for i = 1:numFiles
      phi(i,:,:) = loadPhi(fileList(i,:));
      maxPhi = max(max(max(max(phi)), maxPhi));
      minPhi = min(min(min(min(phi)), minPhi));
   end

   xValues = [0:xSize-1];
   yValues = [0:ySize-1];

   fig1 = figure('Position', [100 100 800 600]);
   winsize = get(fig1, 'Position');
   winsize(1:2) = [0 0];

   % Set up movie
   A=moviein(numFiles,fig1,winsize);
   set(fig1,'NextPlot','replacechildren')

   for i = 1:numFiles
      surf(xValues, yValues, squeeze(phi(i,:,:)));
      axis([0, xSize, 0, ySize, minPhi, maxPhi]);
      A(:,i)=getframe(fig1,winsize);
   end
   clear phi;
   movie(fig1,A,1,fps,winsize);

   ret = 0;
end
