
function ret = pmovie()
   % get file list
   fileList = ls('phi*');
   tmp = size(fileList);
   numFiles = tmp(1);

   gridSize = getGridSize(fileList(1,:));
   xSize = gridSize(1);
   ySize = gridSize(2);
   maxPhi = 0;
   minPhi = 0;
   for i = 1:numFiles
      phi = loadPhi(fileList(i,:));
      maxPhi = max(max(max(max(phi)), maxPhi));
      minPhi = min(min(min(min(phi)), minPhi));
   end

   xValues = [0:xSize-1];
   yValues = [0:ySize-1];

   fig1 = figure('Position', [100 100 800 600]);
   winsize = get(fig1, 'Position');
   winsize(1:2) = [0 0];

   % Set up movie
   A=moviein(numFiles*5,fig1,winsize);
   fps = 1;

   set(fig1,'NextPlot','replacechildren')

   for i = 1:numFiles
      phi = loadPhi(fileList(i,:));
      %surf(xValues, yValues, phi);
      %A(:,i)=getframe(fig1,winsize);
      %axis([0, xSize, 0, ySize, minPhi, maxPhi]);
      surf(xValues, yValues, phi);
      axis([0, xSize, 0, ySize, minPhi, maxPhi]);
      A(:,(i-1)*5+1)=getframe(fig1,winsize);
      A(:,(i-1)*5+2)=getframe(fig1,winsize);
      A(:,(i-1)*5+3)=getframe(fig1,winsize);
      A(:,(i-1)*5+4)=getframe(fig1,winsize);
      A(:,(i-1)*5+5)=getframe(fig1,winsize);
   end
   clear phi;
   movie(fig1,A,1,fps,winsize);

   ret = 0;
end
