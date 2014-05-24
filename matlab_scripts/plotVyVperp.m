%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2014, Stephen C. Sewell
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
% 1. Redistributions of source code must retain the above copyright notice, this
%    list of conditions and the following disclaimer.
% 
% 2. Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the documentation
%    and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
% ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ret = plotVyVperp(fName, xMin, xMax, yMin, yMax)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end
   
   yMinStr = int2str(yMin);
   yMaxStr = int2str(yMax);
   
   vperpMin = 0;
   vperpMax = 7;
   vyMin = -6;
   vyMax = 2;
   cellSize = 0.25;
   bins = zeros((vyMax - vyMin) / cellSize, (vperpMax - vperpMin) / cellSize);
   if vperpMin < 0
       vperpBinOffset = abs(vperpMin);
   else
       vperpBinOffset = -vperpMin;
   end
   if vyMin < 0
       vyBinOffset = abs(vyMin);
   else
       vyBinOffset = -vyMin;
   end
   
   sizeOfFloat = 4;
   maxPlottableParticles = 4000;

   numParticles = fread(f, 1, 'int32');
   numHot = fread(f, 1, 'int32');
   numHotCulled = numHot;
   numCold = fread(f, 1, 'int32');
   %numColdCulled = numCold;
   dataStart = ftell(f);
   
%    if numHotCulled > maxPlottableParticles
%       numHotCulled = maxPlottableParticles;
%       sliceHot = floor(numHot / maxPlottableParticles);
%       fprintf('Limiting number of hot particles plotted to %d\n', ...
%           numHotCulled);
%    end
%    if numColdCulled > maxPlottableParticles
%        numColdCulled = maxPlottableParticles;
%        sliceCold = floor(numCold / maxPlottableParticles);
%        fprintf('Limiting number of cold particles plotted to %d\n', ...
%            numColdCulled);
%    end
   
%    hotP = zeros(4, numHotCulled);
%    coldP = zeros(4, numColdCulled);
%    nextSpace = 1;
   for i=1:numHotCulled
      tmpX = fread(f, 1, 'float'); % Read x
      tmpY = fread(f, 1, 'float'); % Read y
      tmpVx = fread(f, 1, 'float'); % Read vx
      tmpVy = fread(f, 1, 'float'); % Read vy
      tmpVz = fread(f, 1, 'float'); % Read vz
      %if tmpY >= yMin && tmpY <= yMaxv
      if tmpY >= yMin && tmpY <= yMax && tmpX > xMin && tmpX < xMax
          vyIdx = int32((tmpVy + vyBinOffset)/cellSize) + 1;
          vperp = sqrt(tmpVx^2 + tmpVz^2);
          vperpIdx = int32((vperp + vperpBinOffset)/cellSize) + 1;
          binSize = size(bins);
          if vyIdx > 0 && vperpIdx > 0 && vyIdx <= binSize(1) && vperpIdx <= binSize(2)
              bins(vyIdx, vperpIdx) = bins(vyIdx, vperpIdx) + 1;
          end
%           hotP(1,nextSpace) = tmpY;
%           hotP(2,nextSpace) = tmpVx;
%           hotP(3,nextSpace) = tmpVy;
%           hotP(4,nextSpace) = tmpVz;
%           nextSpace = nextSpace + 1;
      end
      % Skip whatever I need to reach
      % the next particle I care about
%       skipBytes = sizeOfFloat * 5 * (sliceHot-1);
%       fseek(f, skipBytes, 'cof');
   end
%    fseek(f, dataStart, 'bof');
%    fseek(f, sizeOfFloat * 5 * numHot, 'cof');
%    hotP = hotP(:, 1:nextSpace-1);
%    nextSpace = 1;
%    for i=1:numColdCulled
%       fseek(f, sizeOfFloat, 'cof'); % Skip x
%       tmpY = fread(f, 1, 'float'); % Read y
%       tmpVx = fread(f, 1, 'float'); % Read vx
%       tmpVy = fread(f, 1, 'float'); % Read vy
%       tmpVz = fread(f, 1, 'float'); % Read vz
%       if tmpY >= yMin && tmpY <= yMax
%           coldP(1,nextSpace) = tmpY;
%           coldP(2,nextSpace) = tmpVx;
%           coldP(3,nextSpace) = tmpVy;
%           coldP(4,nextSpace) = tmpVz;
%           nextSpace = nextSpace + 1;
%       end
%       % Skip whatever I need to reach
%       % the next particle I care about
%       skipBytes = sizeOfFloat * 5 * (sliceCold-1);
%       fseek(f, skipBytes, 'cof');
%    end
%    coldP = coldP(:, 1:nextSpace-1);
   fclose(f);
%    yMax = max(hotP(2,:));
%    
%    xValues = sqrt(coldP(2,:).^2 + coldP(4,:).^2);
%    figure;
%    scatter(xValues, coldP(3,:), 3)
%    fields = strsplit(fName, '_');
%    title(strcat( ...
%        [fields{1} ' cold ' fields{2} ' Vy vs Vperp yMin = ' ...
%        yMinStr ' yMax = ' yMaxStr ]));
%    xlabel('vperp');
%    ylabel('vy');
%    axis([min(xValues) max(xValues) min(coldP(3,:)) max(coldP(3,:))]);
%    fields = strsplit(fName, '_');
%    outFile = strcat(fields{1}, '_cold_vy_vs_vperp_', fields{2}, ...
%        '_y', yMinStr, '-y', yMaxStr);
%    print('-dpng', outFile);
%       
%    xValues = sqrt(hotP(2,:).^2 + hotP(4,:).^2);
%    figure;
%    scatter(xValues, hotP(3,:), 3)
%    fields = strsplit(fName, '_');
%    title(strcat([fields{1} ' hot ' fields{2} ' Vy vs Vperp yMin = ' ...
%        yMinStr ' yMax = ' yMaxStr ' 206 < x < 306']));
%    %title(strcat([fields{1} ' hot ' fields{2} ' Vy vs Vperp yMin = ' ...
%    %    yMinStr ' yMax = ' yMaxStr]));
%    ylabel('vy');
%    %axis([min(xValues) max(xValues) min(hotP(3,:)) max(hotP(3,:))]);
%    axis([0 5.5 -7 5]);
%    fields = strsplit(fName, '_');
%    outFile = strcat(fields{1}, '_hot_vy_vs_vperp_', fields{2}, ...
%        '_y', yMinStr, '-y', yMaxStr);
%    print('-dpng', outFile);

   binSize = size(bins);
   for i=1:binSize(2)
       scaledVperp = (i * cellSize - vperpBinOffset) * 0.25 * 0.25;
       bins(:,i) = bins(:,i) / scaledVperp;
   end
   xValues = vperpMin:cellSize:vperpMax-cellSize;
   yValues = vyMin:cellSize:vyMax-cellSize;
   figure;
   contour(xValues, yValues, bins)
   fields = strsplit(fName, '_');
   xMinStr = int2str(xMin);
   xMaxStr = int2str(xMax);
   title(strcat([fields{1} ' hot ' fields{2} ' Vy vs Vperp yMin = ' ...
       yMinStr ' yMax = ' yMaxStr ' ' xMinStr ' < x < ' xMaxStr]));
   %title(strcat([fields{1} ' hot ' fields{2} ' Vy vs Vperp yMin = ' ...
   %    yMinStr ' yMax = ' yMaxStr]));
   xlabel('vperp');
   ylabel('vy');
   %axis([min(xValues) max(xValues) min(hotP(3,:)) max(hotP(3,:))]);
   axis([vperpMin+cellSize vperpMax vyMin vyMax]);
   colorbar;
   fields = strsplit(fName, '_');
   outFile = strcat(fields{1}, '_hot_vy_vs_vperp_', fields{2}, ...
       '_y', yMinStr, '-y', yMaxStr);
   print('-dpng', outFile);


end
