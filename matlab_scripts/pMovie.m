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

function ret = pmovie(frameDivider)
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
   numFrames = numFiles * frameDivider;
   A=moviein(numFrames,fig1,winsize);
   fps = 1;

   set(fig1,'NextPlot','replacechildren')

   for i = 1:numFiles
      phi = loadPhi(fileList(i,:));
      surf(xValues, yValues, phi);
      axis([0, xSize, 0, ySize, minPhi, maxPhi]);
      for j = 1:frameDivider
         A(:,(i-1)*frameDivider+j)=getframe(fig1,winsize);
      end
   end
   clear phi;
   movie(fig1,A,1,fps,winsize);
   %movie2avi(A, 'test.avi');

   ret = 0;
end
