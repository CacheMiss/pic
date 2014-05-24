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
