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
function ret = plotVy(fName, titleStr, sliceSize)

   f = fopen(fName, 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', fName);
      return;
   end

   numParticles = fread(f, 1, 'int32');
   numHot = fread(f, 1, 'int32');
   numCold = fread(f, 1, 'int32');
   finalHot = floor(numHot / sliceSize);
   finalCold = floor(numCold / sliceSize);
   hotP = zeros(2, finalHot);
   coldP = zeros(2, finalCold);
   for i=1:finalHot
       sizeOfFloat = 4;
       fseek(f, sizeOfFloat, 'cof'); % Skip x
       hotP(2,i) = fread(f, 1, 'float'); % Read y
       fseek(f, sizeOfFloat, 'cof'); % Skip vx
       hotP(1,i) = fread(f, 1, 'float'); % Read vy
       % Skip the  remaining velocity plus whatever else I need to reach
       % the next particle I care about
       skipBytes = sizeOfFloat * 1 + sizeOfFloat * 5 * (sliceSize-1);
       fseek(f, skipBytes, 'cof');
   end
   for i=1:finalCold
       sizeOfFloat = 4;
       fseek(f, sizeOfFloat, 'cof'); % Skip x
       coldP(2,i) = fread(f, 1, 'float'); % Read y
       fseek(f, sizeOfFloat, 'cof'); % Skip vx
       coldP(1,i) = fread(f, 1, 'float'); % Read vy
       % Skip the  remaining velocity plus whatever else I need to reach
       % the next particle I care about
       skipBytes = sizeOfFloat * 1 + sizeOfFloat * 5 * (sliceSize-1);
       fseek(f, skipBytes, 'cof');
   end
   fclose(f);
   yMax = 2^nextpow2(max(hotP(2,:)));
   yMax = max(yMax, 2^nextpow2(max(coldP(2,:))));
   
   stdDevMultiplier = 2;
   stdHotX = std(hotP(1,:));
   stdColdX = std(coldP(1,:));
   meanHotX = mean(hotP(1,:));
   meanColdX = mean(coldP(1,:));
   
   xMaxHot = meanHotX + stdHotX * stdDevMultiplier;
   xMaxCold = meanColdX + stdColdX * stdDevMultiplier;
   
   xMinHot = meanHotX - stdHotX * stdDevMultiplier;
   xMinCold = meanColdX - stdColdX * stdDevMultiplier;
   
   figure;
   scatter(hotP(1,:), hotP(2,:), 0.4)
   title(strcat([titleStr ' Hot']));
   xlabel('vy');
   ylabel('y');
   axis([xMinHot xMaxHot 0 yMax]);

   figure;
   scatter(coldP(1,:), coldP(2,:), 0.4)
   title(strcat([titleStr ' Cold']));
   xlabel('vy');
   ylabel('v');
   axis([xMinCold xMaxHot 0 yMax]);

end
