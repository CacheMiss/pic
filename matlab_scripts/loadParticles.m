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
function [hot cold] = loadParticles(fileName, varargin)
   f = fopen(fileName, 'rb');

   if f <= 0
      error('Unable to open "%s"\n', fileName);
   end
   
   optArgs = parseArgs(varargin);
   
   fseek(f, 4, 'cof'); % Skip numParticles
   numHot = fread(f, 1, 'int32');
   numCold = fread(f, 1, 'int32');
   if optArgs.loadHot
       fprintf('Num hot = %d\n', numHot);
   end
   if optArgs.loadCold
       fprintf('Num cold = %d\n', numCold);
   end
   sizeOfFloat = 4;
   
   maxHot = numHot;
   maxCold = numCold;
   sliceHot = 1;
   sliceCold = 1;
   
   if optArgs.enableCulling
       if(optArgs.loadHot && numHot > optArgs.maxParticles)
           maxHot = optArgs.maxParticles;
           sliceHot = floor(numHot / maxHot);
           fprintf('Limiting number of hot particles to %d\n', ...
               optArgs.maxParticles);
       end
       if(optArgs.loadCold && numCold > optArgs.maxParticles)
           maxCold = optArgs.maxParticles;
           sliceCold = floor(numCold / maxCold);
           fprintf('Limiting number of cold particles to %d\n', ...
               optArgs.maxParticles);
       end
   end
   
   if optArgs.loadHot
       hot = particleNull(maxHot);
   else
       hot = particleNull(0);
   end
   
   if optArgs.loadCold
       cold = particleNull(maxCold);
   else
       cold = particleNull(0);
   end

   % Store the location of the beginning of the hot particles
   dataStart = ftell(f);
   
   if optArgs.loadHot
       hot = readParticles(f, maxHot, sliceHot);
   end

   if optArgs.loadCold
       % Reposition file pointer to the beginning of the cold particles
       fseek(f, dataStart, 'bof');
       fseek(f, sizeOfFloat * 5 * numHot, 'cof');
       
       cold = readParticles(f, maxCold, sliceCold);
   end
   
   fclose(f);
end

function ret = readParticles(f, numParticles, slice)
       floatsPerPart = 5;
       sizeOfFloat = 4;
       skipBytes = sizeOfFloat * ...
           (floatsPerPart - 1 + floatsPerPart * (slice-1));
       dataStart = ftell(f);
       ret.x = fread(f, numParticles, 'float', skipBytes);
       advanceToField(f, dataStart, 2);
       ret.y = fread(f, numParticles, 'float', skipBytes);
       advanceToField(f, dataStart, 3);
       ret.vx = fread(f, numParticles, 'float', skipBytes);
       advanceToField(f, dataStart, 4);
       ret.vy = fread(f, numParticles, 'float', skipBytes);
       advanceToField(f, dataStart, 5);
       ret.vz = fread(f, numParticles, 'float', skipBytes);
end

function advanceToField(f, start, fieldNum)
   sizeOfFloat = 4;
   fseek(f, start, 'bof');
   fseek(f, (fieldNum-1) * sizeOfFloat, 'cof');
end

function ret = parseArgs(args)
    ret = struct( ...
        'enableCulling', false, ...
        'maxParticles', 64000, ...
        'loadHot', true, ...
        'loadCold', true ...
        );
    if ~isempty(args)
        i = 1;
        while i <= length(args)
            if strcmp(args{i}, 'enableCulling')
                ret.enableCulling = args{i+1};
                i = i + 1;
            elseif strcmp(args{i}, 'maxParticles')
                ret.maxParticles = args{i+1};
                i = i + 1;
            elseif strcmp(args{i}, 'loadHot')
                ret.loadHot = args{i+1};
                i = i + 1;
            elseif strcmp(args{i}, 'loadCold')
                ret.loadCold = args{i+1};
                i = i + 1;
            else
                error('Invalid option!');
            end
            i = i + 1;
        end
    end
end