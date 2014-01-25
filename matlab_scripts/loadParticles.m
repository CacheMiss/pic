function [hot cold] = loadParticles(fileName, varargin)
   f = fopen(fileName, 'rb');

   if f <= 0
      error('Unable to open "%s"\n', fName);
   end
   
   if isempty(varargin)
       % I still need initial values, even if I don't have arguments
       optArgs = parseArgs; 
   else
       optArgs = parseArgs(varargin);
   end
   
   fseek(f, 4, 'cof'); % Skip numParticles
   numHot = fread(f, 1, 'int32');
   numCold = fread(f, 1, 'int32');
   fprintf('Num hot = %d\nNum cold = %d\n', numHot, numCold);
   sizeOfFloat = 4;
   
   maxHot = optArgs.maxParticles;
   maxCold = optArgs.maxParticles;
   sliceHot = 1;
   sliceCold = 1;
   
   if(numHot > optArgs.maxParticles)
       sliceHot = floor(numHot / maxHot);
       fprintf('Limiting number of hot particles to %d\n', ...
           optArgs.maxParticles);
   end
   if(numCold > optArgs.maxParticles)
       sliceCold = floor(numCold / maxCold);
       fprintf('Limiting number of cold particles to %d\n', ...
           optArgs.maxParticles);
   end
   
   hot = particleNull(maxHot);
   cold = particleNull(maxCold);

   % Store the location of the beginning of the hot particles
   dataStart = ftell(f);
   
   floatsPerPart = 5;
   
   % Read hot particles
   skipBytes = sizeOfFloat * ...
       (floatsPerPart - 1 + floatsPerPart * sliceHot);
   hot.x = fread(f, numHot, 'float', skipBytes);
   advanceToField(f, dataStart, 2);
   hot.y = fread(f, numHot, 'float', skipBytes);
   advanceToField(f, dataStart, 3);
   hot.vx = fread(f, numHot, 'float', skipBytes);
   advanceToField(f, dataStart, 4);
   hot.vy = fread(f, numHot, 'float', skipBytes);
   advanceToField(f, dataStart, 5);
   hot.vz = fread(f, numHot, 'float', skipBytes);
   
   % Reposition file pointer to the beginning of the cold particles
   fseek(f, dataStart, 'bof');
   fseek(f, sizeOfFloat * 5 * numHot, 'cof');
   % Store the location of the beginning of the cold particles
   dataStart = ftell(f);
   
   skipBytes = sizeOfFloat * ...
       (floatsPerPart - 1 + floatsPerPart * sliceCold);
   cold.x = fread(f, numCold, 'float', skipBytes);
   advanceToField(f, dataStart, 2);
   cold.y = fread(f, numCold, 'float', skipBytes);
   advanceToField(f, dataStart, 3);
   cold.vx = fread(f, numCold, 'float', skipBytes);
   advanceToField(f, dataStart, 4);
   cold.vy = fread(f, numCold, 'float', skipBytes);
   advanceToField(f, dataStart, 5);
   cold.vz = fread(f, numCold, 'float', skipBytes);
   
   fclose(f);
end

function advanceToField(f, start, fieldNum)
   sizeOfFloat = 4;
   fseek(f, start, 'bof');
   fseek(f, (fieldNum-1) * sizeOfFloat, 'cof');
end

function ret = parseArgs(varargin)
    ret = struct( ...
        'maxParticles', 8000 ...
        );
    if ~isempty(varargin)
        for i = 1:length(varargin)
            if strcmp(varargin{i}(1), 'maxParticles')
                ret.maxParticles = varargin{i}{2};
            else
                error('Invalid option!');
            end
        end
    end
end