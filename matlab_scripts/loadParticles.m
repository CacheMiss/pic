function [hot cold] = loadParticles(fileName, varargin)
   f = fopen(fileName, 'rb');

   if f <= 0
      error('Unable to open "%s"\n', fName);
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
   
   floatsPerPart = 5;
   
   if optArgs.loadHot
       % Read hot particles
       skipBytes = sizeOfFloat * ...
           (floatsPerPart - 1 + floatsPerPart * sliceHot);
       hot.x = fread(f, maxHot, 'float', skipBytes);
       advanceToField(f, dataStart, 2);
       hot.y = fread(f, maxHot, 'float', skipBytes);
       advanceToField(f, dataStart, 3);
       hot.vx = fread(f, maxHot, 'float', skipBytes);
       advanceToField(f, dataStart, 4);
       hot.vy = fread(f, maxHot, 'float', skipBytes);
       advanceToField(f, dataStart, 5);
       hot.vz = fread(f, maxHot, 'float', skipBytes);
   end

   if optArgs.loadCold
       % Reposition file pointer to the beginning of the cold particles
       fseek(f, dataStart, 'bof');
       fseek(f, sizeOfFloat * 5 * numHot, 'cof');
       % Store the location of the beginning of the cold particles
       dataStart = ftell(f);

       skipBytes = sizeOfFloat * ...
           (floatsPerPart - 1 + floatsPerPart * sliceCold);
       cold.x = fread(f, maxCold, 'float', skipBytes);
       advanceToField(f, dataStart, 2);
       cold.y = fread(f, maxCold, 'float', skipBytes);
       advanceToField(f, dataStart, 3);
       cold.vx = fread(f, maxCold, 'float', skipBytes);
       advanceToField(f, dataStart, 4);
       cold.vy = fread(f, maxCold, 'float', skipBytes);
       advanceToField(f, dataStart, 5);
       cold.vz = fread(f, maxCold, 'float', skipBytes);
   end
   
   fclose(f);
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