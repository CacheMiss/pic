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
   finalHot = numHot;
   finalCold = numCold;
   sliceHot = 1;
   sliceCold = 1;
   fprintf('Num hot = %d\nNum cold = %d\n', numHot, numCold);
   sizeOfFloat = 4;
   
   if finalHot > optArgs.maxParticles
       finalHot = optArgs.maxParticles;
       sliceHot = floor(numHot / optArgs.maxParticles);
       fprintf('Limiting number of hot particles plotted to %d\n', ...
           finalHot);
   end
   if finalCold > optArgs.maxParticles
       finalCold = optArgs.maxParticles;
       sliceCold = floor(numCold / optArgs.maxParticles);
       fprintf('Limiting number of cold particles plotted to %d\n', ...
           finalCold);
   end
   hot = particleNull(finalHot);
   cold = particleNull(finalCold);

   dataStart = ftell(f);
   for i=1:finalHot
       hot.x(i) = fread(f, 1, 'float');
       hot.y(i) = fread(f, 1, 'float');
       hot.vx(i) = fread(f, 1, 'float');
       hot.vy(i) = fread(f, 1, 'float');
       hot.vz(i) = fread(f, 1, 'float');

       % Skip to the next particle if necessary
       skipBytes = sizeOfFloat * 5 * (sliceHot-1);
       fseek(f, skipBytes, 'cof');
   end
   fseek(f, dataStart, 'bof');
   fseek(f, sizeOfFloat * 5 * numHot, 'cof');
   for i=1:finalCold
       cold.x(i) = fread(f, 1, 'float');
       cold.y(i) = fread(f, 1, 'float');
       cold.vx(i) = fread(f, 1, 'float');
       cold.vy(i) = fread(f, 1, 'float');
       cold.vz(i) = fread(f, 1, 'float');

       % Skip to the next particle if necessary
       skipBytes = sizeOfFloat * 5 * (sliceCold-1);
       fseek(f, skipBytes, 'cof');
   end

   fclose(f);
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