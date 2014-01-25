% Randomize a particle structure to increase printing speed
function ret = particleMix(part)
   len = length(part.x);
   % Create a vector of random integers from 1 to len
   randInd = randperm(len);
   % Randomize the input vectors
   part.x = part.x(randInd);
   part.y = part.y(randInd);
   part.vx = part.vx(randInd);
   part.vy = part.vy(randInd);
   part.vz = part.vz(randInd);
   
   ret = part;
end