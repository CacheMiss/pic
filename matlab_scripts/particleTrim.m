% Resize the arrays cutting off the right side
function ret = particleTrim(part, newSize)
    part.x = part.x(1:newSize);
    part.y = part.y(1:newSize);
    part.vx = part.vx(1:newSize);
    part.vy = part.vy(1:newSize);
    part.vz = part.vz(1:newSize);
    
    ret = part;
end