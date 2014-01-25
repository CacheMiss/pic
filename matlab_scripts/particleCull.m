% Keep every skipSize'th particle
function ret = particleCull(part, skipSize)
    part.x = part.x(1:skipSize:end);
    part.y = part.y(1:skipSize:end);
    part.vx = part.vx(1:skipSize:end);
    part.vy = part.vy(1:skipSize:end);
    part.vz = part.vz(1:skipSize:end);
    
    ret = part;
end