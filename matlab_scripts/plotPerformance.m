
function ret = plotPerformance(fName)

   d = load(fName);

   %labels = ['Iteration Num', 'Sim Time', 'Num Ele Hot', 'Num Ele Cold' \
   %          'Num Ion Hot', 'Num Ion Cold', 'Iteration Time (ms)',
   %          'Inject Time', 'Dens Time', 'Potent2 Time',
   %          'Field Time', 'Movep Time'];

   figure;
   plot(d(:,2), smoothLine(d(:,7),16));
   xlabel('Sim Time (s)');
   ylabel('Iteration Time (ms)');
   totalParticles = d(:,3) + d(:,4) + d(:,5) + d(:,6);
   figure;
   plot(totalParticles, smoothLine(d(:,7),16));
   xlabel('Num Particles');
   ylabel('Iteration Time (ms)');
   figure;
   plot(d(:,2), totalParticles);
   xlabel('Sim Time (s)');
   ylabel('Num Particles');
   figure;
   plot(d(:,2), smoothLine(d(:,9),16));
   xlabel('Sim Time');
   ylabel('Dens Time (ms)');
   figure;
   plot(d(:,2), smoothLine(d(:,12),16));
   xlabel('Sim Time');
   ylabel('Movep Time (ms)');

end

function ret = smoothLine(line, window)
   halfWin = window / 2;
   lineSize = size(line);
   lineSize = lineSize(1);
   ret = zeros(lineSize, 1);
   for i=1 : lineSize
       if i < halfWin+1
           ret(i) = mean(line(1:i+halfWin));
       elseif i < lineSize-halfWin
           ret(i) = mean(line(i-halfWin:i+halfWin));
       else
           ret(i) = mean(line(i-halfWin:end));
       end
   end
end