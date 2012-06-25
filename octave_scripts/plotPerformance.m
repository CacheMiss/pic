
function ret = plotPerformance(fName)

   d = load(fName);

   %labels = ['Iteration Num', 'Sim Time', 'Num Ele Hot', 'Num Ele Cold' \
   %          'Num Ion Hot', 'Num Ion Cold', 'Iteration Time (ms)'];

   figure;
   plot(d(:,2), d(:,7));
   xlabel('Sim Time (s)');
   ylabel('Iteration Time (ms)');
   totalParticles = d(:,3) + d(:,4) + d(:,5) + d(:,6);
   figure;
   plot(totalParticles, d(:,7));
   xlabel('Num Particles');
   ylabel('Iteration Time (ms)');
   figure;
   plot(d(:,2), totalParticles);
   xlabel('Sim Time (s)');
   ylabel('Num Particles');


endfunction
