function ret = plotPart(fName)
   
   fNameTokens = strsplit(fName, '_');
   
   massRatio = 400;
   if strcmp(fNameTokens{1}, 'ele')
       massRatio = 1;
   end
   
   [hotP coldP] = loadParticles(fName);
   hotP = particleMix(hotP);
   coldP = particleMix(coldP);
   energyHot = massRatio * (hotP.vx.^2 + hotP.vy.^2 + hotP.vz.^2) / 2;
   energyCold = massRatio * (coldP.vx.^2 + coldP.vy.^2 + coldP.vz.^2) / 2;
   
   dotScale = 20;
   if ~ isempty(hotP)
       xMax = 2^nextpow2(max(hotP.x));
       % yMax = 2^nextpow2(max(hotP(2,:)));
       yMax = max(hotP.y);
       
       figure;
       scatter(hotP.x, hotP.y, energyHot/norm(energyHot)*dotScale, energyHot);
       colorbar;
       titleStr = strcat([fNameTokens{1}, ' hot ', fNameTokens{2}]);
       title(titleStr);
       xlabel('x');
       ylabel('y');
       axis([0 xMax 0 yMax]);
       outName = strcat(fNameTokens{1}, '_hot_', fNameTokens{2});
       print('-dpng', outName);
   end
   
   if ~ isempty(coldP)
       xMax = max(xMax, 2^nextpow2(max(coldP.x)));
       % yMax = max(yMax, 2^nextpow2(max(coldP(2,:))));
       yMax = max(coldP.y);
       
       figure;
       scatter(coldP.x, coldP.y, energyCold/norm(energyCold)*dotScale, energyCold);
       colorbar;
       titleStr = strcat([fNameTokens{1}, ' cold ', fNameTokens{2}]);
       title(titleStr);
       xlabel('x');
       ylabel('y');
       axis([0 xMax 0 yMax]);
       outName = strcat(fNameTokens{1}, '_cold_', fNameTokens{2});
       print('-dpng', outName);
   end

end
