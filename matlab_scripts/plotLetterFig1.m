function plotLetterFig1
    phiData = loadPic2dFile('phi_160000');
    f = figure;
    
    subplotWidth = 0.46;
    subplotHeight = 0.88;
    
    %subplot(1, 2, 1);
    nextWidth = 0.05;
    subplot('Position', [nextWidth 0.05 subplotWidth subplotHeight]);
    phi = phiData(:, 256);
    clear phiData;
    plot(0:size(phi, 1)-1, phi);
    axis([0, size(phi, 1)-1 min(phi) max(phi)]);
    title('Figure 1a: Potential');
    clear phi;
    
    rhoiData = loadPic2dFile('rhoi_160000');
    rhoi = rhoiData(:, 256);
    clear rhoiData;
    rhoi = smoothLine(rhoi, 64);
    
    rhoeData = loadPic2dFile('rhoe_160000');
    rhoe = rhoeData(:, 256);
    clear rhoeData;
    rhoe = smoothLine(rhoe, 64);
    
    %subplot(1, 2, 2);
    nextWidth = nextWidth + subplotWidth + 0.01;
    subplot('Position', [nextWidth 0.05 subplotWidth subplotHeight]);
    semilogy(0:size(rhoi, 1)-1, rhoi, 0:size(rhoe, 1)-1, rhoe);
    title('Figure 1b: Density');
    legend('\rho_{e}', '\rho_{i}');
    set(gca, 'XTickLabelMode', 'Manual')
    set(gca, 'YTick', [])
    
    saveSameSize(f, 'format', '-dpdfwrite', 'file', 'figure1');
end

function s = smoothLine(yValues, avgWindow)
   window = avgWindow;
   halfWin = window / 2;
   numValues = size(yValues);
   numValues = numValues(1);
   s = zeros(numValues, 1);
   for i=1 : numValues
       if i < halfWin+1
           s(i) = mean(yValues(1:i+halfWin));
       elseif i < numValues-halfWin
           s(i) = mean(yValues(i-halfWin:i+halfWin));
       else
           s(i) = mean(yValues(i-halfWin:end));
       end
   end
end