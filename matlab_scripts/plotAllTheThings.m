function plotAllTheThings(startIdx, endIdx, stepSize)
    for i=startIdx:stepSize:endIdx
        idxStr = sprintf('%06d', i);
        eleName = strcat('ele_', idxStr);
        ionName = strcat('ion_', idxStr);
        %rhoName = strcat('rho_', idxStr);
        phiName = strcat('phi_', idxStr);
        %phiAvgName = strcat('phiAvg_', idxStr);
        
        plotPart(eleName, 128);
        plotPart(ionName, 128);
        plotRho(idxStr, 128);
        plotPhi(phiName, 128);
        %plotPhi(phiAvgName, 128);
        plotPhiAll(phiName, 4, 32);
        contourPhi(phiName);
        
        % Find all windows of type figure, which have an empty FileName attribute.
        allPlots = findall(0, 'Type', 'figure', 'FileName', []);
        % Close.
        delete(allPlots);
        fclose('all');
    end
end