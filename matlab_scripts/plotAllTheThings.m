function plotAllTheThings(startIdx, endIdx, stepSize)
    for i=startIdx:stepSize:endIdx
        idxStr = sprintf('%06d', i);
        eleName = strcat('ele_', idxStr);
        ionName = strcat('ion_', idxStr);
        rhoName = strcat('rho_', idxStr);
        rhoeName = strcat('rhoe_', idxStr);
        rhoiName = strcat('rhoi_', idxStr);
        phiName = strcat('phi_', idxStr);
        phiAvgName = strcat('phiAvg_', idxStr);
        
        partStepSize = 512;
        plotPart(eleName, partStepSize);
        plotPart(ionName, partStepSize);
        plotVxRange(eleName, 128, 20, partStepSize, partStepSize);
        plotVxRange(ionName, 128, 20, partStepSize, partStepSize);
        plotVyRange(eleName, 128, 20, partStepSize, partStepSize);
        plotVyRange(ionName, 128, 20, partStepSize, partStepSize);
        plotVxVy(eleName, partStepSize, partStepSize);
        plotVxVy(ionName, partStepSize, partStepSize);
        plotRho(idxStr, 128);
        contourPhi(rhoName);
        plotPhiAll(rhoiName, 4, 32);
        contourRho(rhoiName, 4, 32);
        %contourRhoBottom(rhoiName);
        plotPhi(phiName, 128);
        plotPhi(phiAvgName, 128);
        plotPhiAll(phiName, 4, 32);
        contourPhi(phiName);
        contourPhi(phiAvgName);
        
        % Find all windows of type figure, which have an empty FileName attribute.
        allPlots = findall(0, 'Type', 'figure', 'FileName', []);
        % Close.
        delete(allPlots);
        fclose('all');
    end
end