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
        gridWidth = 512;
        
        plotPart(eleName, partStepSize);
        plotPart(ionName, partStepSize);
        plotVxRange(eleName, gridWidth/2, 20, partStepSize, partStepSize);
        plotVxRange(ionName, gridWidth/2, 20, partStepSize, partStepSize);
        plotVyRange(eleName, gridWidth/2, 20, partStepSize, partStepSize);
        plotVyRange(ionName, gridWidth/2, 20, partStepSize, partStepSize);
        plotVxVy(eleName, partStepSize, partStepSize);
        plotVxVy(ionName, partStepSize, partStepSize);
        plotRho(idxStr, gridWidth/2);
        contourPhi(rhoName, 4, 32);
        plotPhiAll(rhoiName, 4, 32);
        contourRho(rhoiName, 4, 32);
        %contourRhoBottom(rhoiName);
        plotPhi(phiName, gridWidth/2);
        plotPhi(phiAvgName, gridWidth/2);
        plotPhiAll(phiName, 4, 32);
        contourPhi(phiName, 4, 32);
        contourPhi(phiAvgName, 4, 32);
        
        % Find all windows of type figure, which have an empty FileName attribute.
        allPlots = findall(0, 'Type', 'figure', 'FileName', []);
        % Close.
        delete(allPlots);
        fclose('all');
    end
end