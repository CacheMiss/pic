function followParticle(startIdx, endIdx, stepSize, zStr)
    numStates = (endIdx-startIdx)/stepSize;
    hotStates = zeros(5, numStates);
    coldStates = zeros(5, numStates);
    numHotStates = 0;
    numColdStates = 0;
    for i=startIdx:stepSize:endIdx
        idxStr = sprintf('%06d', i);
        eleName = strcat('ele_', idxStr);
        f = fopen(eleName, 'rb');
        if f <= 0
            ret = -1;
            fprintf('Unable to open "%s"\n', eleName);
            return;
        end
      
        numParticles = fread(f, 1, 'int32');
        numHot = fread(f, 1, 'int32');
        numCold = fread(f, 1, 'int32');
        if numParticles == 0
            break
        end
        if numHot > 0
            numHotStates = numHotStates + 1;
            hotStates(:,numHotStates) = fread(f, 5, 'float');
        end
        if numCold > 0
            numColdStates = numColdStates + 1;
            coldStates(:,numColdStates) = fread(f, 5, 'float');
        end
        fclose(f);
    end
    
    hotSampleNumbers = 1:numHotStates;
    coldSampleNumbers = 1:numColdStates;
    hotStates = hotStates(:, 1:numHotStates);
    coldStates = coldStates(:, 1:numColdStates);
    
    plotStates(hotSampleNumbers, hotStates, 'hot', zStr);
    plotStates(coldSampleNumbers, coldStates, 'cold', zStr);
    
    % Find all windows of type figure, which have an empty FileName attribute.
    allPlots = findall(0, 'Type', 'figure', 'FileName', []);
    % Close.
    delete(allPlots);
    fclose('all');
end

function plotStates(sampleNumbers, states, tempStr, zStr)
figure;
    title('Pos X')
    plot(sampleNumbers, states(1,:));
    ylabel('X');
    title(strcat([tempStr ' ele vz=' zStr]));
    print('-dpng', strcat(tempStr, '_z', zStr, '_x_vs_time'));
    
    figure;
    title('Pos Y')
    plot(sampleNumbers, states(2,:));
    xlabel('Time');
    ylabel('Y');
    title(strcat([tempStr ' ele vz=' zStr]));
    print('-dpng', strcat(tempStr, '_z', zStr, '_y_vs_time'));
        
    figure;
    title('Vel X')
    plot(sampleNumbers, states(3,:));
    xlabel('Time');
    ylabel('VX');
    title(strcat([tempStr ' ele vz=' zStr]));
    print('-dpng', strcat(tempStr, '_z', zStr, '_vx_vs_time'));
            
    figure;
    title('Vel Y')
    plot(sampleNumbers, states(4,:));
    xlabel('Time');
    ylabel('VY');
    title(strcat([tempStr ' ele vz=' zStr]));
    print('-dpng', strcat(tempStr, '_z', zStr, '_vy_vs_time'));
            
    figure;
    title('Vel Z')
    plot(sampleNumbers, states(5,:));
    xlabel('Time');
    ylabel('VZ');
    title(strcat([tempStr ' ele vz=' zStr]));
    print('-dpng', strcat(tempStr, '_z', zStr, '_vz_vs_time'));
end