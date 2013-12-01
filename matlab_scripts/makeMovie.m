function makeMovie(baseName, startIdx, endIdx, stepSize, frameRate)
    vObject = VideoWriter(strcat(baseName, '.avi'));
    vObject.FrameRate = frameRate;
    open(vObject);
    for i=startIdx:stepSize:endIdx
        idxStr = sprintf('%06d', i);
        fileName = strcat(baseName, '_', idxStr, '.png'); 
        frame = imread(fileName);
        writeVideo(vObject, frame);
    end
    close(vObject)
end