function ret = plotRho(index, titleStr, column)

   rhoName = strcat('rho_', index);
   rhoEName = strcat('rhoe_', index);
   rhoIName = strcat('rhoi_', index);
   
   f = fopen(rhoName, 'rb');

   rho = readRho(rhoName, column);

   yValues = 0:size(rho)-1;
   figure;
   plot(yValues, rho);
   title(strcat([titleStr ' x=' int2str(column)]));
   xlabel('y');
   ylabel('rho');
   %axis([2800 max(yValues) -1.5 1.5]);
   axis([0 max(yValues) min(rho) max(rho)]);
   
   rhoe = readRho(rhoEName, column);
   rhoi = readRho(rhoIName, column);
   
   yValues = 0:size(rho)-1;
   figure;
   a1 = plot(yValues, rhoe, yValues, rhoi);
   legend('rhoe', 'rhoi');
   title(strcat([titleStr ' x=' int2str(column)]));
   xlabel('y');
   ylabel('rho');
   %axis([2800 max(yValues) 0 2]);
   axis([0 max(yValues) min(min(rhoe), min(rhoi)) max(max(rhoe), max(rhoi))]);

   fclose(f);

end

function r = readRho(fileName, column)
    f = fopen(fileName, 'rb');

    if f <= 0
        fprintf('Unable to open "%s"\n', fileName);
        return;
    end

    numRows = fread(f, 1, 'int32');
    numColumns = fread(f, 1, 'int32');
    columnOrder = fread(f, 1, 'int32');
    rho = fread(f, [numRows,numColumns], 'float');
    r = rho(:,column);
    
end
