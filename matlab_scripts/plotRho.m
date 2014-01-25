function ret = plotRho(index, column)

   rhoName = strcat('rho_', index);
   rhoEName = strcat('rhoe_', index);
   rhoIName = strcat('rhoi_', index);
   
   f = fopen(rhoName, 'rb');

   %rho = readRho(rhoName, column);

   %yValues = 0:size(rho)-1;
   %figure;
   %plot(yValues, rho);
   %title(strcat([titleStr ' x=' int2str(column)]));
   %xlabel('y');
   %ylabel('rho');
   %axis([0 10 min(rho(1:10)) max(rho(1:10))]);
   %axis([20 max(yValues) min(rho(5:end)) max(rho(5:end))]);
   %axis([max(yValues)-5 max(yValues) min(rho(5:end)) max(rho(5:end))]);
   %axis([0 max(yValues) min(rho) max(rho)]);
   
   rhoe = readRho(rhoEName, column);
   rhoi = readRho(rhoIName, column);
   
   yValues = 0:size(rhoe)-1;
   
   %figure;
   %a1 = plot(yValues, rhoe, yValues, rhoi);
   %legend('rhoe', 'rhoi');
   %title(strcat([titleStr ' x=' int2str(column)]));
   %xlabel('y');
   %ylabel('rho');
   %axis([0 10 min(min(rhoe(1:10)), min(rhoi(1:10))) max(max(rhoe(1:10)), max(rhoi(1:10)))]);
   
   %figure;
   %a1 = plot(yValues, rhoe, yValues, rhoi);
   %legend('rhoe', 'rhoi');
   %title(strcat([titleStr ' x=' int2str(column)]));
   %xlabel('y');
   %ylabel('rho');
   %axis([20 max(yValues) min(min(rhoe(5:end)), min(rhoi(5:end))) max(max(rhoe(5:end)), max(rhoi(5:end)))]);
   %axis([max(yValues)-5 max(yValues) min(min(rhoe(5:end)), min(rhoi(5:end))) max(max(rhoe(5:end)), max(rhoi(5:end)))]);
   %axis([0 max(yValues) min(min(rhoe), min(rhoi)) max(max(rhoe), max(rhoi))]);
   
   smoothE = smoothLine(rhoe, 64);
   smoothI = smoothLine(rhoi, 64);
   %smoothE = rhoe;
   %smoothI = rhoi;
   
   figure;
   a1 = semilogy(yValues, smoothE, yValues, smoothI);
   fields = strsplit(rhoName, '_');
   legend('rhoe', 'rhoi');
   title(strcat([fields{1} ' ' fields{2} ' x=' int2str(column)]));
   xlabel('y');
   ylabel('rho');
   %axis([20 max(yValues) min(min(rhoe(5:end)), min(rhoi(5:end))) max(max(rhoe(5:end)), max(rhoi(5:end)))]);
   %axis([max(yValues)-5 max(yValues) min(min(rhoe(5:end)), min(rhoi(5:end))) max(max(rhoe(5:end)), max(rhoi(5:end)))]);
   %axis([0 max(yValues) min(min(rhoe), min(rhoi)) max(max(rhoe), max(rhoi))]);
   axis([0 max(yValues) 0.001 2]);
   %axis([0 20 0.001 2]);
   print('-dpng', rhoName);

   fclose(f);

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
