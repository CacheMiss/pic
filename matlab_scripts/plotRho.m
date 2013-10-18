function ret = plotRho(index, titleStr, column)

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
   
   window = 64;
   halfWin = window / 2;
   rhoeSize = size(rhoe);
   rhoeSize = rhoeSize(1);
   smoothE = zeros(rhoeSize, 1);
   for i=1 : rhoeSize
       if i < halfWin+1
           smoothE(i) = mean(rhoe(1:i+halfWin));
       elseif i < rhoeSize-halfWin
           smoothE(i) = mean(rhoe(i-halfWin:i+halfWin));
       else
           smoothE(i) = mean(rhoe(i-halfWin:end));
       end
   end
   rhoiSize = size(rhoi);
   rhoiSize = rhoiSize(1);
   smoothI = zeros(rhoiSize, 1);
   for i=1 : rhoiSize
       if i < halfWin+1
           smoothI(i) = mean(rhoi(1:i+halfWin));
       elseif i < rhoiSize-halfWin
           smoothI(i) = mean(rhoi(i-halfWin:i+halfWin));
       else
           smoothI(i) = mean(rhoi(i-halfWin:end));
       end
   end
   
   figure;
   a1 = semilogy(yValues, smoothE, yValues, smoothI);
   legend('rhoe', 'rhoi');
   title(strcat([titleStr ' x=' int2str(column)]));
   xlabel('y');
   ylabel('rho');
   %axis([20 max(yValues) min(min(rhoe(5:end)), min(rhoi(5:end))) max(max(rhoe(5:end)), max(rhoi(5:end)))]);
   %axis([max(yValues)-5 max(yValues) min(min(rhoe(5:end)), min(rhoi(5:end))) max(max(rhoe(5:end)), max(rhoi(5:end)))]);
   %axis([0 max(yValues) min(min(rhoe), min(rhoi)) max(max(rhoe), max(rhoi))]);
   axis([0 max(yValues) 0.001 2]);

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
