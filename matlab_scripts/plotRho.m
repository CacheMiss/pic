%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2014, Stephen C. Sewell
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
% 1. Redistributions of source code must retain the above copyright notice, this
%    list of conditions and the following disclaimer.
% 
% 2. Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the documentation
%    and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
% ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
