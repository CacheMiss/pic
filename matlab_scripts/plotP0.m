
function ret = plotP0()

   f = fopen('p0.dat', 'rb');

   if f <= 0
      ret = -1;
      fprintf('Unable to open "%s"\n', 'bym.dat');
      return;
   end

   width = fread(f, 1, 'int32');
   p0 = fread(f, width, 'float');

   xValues = 0:width-1;
   figure;
   plot(xValues, p0);
   title('p0 vs X');
   xlabel('X');
   ylabel('p0');
   axis([0 width, min(p0), max(p0)]);
   print('-dpng', 'p0_vs_x');

   fclose(f);
   
end
