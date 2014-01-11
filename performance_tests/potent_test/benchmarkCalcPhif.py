#!/bin/env python

import os
import shutil
import string
import subprocess
import sys

from os.path import sep

outBase = '..' + sep + '..' + sep + 'output' + sep + 'calcPhifBenchmark'

def findAveragePotentTime(dataPath):
   avgTime = 0
   f = open(dataPath + sep + 'performance.csv', 'r')
   
   potentTimeSum = 0
   numValues = 0
   for line in f:
      fields = string.split(line, sep=',')
      potentTimeSum = potentTimeSum + float(fields[-3])
      numValues = numValues + 1

   avgTime = potentTimeSum / numValues

   return avgTime

def runPic(width, height):
   exeName = '../../bin/pic'
   width = str(width)
   height = str(height)
   maxTime =  str(4)
   logInterval =  str(maxTime)
   b0 =  str(3)
   sigmaHe =  str(1) 
   sigmaHi =  str(0.5) 
   sigmaCe =  str(100) 
   sigmaCi =  str(100) 
   sigmaHePerp =  str(0.25) 
   sigmaHiPerp =  str(0.125) 
   sigmaCeSecondary =  str(2) 
   percentageSecondary =  str(0.05) 
   outputDir = outBase + sep + 'width_' + str(width) + '_height_' + str(height)

   argList = [ exeName \
      , '-x', width \
      , '-y', height \
      , '-t', maxTime \
      , '-l', logInterval \
      , '--inject-width', width \
      , '--b0', b0 \
      , 'sigma-he', sigmaHe \
      , 'sigma-hi', sigmaHi \
      , 'sigma-ce', sigmaCe \
      , 'sigma-ci', sigmaCi \
      , 'sigma-he-perp', sigmaHePerp \
      , 'sigma-hi-perp', sigmaHiPerp \
      , 'sigma-ce-secondary', sigmaCeSecondary \
      , 'percentage-secondary', percentageSecondary \
      , '-o', outputDir \
      ]
   if sys.platform == 'linux2':
      f = open('/dev/null', 'r')
      subprocess.call(argList, stdout=f)
   else:
      subprocess.call(argList)

   return findAveragePotentTime(outputDir)

def main():
   startWidthPower = 5 # 32
   endWidthPower = 10 # 1024
   heightIncrement = 2500
   endHeight = 30000

   if os.path.exists(outBase):
      shutil.rmtree(outBase)
   os.makedirs(outBase)


   f = open('potentPerformance.csv', 'w')
   first = True
   for widthPower in range(startWidthPower, endWidthPower+1):
      runtimeData = []
      width = 2**widthPower
      for height in range(heightIncrement, endHeight+1, heightIncrement):
         if first:
            f.write('w=' + str(width) + ' h=' + str(height))
            first = False
         else:
            f.write(',w=' + str(width) + ' h=' + str(height))
   for widthPower in range(startWidthPower, endWidthPower+1):
      runtimeData = []
      width = 2**widthPower
      for height in range(heightIncrement, endHeight+1, heightIncrement):
         print 'Benchmarking width=' + str(width) + ' height=' + str(height)
         runtimeData.append(runPic(width, height))
      first = True
      f.write('\n')
      for i in runtimeData:
         if first:
            f.write(str(i))
            first = False
         else:
            f.write(',' + str(i))
   f.close()
   shutil.rmtree(outBase)

   return

main()
