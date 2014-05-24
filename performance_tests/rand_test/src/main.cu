////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014, Stephen C. Sewell
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////
#include <curand.h>
#include <fstream>
#include <iostream>

#include "dev_mem.h"
#include "device_stats.h"
#include "precisiontimer.h"

int main()
{
   DeviceStats &device = DeviceStats::getRef();

   const unsigned int NX1 = 512;
   const unsigned int NIJ = 32;
   const int neededRands = NX1 * NIJ * 4 * 6;
   const int maxTime = 80000;
   const int iterationsPerTime = 20;
   DevMem<float> randArray(neededRands);

   PrecisionTimer timer;
   std::ofstream randTimes("randTimes.txt");

   curandGenerator_t randGenerator;
   curandCreateGenerator (&randGenerator, CURAND_RNG_PSEUDO_MTGP32);
   curandSetPseudoRandomGeneratorSeed(randGenerator, 1);

   randTimes << "simTime,timeToRestore(microS)" << std::endl;
   for(int endTime = 0; endTime <= maxTime; endTime += 5000)
   {
      const int numIterations = endTime * iterationsPerTime;
      timer.start();
      for(int i = 0; i < numIterations; i++)
      {
         curandGenerateUniform(randGenerator, randArray.getPtr(), neededRands);

      }
      timer.stop();
      randTimes << endTime << "," << timer.intervalInMicroS() << std::endl;
      std::cout << "sim time:" << endTime << " time: " << timer.intervalInMicroS() << std::endl;
   }

   return 0;
}