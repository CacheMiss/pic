////////////////////////////////////////////////////////////////////////////////
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
// 
//   http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
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
