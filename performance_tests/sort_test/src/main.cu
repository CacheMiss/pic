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
#include <thrust/sort.h>

#include "cpu_only.h"
#include "dev_mem.h"
#include "device_stats.h"
#include "precisiontimer.h"
#include "sort_thread.h"

#define NX1 512
#define NY1 10000

struct Particle_t
{
   float2 pos;
   float3 vel;
};

__host__ __device__
bool operator<(const Particle_t& lhs, const Particle_t& rhs)
{
   unsigned int lhsX = static_cast<unsigned int>(lhs.pos.x);
   unsigned int lhsY = static_cast<unsigned int>(lhs.pos.y);
   unsigned int rhsX = static_cast<unsigned int>(rhs.pos.x);
   unsigned int rhsY = static_cast<unsigned int>(rhs.pos.y);
   bool ret = false;
   if(lhsY < rhsY)
   {
      ret = true;
   }
   else if(lhsY == rhsY)
   {
      if(lhsX < rhsX)
      {
         ret = true;
      }
   }
   return ret;
}

__global__
void initParticles(float2* pos, const float *randArray, unsigned int numParticles)
{
   unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;
   if(threadX < numParticles)
   {
      pos[threadX].x = NX1 * randArray[threadX];
      pos[threadX].y = NY1 * randArray[numParticles+threadX];
   }
}

__global__
void initParticles(Particle_t* particle, const float *randArray, unsigned int numParticles)
{
   unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;
   if(threadX < numParticles)
   {
      particle[threadX].pos.x = NX1 * randArray[threadX];
      particle[threadX].pos.y = NY1 * randArray[numParticles+threadX];
   }
}

__global__
void binParticles(const float2* pos, unsigned int* bins, unsigned int numParticles)
{
   unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;
   if(threadX < numParticles)
   {
      bins[threadX] = NX1 * pos[threadX].y + pos[threadX].x;
   }
}

void restorePositions(DevMem<float2> &pos, const DevMem<float> randArray, unsigned int numParticles)
{
   pos.resize(numParticles);
   const unsigned int threadsPerBlock = 512;
   const unsigned int numBlocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
   initParticles<<<numBlocks, threadsPerBlock>>>(pos.getPtr(), randArray.getPtr(), numParticles);
   checkCuda(cudaGetLastError());
   checkCuda(cudaDeviceSynchronize());
}

void restoreBins(DevMem<unsigned int> &bins, const DevMem<float2> &pos, unsigned int numParticles)
{
   bins.resize(numParticles);
   const unsigned int threadsPerBlock = 512;
   const unsigned int numBlocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
   checkCuda(cudaGetLastError());
   binParticles<<<numBlocks, threadsPerBlock>>>(pos.getPtr(), bins.getPtr(), numParticles);
   checkCuda(cudaDeviceSynchronize());
}

void timeSorts(const unsigned int numParticles, 
               double& gpuIntBin, double& gpuCustomOp, 
               double& timeSortClass, double& gpuFullSortInt,
               double& timeDevToHost, double &timeHostToDev,
               double& timeTbbSort)
{
   PrecisionTimer t;
   const int neededRands = numParticles * 2;
   DevMem<float> randArray(neededRands);

   PrecisionTimer timer;

   curandGenerator_t randGenerator;
   curandCreateGenerator (&randGenerator, CURAND_RNG_PSEUDO_MTGP32);
   curandSetPseudoRandomGeneratorSeed(randGenerator, 1);
   curandGenerateUniform(randGenerator, randArray.getPtr(), neededRands);
   curandDestroyGenerator(randGenerator);

   /////////////////////////////////////////////////////////////////////////////
   // Memcpy To Host Timing
   {
      DevMem<float2> pos(numParticles);
      DevMem<float3> fakeVelocities(numParticles);
      restorePositions(pos, randArray, numParticles);
      checkCuda(cudaDeviceSynchronize());
      HostMem<SortThread::Particle> cpuParticles(numParticles);
      DevStream readStream;
      t.start();
      checkCuda(cudaMemcpy2DAsync(&cpuParticles[0].pos, sizeof(SortThread::Particle),
                pos.getPtr(), sizeof(float2),
                sizeof(float2),
                numParticles,
                cudaMemcpyDeviceToHost, 
                *readStream));
      checkCuda(cudaMemcpy2DAsync(&cpuParticles[0].vel, sizeof(SortThread::Particle),
                fakeVelocities.getPtr(), sizeof(float3),
                sizeof(float3),
                numParticles,
                cudaMemcpyDeviceToHost, 
                *readStream));
      readStream.synchronize();
      t.stop();
      timeDevToHost = t.intervalInNanoS() / 1000000;
   /////////////////////////////////////////////////////////////////////////////
   // TBB Sort Timing
      unsigned int numOutOfOrder = 0;
      for(std::size_t i = 1; i < cpuParticles.size(); i++)
      {
         if(cpuParticles[i-1] < cpuParticles[i])
         {
            numOutOfOrder++;
            break;
         }
      }
      assert(numOutOfOrder > 0);
      checkCuda(cudaDeviceSynchronize());
      t.start();
      cpuSort(cpuParticles);
      t.stop();
      checkCuda(cudaDeviceSynchronize());
      timeTbbSort = t.intervalInNanoS() / 1000000;
      assert(timeTbbSort == t.intervalInMilliS());
      for(std::size_t i = 1; i < cpuParticles.size(); i++)
      {
         assert(cpuParticles[i-1] < cpuParticles[i]);
      }
   /////////////////////////////////////////////////////////////////////////////
   // Memcpy To Device Timing
      DevStream writeStream;

      t.start();
      checkCuda(cudaMemcpy2DAsync(pos.getPtr(), sizeof(float2),
                &cpuParticles[0].pos, sizeof(SortThread::Particle),
                sizeof(float2),
                numParticles,
                cudaMemcpyHostToDevice, 
                *writeStream));
      checkCuda(cudaMemcpy2DAsync(fakeVelocities.getPtr(), sizeof(float3),
                &cpuParticles[0].vel, sizeof(SortThread::Particle),
                sizeof(float3),
                numParticles,
                cudaMemcpyHostToDevice, 
                *writeStream));
      writeStream.synchronize();
      t.stop();
      timeHostToDev = t.intervalInNanoS() / 1000000;
   }
   /////////////////////////////////////////////////////////////////////////////
   // Bin Sort Timing
   {
      DevMem<float2> pos(numParticles);
      DevMem<unsigned int> bins(numParticles);
      restorePositions(pos, randArray, numParticles);
      restoreBins(bins, pos, numParticles);

      t.start();
      thrust::sort_by_key(bins.getThrustPtr(), bins.getThrustPtr()+numParticles, pos.getThrustPtr());
      checkCuda(cudaDeviceSynchronize());
      t.stop();
      gpuIntBin = t.intervalInNanoS() / 1000000;
   }
   /////////////////////////////////////////////////////////////////////////////
   // Custom operator< timing
   {
      DevMem<Particle_t> particles(numParticles);
      const unsigned int threadsPerBlock = 512;
      const unsigned int numBlocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
      initParticles<<<numBlocks, threadsPerBlock>>>(particles.getPtr(), randArray.getPtr(), numParticles);
      checkCuda(cudaDeviceSynchronize());

      t.start();
      thrust::sort(particles.getThrustPtr(), particles.getThrustPtr()+numParticles);
      checkCuda(cudaDeviceSynchronize());
      t.stop();
      gpuCustomOp = t.intervalInNanoS() / 1000000;
   }
   /////////////////////////////////////////////////////////////////////////////
   // Sort Thread Class Timing
   {
      SortThread sortThread(NY1 + 1000);
      sortThread.setNumSortThreads(1);
      sortThread.disableParticleElimination();
      sortThread.run();

      DevMem<float2> pos(numParticles);
      restorePositions(pos, randArray, numParticles);
      DevMem<float3> fakeVelocities(numParticles);

      t.start();
      sortThread.sortAsync(pos, fakeVelocities, numParticles);
      sortThread.waitForSort(pos, fakeVelocities);
      t.stop();
      timeSortClass = t.intervalInNanoS() / 1000000;

      sortThread.join();
   }
   /////////////////////////////////////////////////////////////////////////////
   // Double bin sort timing
   // I need another copy of this; one to sort pos and one to sort vel
   {
      DevMem<float2> pos(numParticles);
      restorePositions(pos, randArray, numParticles);
      DevMem<float3> fakeVelocities(numParticles);
      DevMem<unsigned int> bins(numParticles);
      restoreBins(bins, pos, numParticles);
      DevMem<unsigned int> bins2 = bins;

      t.start();
      thrust::sort_by_key(bins.getThrustPtr(), bins.getThrustPtr()+numParticles, pos.getThrustPtr());
      thrust::sort_by_key(bins2.getThrustPtr(), bins2.getThrustPtr()+numParticles, fakeVelocities.getThrustPtr());
      checkCuda(cudaDeviceSynchronize());
      t.stop();
      gpuFullSortInt = t.intervalInNanoS() / 1000000;
   }
}

int main()
{
   DeviceStats &device = DeviceStats::getRef();

   const int maxParticles = 5000000;

   double timeBin;
   double timeFull;
   double cpuSort;
   double timeFullInt;
   double memcpyToHost;
   double memcpyToDevice;
   double timeTbbSort;

   std::ofstream sortTimes("sortTimes.txt");
   sortTimes << "numParticles,binSortTime(ms),fullSortTime(ms),fullSortWithIntTime(ms)" 
      << ",memcpyToHost,timTbbSort,memcpyToDevice" << std::endl;

   for(int i = 100000; i <= maxParticles; i+=100000)
   {
      timeSorts(i, timeBin, timeFull, 
                cpuSort, timeFullInt, 
                memcpyToHost, memcpyToDevice,
                timeTbbSort);
      std::cout << i << " particles; binSort: " << timeBin << "ms fullSort: " 
         << timeFull << "ms cpuSort: " << cpuSort << " ms gpuFullSortIntKeys: "
         << timeFullInt << " ms" << std::endl;
      sortTimes << i << "," << timeBin << "," << timeFull << "," << cpuSort 
         << "," << timeFullInt << "," << memcpyToHost << "," << timeTbbSort
         << "," << memcpyToDevice << std::endl;
   }

   return 0;
}