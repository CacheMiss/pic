#include <curand.h>
#include <fstream>
#include <iostream>
#include <thrust/sort.h>

#include "dev_mem.h"
#include "device_stats.h"
#include "precisiontimer.h"

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
   if(lhs.pos.y < rhs.pos.y)
   {
      return true;
   }
   else if(lhs.pos.y == rhs.pos.y)
   {
      if(lhs.pos.x < rhs.pos.x)
      {
         return true;
      }
   }
   else
   {
      return false;
   }
   return false;
}

__global__
void initParticles(float2* pos, float *randArray, unsigned int numParticles)
{
   unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;
   if(threadX < numParticles)
   {
      pos[threadX].x = NX1 * randArray[threadX];
      pos[threadX].y = NY1 * randArray[numParticles+threadX];
   }
}

__global__
void initParticles(Particle_t* particle, float *randArray, unsigned int numParticles)
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

void timeSorts(const unsigned int numParticles, float& timeBin, float& timeFull,
               cudaEvent_t& eventBeg, cudaEvent_t& eventEnd)
{
   const int neededRands = numParticles * 2;
   DevMem<float> randArray(neededRands);

   PrecisionTimer timer;

   curandGenerator_t randGenerator;
   curandCreateGenerator (&randGenerator, CURAND_RNG_PSEUDO_MTGP32);
   curandSetPseudoRandomGeneratorSeed(randGenerator, 1);
   curandGenerateUniform(randGenerator, randArray.getPtr(), neededRands);
   curandDestroyGenerator(randGenerator);

   DevMem<float2> pos(numParticles);
   DevMem<unsigned int> bins(numParticles);

   const unsigned int threadsPerBlock = 512;
   const unsigned int numBlocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
   initParticles<<<numBlocks, threadsPerBlock>>>(pos.getPtr(), randArray.getPtr(), numParticles);
   checkCuda(cudaGetLastError());
   binParticles<<<numBlocks, threadsPerBlock>>>(pos.getPtr(), bins.getPtr(), numParticles);
   checkCuda(cudaGetLastError());
   checkCuda(cudaDeviceSynchronize());

   checkCuda(cudaEventRecord(eventBeg));
   thrust::sort_by_key(bins.getThrustPtr(), bins.getThrustPtr()+numParticles, pos.getThrustPtr());
   checkCuda(cudaEventRecord(eventEnd));
   checkCuda(cudaDeviceSynchronize());

   checkCuda(cudaEventElapsedTime(&timeBin, eventBeg, eventEnd));

   pos.freeMem();
   bins.freeMem();

   DevMem<Particle_t> particles(numParticles);
   initParticles<<<numBlocks, threadsPerBlock>>>(particles.getPtr(), randArray.getPtr(), numParticles);
   checkCuda(cudaDeviceSynchronize());

   checkCuda(cudaEventRecord(eventBeg));
   thrust::sort(particles.getThrustPtr(), particles.getThrustPtr()+numParticles);
   checkCuda(cudaEventRecord(eventEnd));
   checkCuda(cudaDeviceSynchronize());

   checkCuda(cudaEventElapsedTime(&timeFull, eventBeg, eventEnd));
}

int main()
{
   DeviceStats &device = DeviceStats::getRef();

   const int maxParticles = 5000000;

   cudaEvent_t eventBeg;
   cudaEvent_t eventEnd;

   checkCuda(cudaEventCreate(&eventBeg));
   checkCuda(cudaEventCreate(&eventEnd));

   float timeBin;
   float timeFull;

   std::ofstream sortTimes("sortTimes.txt");
   sortTimes << "numParticles,binSortTime(ms),fullSortTime(ms)" << std::endl;

   for(int i = 100000; i <= maxParticles; i+=100000)
   {
      timeSorts(i, timeBin, timeFull, eventBeg, eventEnd);
      std::cout << i << " particles; binSort: " << timeBin << "ms fullSort: " << timeFull << "ms" << std::endl;
      sortTimes << i << "," << timeBin << "," << timeFull << std::endl;
   }

   checkCuda(cudaEventDestroy(eventBeg));
   checkCuda(cudaEventDestroy(eventEnd));

   return 0;
}