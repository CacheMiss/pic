#include "movep.h"

#include "commandline_options.h"
#include "dev_mem_reuse.h"
#include "device_stats.h"
#include "device_utils.h"
#include "global_variables.h"
#include "logging_thread.h"
#include "logging_types.h"
#include "particle_allocator.h"
#include "pic_utils.h"

//texture<float, 1, cudaReadModeElementType > exTex;
//texture<float, 1, cudaReadModeElementType > eyTex;

__global__
void checkBoundaryConditions(float2 location[], int numParticles, int height, int width, 
                             bool *success)
{
   int threadId = blockDim.x * blockIdx.x + threadIdx.x;
   if(threadId >= numParticles)
   {
      return;
   }
   if(location[threadId].x < 0 || location[threadId].x > width ||
      location[threadId].y < 0 || location[threadId].y > height)
   {
      *success = false;
   }
}

__device__
void writeParticlesBack(float global[], volatile float shared[], 
                        const unsigned int startingIndex,
                        const unsigned int numParticles,
                        const int warpId, const int threadInWarp)
{
   unsigned int storeIndex = warpId * warpSize * 5 + threadInWarp;

   #pragma unroll 5
   for(int i = 0; i < 5; ++i)
   {
      if(startingIndex + storeIndex < numParticles * 5)
      {
         global[startingIndex + storeIndex] = 
            shared[storeIndex];
      }
      storeIndex += warpSize;
   }
}

__device__
void deconflictBanks(volatile float conflicted[], 
                     volatile float fixed[],
                     const int warpId, const int threadInWarp)
{
   unsigned int end = (warpId+1) * warpSize * 5;

   #pragma unroll 5
   for(int currentIndex = warpId * warpSize * 5 + threadInWarp;
       currentIndex < end;
       currentIndex += warpSize)
   {
      fixed[blockDim.x * (currentIndex % 5) + currentIndex / 5] =
         conflicted[currentIndex];
   }
}

__device__
void reconflictBanks(volatile float conflicted[], 
                     volatile const float fixed[],
                     const int warpId, const int threadInWarp)
{
   unsigned int end = (warpId+1) * warpSize * 5;

   #pragma unroll 5
   for(int currentIndex = warpId * warpSize * 5 + threadInWarp;
       currentIndex < end;
       currentIndex += warpSize)
   {
      conflicted[currentIndex] = 
         fixed[blockDim.x * (currentIndex % 5) + currentIndex / 5];
   }
}

//******************************************************************************
// Function: moveParticles
// Code Type: Device
// Block Structure: 1 thread per particle
// Purpose: Apply equations of motion to all particles in grid; apply periodic
//          boundary conditions; and flag particles for removal which fall
//          outside of the top and bottom of the grid
// Parameters:
// -------------------
// particles - The array of particles (This array has 5 elements per part)
// ex - The electrical field in x
// ey - The electircal field in y
// numParticles - The number of particles in the particles array
// mass - The mass of the type of particle being moved
// oobIdx - The number of particles which are lower than NY or higher than
//          DY * (NY - 1)
// oobArry - Contains oobIdx entries of particles which need to be removed
// NX1 - The width of the grid (A power of 2)
// NY1 - The height of the grid (A power of 2)
// MAX_OOB_BUFFER - The max number of particles that can be eliminated
//******************************************************************************
__global__ 
void moveParticles(float2 d_partLoc[], float3 d_partVel[],
                   const float ex[], const float ey[],
                   const unsigned int numParticles,
                   const float mass,
                   const unsigned int NX1, const unsigned int NY1)
{
   const unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;

   float2 pLoc;
   float3 pVel;

   pLoc = d_partLoc[threadX];
   pVel = d_partVel[threadX];

   // A local 3 x 3 array
   //volatile float *a = reinterpret_cast<volatile float*>(sharedWork) + threadIdx.x;
   float a[9];
   // A local array of size 3
   //volatile float *b = a + blockDim.x * 9;
   float b[3];

   float dela;
   float delb;
   float a1;
   float a2;
   float a3;
   float a4;
   float local_expf;
   float local_eypf;
   float c1;
   float c2;
   float c3;
   float c4;
   float det;
   float det1;
   float det2;
   float det3;
   int nii;
   int nj;
   int nj1;

   if(threadX < numParticles)
   {
      if (pLoc.x <= D_DX * NX1 && 
          pLoc.x >= 0)
      {
         nii=(int) (pLoc.y/D_DY);  // Y Position of pBuficle
         nj=(int) ((pLoc.x)/D_DX); // X Position of pBuficle
         if(nj == NX1-1)
         {
            nj1 = 0;
         }
         else
         {
            nj1 = nj + 1;
         }
         dela=(float)(fabs((float) (pLoc.x-(D_DX * nj))));
         delb=(float)(fabs((float) (pLoc.y-(D_DY * nii))));
         a1=dela*delb;
         a2=D_DX * delb - a1;
         a3=D_DY * dela - a1;
         a4=D_DX * D_DY-(a1+a2+a3);
         local_expf=(a1*ex[NX1 * (nii+1) + nj1] + 
            a2*ex[NX1 * (nii+1) + nj] +
            a3*ex[NX1 * nii + nj1] + 
            a4*ex[NX1 * nii + nj])/D_TOTA;
         local_eypf=(a1*ey[NX1 * (nii+1) + nj1] +
            a2*ey[NX1 * (nii+1) + nj] +
            a3*ey[NX1 * nii + nj1]+
            a4*ey[NX1 * nii + nj])/D_TOTA;
         c1= D_DELT * mass;
         c2=0.5f;
         c3=c1*0.5f;
         c4=c1*c2;
         a[0] = 1.0f;         // a[0, 0]
         a[1] = -c3*D_BZM;    // a[0, 1]
         a[2] = c3*D_BYM;     // a[0, 2]
         a[3] = c3*D_BZM;     // a[1, 0]
         a[4] = 1.0f;         // a[1, 1]
         a[5] = -c3*D_BXM;    // a[1, 2]
         a[6] = -c3*D_BYM;    // a[2, 0]
         a[7] = c3*D_BXM;     // a[2, 1]
         a[8] = 1.0f;         // a[2, 2]
         b[0]=pVel.x + pVel.y*c4*D_BZM - 
            pVel.z*c4*D_BYM + c1*local_expf;
         b[1]=pVel.y + pVel.z*c4*D_BXM - 
            pVel.x*c4*D_BZM + c1*local_eypf;
         b[2]=pVel.z + pVel.x*c4*D_BYM - 
            pVel.y*c4*D_BXM;
         det=a[0]*(a[4]*a[8] - 
            a[5]*a[7])-
            a[1]*(a[3]*a[8] - 
            a[5]*a[6])+
            a[2]*(a[3]*a[7] - 
            a[4]*a[6]);
         det1=b[0]*(a[4]*a[8] - 
            a[5]*a[7]) -
            a[1]*(b[1]*a[8] - 
            b[2]*a[5]) +
            a[2]*(b[1]*a[7] - 
            b[2]*a[4]);
         det2=a[0]*(b[1]*a[8] - 
            b[2]*a[5]) -
            b[0]*(a[3]*a[8] - 
            a[5]*a[6]) +
            a[2]*(b[2]*a[3] - 
            b[1]*a[6]);
         det3=a[0]*(a[4]*b[2] - 
            b[1]*a[7]) -
            a[1]*(a[3]*b[2] - 
            b[1]*a[6])+
            b[0]*(a[3]*a[7] - 
            a[4]*a[6]);
         // for 2^126 <= y <= 2^128, __fdividef(x,y) delivers a result of zero,
         pVel.x = __fdividef(det1, det);
         pVel.y = __fdividef(det2, det);
         pVel.z = __fdividef(det3, det);
         pLoc.x = pLoc.x + pVel.x*D_DELT;
         pLoc.y = pLoc.y + pVel.y*D_DELT;

      }
      // Enforce periodic boundary condition on x
      if(pLoc.x > D_DX * NX1)
      {
         pLoc.x= pLoc.x - (D_DX * NX1);
      }
      else if (pLoc.x <  0)
      { 
         pLoc.x= pLoc.x + (D_DX * NX1);
      }
   }

   //d_partLoc[threadX] = pLoc[threadIdx.x];
   {
      float2 tmpLoc;
      tmpLoc.x = pLoc.x;
      tmpLoc.y = pLoc.y;
      d_partLoc[threadX] = tmpLoc;
   }
   d_partVel[threadX] = pVel;

}

__global__ 
void findOobParticles(float2 d_partLoc[], 
                      const unsigned int numParticles,
                      unsigned int *oobIdx, unsigned int oobArry[],
                      const unsigned int NX1, const unsigned int NY1,
                      const unsigned int MAX_OOB_BUFFER)
{
   const int warpId = threadIdx.x / warpSize;
   const int threadInWarp = threadIdx.x % warpSize;
   const unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;

   // Beg Shared Memory
#if (__CUDA_ARCH__ >= 200)
   __shared__ int blockOobStartIdx;
   if(threadIdx.x == 0) blockOobStartIdx = 0;
   __syncthreads();
#endif
   extern __shared__ char sharedBeg[];
   volatile float2 *pLoc = (float2*)sharedBeg;
   //volatile unsigned int *sharedWork = 
   //   (volatile unsigned int*)(sharedBeg + ((sizeof(float2) + sizeof(float3)) * blockDim.x));
   //volatile unsigned int *warpBegIdx = 
   //   const_cast<unsigned int*>(&sharedWork[blockDim.x * 16] + warpId);
#if (__CUDA_ARCH__ < 200)
   volatile unsigned int *sharedWork = 
      (volatile unsigned int*)(sharedBeg + (sizeof(float2) * blockDim.x));
   volatile unsigned int *warpBegIdx = sharedWork + blockDim.x + warpId;
#else
   volatile unsigned int *warpOobStartIdx = 
      (volatile unsigned int*)(sharedBeg + sizeof(float2) * blockDim.x) + warpId;
#endif
   // End Shared Memory

   // Load the particles into shared memory
   {
      float2 tmpLoc = d_partLoc[threadX];
      pLoc[threadIdx.x].x = tmpLoc.x;
      pLoc[threadIdx.x].y = tmpLoc.y;
   }

   if(threadX < numParticles)
   {
#if (__CUDA_ARCH__ < 200)
      // Get a block of shared memory for this warp
      sharedWork += warpId * warpSize;

      // The first thread in the warp will find how many particles within
      // the warp have left the boundary condition on the top or bottom
      if(threadInWarp == 0)
      {
         // Each warp will reserve between 0 and warpSize elements out of the
         // oobArry for out of bounds particles. numOobInWarp is used to help
         // the threads in a warp coordinate which locations in the reserved
         // space they can write to
         unsigned int numOobInWarp = 0;
         // For all particles in warp
         for(unsigned int i = 0; (i < warpSize) && 
             ((blockDim.x*blockIdx.x + warpSize*warpId + i) < numParticles); i++)
         {
            // If this particle is out of bounds
            if(pLoc[warpId * warpSize + i].y > D_DY * (NY1-1) ||
               pLoc[warpId * warpSize + i].y < D_DY)
            {
               // The thread associated with this particle will use this value
               // as an offset combined with the warpBegIdx to write into the
               // out of bounds array (oobArry) so that this particle can be
               // removed
               sharedWork[i] = numOobInWarp;
               ++numOobInWarp;
            }
         }
         if(numOobInWarp > 0)
         {
            *warpBegIdx = atomicAdd(oobIdx, numOobInWarp);
         }
      }
      if(pLoc[threadIdx.x].y > D_DY * (NY1-1) ||
         pLoc[threadIdx.x].y < D_DY)
      {
         int idx = *warpBegIdx + sharedWork[threadInWarp];
         if(idx < MAX_OOB_BUFFER)
         {
            oobArry[idx] = threadX;
         }
      }
#else
      const unsigned int oobBallot = 
         __ballot(pLoc[threadIdx.x].y > D_DY * (NY1-1) ||
                  pLoc[threadIdx.x].y < D_DY);
      unsigned int threadBit = 1 << threadInWarp;
      // DEBUG
      if(oobBallot & threadBit)
      {
         pLoc[threadIdx.x].x = 10000;
      }
      // END DEBUG
      // If this thread in the warp is oob, this holds the offset from warpOobStartIdx
      // that it can write to
      unsigned int oobIdxWarpOffset; 
      if(oobBallot != 0)
      {
         if(threadInWarp == 0)
         {
            *warpOobStartIdx = 0;
         }
         if(threadBit & oobBallot)
         {
            // Get an index to write this out of bounds particle to
            oobIdxWarpOffset = atomicAdd((unsigned int*)warpOobStartIdx, 1);
         }
         if(threadInWarp == 0)
         {
            // Aggregate all of the warp oob counts to the block level
            // The result I get back is the offset into this blocks oob space
            // that this warp gets to write to
            *warpOobStartIdx = atomicAdd(&blockOobStartIdx, *warpOobStartIdx);
         }
      }
      // Need a gap in the if to all sycthreads
      // Ensure all warps in the block have a chance to write their oob count out
      __syncthreads();
      if(threadIdx.x == 0)
      {
         // Reserve a group of indices for this whole block
         blockOobStartIdx = atomicAdd(oobIdx, blockOobStartIdx);
      }
      // Make sure that the block gets its space reserved before continuing
      __syncthreads();
      if(oobBallot != 0)
      {
         // Write out which particles are oob
         if(threadBit & oobBallot)
         {
            // I'm trusting that the cache here will grab blockOobStartIdx
            oobArry[blockOobStartIdx + *warpOobStartIdx + oobIdxWarpOffset] = threadX;
         }
      }
#endif
   }
}

//******************************************************************************
// Function: findGoodIndices
// Code Type: Device
// Block Structure: One thread for each particle that needs to be removed
// Purpose: If there are 1000 particles and 100 need to be removed, this kernel
//          finds which indices in the last 100 elements of the array don't need
//          to be removed. These particles are later moved lower into the array
// Parameters:
// -------------------
// particles - The array of particles (This array has 5 elements per part)
// numParticles - The number of particles in the particles array
// moveCandIdx - The number of particles found which are NOT out of bounds
// moveCandidates - The array contaning the indicies of the particles that need
//                  to be pulled down lower into the particle array
// begIndex - For 100 particles where 10 needed to be removed, this would be 90
// numToCull - The number of particles that are going to be removed
// NY1 - The height of the grid (A power of 2)
//******************************************************************************
__global__
void findGoodIndicies(const float2 d_partLoc[], unsigned int numParticles,
   unsigned int *moveCandIdx, unsigned int moveCandidates[],
   const unsigned int begIndex, unsigned int numToCull,
   const unsigned int NY1)
{
   extern __shared__ char beg[];
   float2 *pLoc = (float2*)beg;

   int gStartIndex = begIndex + blockDim.x * blockIdx.x;

   unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int pIndex = begIndex + threadX;

   // Load the particles into shared memory
   pLoc[threadIdx.x] = d_partLoc[gStartIndex + threadIdx.x];

   // Make sure all particles in block are loaded
   __syncthreads();

   // If the particle being considered is high enough that it needs to be
   // shiftded further down in the particles array
   if(pIndex >= numParticles - numToCull && pIndex < numParticles)
   {
      if(pLoc[threadIdx.x].y >= D_DY &&
         pLoc[threadIdx.x].y <= D_DY * (NY1-1))
      {
         int idx = atomicAdd(moveCandIdx, 1);
         moveCandidates[idx] = pIndex;
      }
   }
}

__global__
void killParticles(float2 partLoc[], float3 partVel[], const unsigned int oobArry[],
    const unsigned int moveCandidates[], unsigned int numMoves)
{
   unsigned int threadX = blockIdx.x * blockDim.x + threadIdx.x;

   if(threadX < numMoves)
   {
      unsigned int rcvIndex = oobArry[threadX];
      unsigned int origIndex = moveCandidates[threadX];
      partLoc[rcvIndex] = partLoc[origIndex];
      partVel[rcvIndex] = partVel[origIndex];
   }
}

/**************************************************************
       movep Move particle routine
**************************************************************/
void movep(DevMem<float2> &partLoc, DevMem<float3> &partVel,
           unsigned int &numParticles, float mass,
           const DevMemF &ex, const DevMemF &ey,
           cudaStream_t &stream)
{
   int numThreads;
   dim3 blockSize;
   dim3 numBlocks;
   unsigned int sharedMemoryBytes;

   DeviceStats &dev(DeviceStats::getRef());
   SimulationState &simState(SimulationState::getRef());

   unsigned int maxOobBuffer = simState.maxNumParticles/10;
   DevMem<unsigned int, DevMemReuse> dev_oobIdx;
   assert(dev_oobIdx.getPtr() != NULL);
   dev_oobIdx.zeroMem();
   DevMem<unsigned int, DevMemReuse> dev_moveCandIdx;
   dev_moveCandIdx.zeroMem();
   DevMem<unsigned int, ParticleAllocator> dev_oobArry(maxOobBuffer);
   DevMem<unsigned int, ParticleAllocator> dev_moveCandidates(maxOobBuffer);
   HostMem<unsigned int> oobIdx(1);
   HostMem<unsigned int> moveCandIdx(1);

   // DEBUG
   //{
   //   LoggingThread &logger(LoggingThread::getRef());
   //   cudaThreadSynchronize();
   //   logger.pushLogItem(
   //      new LogParticlesAscii(998, partLoc, partVel,
   //      partLoc, partVel,
   //      partLoc, partVel,
   //      partLoc, partVel,
   //      numParticles, numParticles,
   //      numParticles, numParticles));
   //   logger.flush();
   //}
   // END DEBUG

   numThreads = dev.maxThreadsPerBlock / 4;
   resizeDim3(blockSize, numThreads);
   resizeDim3(numBlocks, calcNumBlocks(numThreads, numParticles));
   cudaStreamSynchronize(stream);
   checkForCudaError("Before moveParticles");
   moveParticles<<<numBlocks, blockSize, 0, stream>>>(
      partLoc.getPtr(), partVel.getPtr(),
      ex.getPtr(), ey.getPtr(),
      numParticles, mass, NX1, NY1);
   checkForCudaError("moveParticles");

   numThreads = dev.maxThreadsPerBlock / 4;
   resizeDim3(blockSize, numThreads);
   resizeDim3(numBlocks, calcNumBlocks(numThreads, numParticles));
   //sharedMemoryBytes = numThreads * (sizeof(float2) + sizeof(float3)) +
   //   numThreads * 16 * sizeof(float) + numThreads/dev.warpSize * sizeof(unsigned int);
   if(dev.major < 2)
   {
      sharedMemoryBytes = numThreads * sizeof(float2) +
         numThreads * sizeof(unsigned int) + sizeof(unsigned int) * numThreads/dev.warpSize;
   }
   else
   {
      sharedMemoryBytes = numThreads * sizeof(float2) + 
         sizeof(unsigned int) * numThreads/dev.warpSize;
   }
   cudaStreamSynchronize(stream);
   checkForCudaError("Before findOobParticles");
   findOobParticles<<<numBlocks, blockSize, sharedMemoryBytes, stream>>>(
      partLoc.getPtr(), 
      numParticles, dev_oobIdx.getPtr(), dev_oobArry.getPtr(),
      NX1, NY1, maxOobBuffer);
   checkForCudaError("moveParticles");

   // DEBUG
   //{
   //   LoggingThread &logger(LoggingThread::getRef());
   //   cudaThreadSynchronize();
   //   logger.pushLogItem(
   //      new LogParticlesAscii(999, partLoc, partVel,
   //      partLoc, partVel,
   //      partLoc, partVel,
   //      partLoc, partVel,
   //      numParticles, numParticles,
   //      numParticles, numParticles));
   //   logger.flush();
   //}
   // END DEBUG

   cudaStreamSynchronize(stream);
   checkForCudaError("Before copy oobIdx to host");
   // Get the number of particles that are outside of the y bounds
   oobIdx = dev_oobIdx;
   if(numParticles == oobIdx[0])
   {
      printf("WARNING: %d of %d particles eliminated\n", oobIdx[0], numParticles);
      numParticles = 0;
      return;
   }
   // There are no out of bounds particles
   else if(oobIdx[0] == 0)
   {
      printf("WARNING: No out of bounds particles were detected.\n");
      return;
   }
   assert(numParticles > oobIdx[0] + 1);
   unsigned int alignedStart = ((numParticles - oobIdx[0]) / (16)) * 16;

   numThreads = MAX_THREADS_PER_BLOCK / 4;
   resizeDim3(blockSize, numThreads);
   resizeDim3(numBlocks, calcNumBlocks(numThreads, 
      numParticles-(alignedStart)));
   sharedMemoryBytes = numThreads * sizeof(float2);
   findGoodIndicies<<<numBlocks, blockSize, sharedMemoryBytes, stream>>>(
      partLoc.getPtr(), numParticles,
      dev_moveCandIdx.getPtr(), dev_moveCandidates.getPtr(),
      alignedStart, oobIdx[0], NY1);
   checkForCudaError("findGoodIndices");

   cudaStreamSynchronize(stream);
   checkForCudaError("Before sorting oobArry");
   moveCandIdx = dev_moveCandIdx;
   
   // If there are good particles in the top portion of the array,
   // find them so they can be moved down
   if(moveCandIdx[0] > 0)
   {
      picSort(dev_oobArry, oobIdx[0]);
      picSort(dev_moveCandidates, moveCandIdx[0]);

      numThreads = MAX_THREADS_PER_BLOCK / 4;
      resizeDim3(blockSize, numThreads);
      resizeDim3(numBlocks, calcNumBlocks(numThreads, moveCandIdx[0]));
      cudaStreamSynchronize(stream);
      checkForCudaError("Before killParticles");
      killParticles<<<numBlocks, blockSize, 0, stream>>>(
         partLoc.getPtr(), partVel.getPtr(),
         dev_oobArry.getPtr(), dev_moveCandidates.getPtr(), moveCandIdx[0]);
      checkForCudaError("killParticles");
   }

   numParticles -= oobIdx[0];

   if(CommandlineOptions::getRef().getParticleBoundCheck())
   {
      DevMem<bool, DevMemReuse> dev_success;
      dev_success.fill(true);
      static HostMem<bool> success(1);
      numThreads = 256;
      numBlocks = (numParticles + numThreads - 1) / numThreads;
      cudaStreamSynchronize(stream);
      checkBoundaryConditions<<<numBlocks, numThreads, 0, stream>>>(
         partLoc.getPtr(), 
         numParticles, 
         NY1, NX1, 
         dev_success.getPtr());
      cudaStreamSynchronize(stream);
      success = dev_success;
      if(!success[0])
      {
         std::cerr << "ERROR: The movep function failed to constrain "
                   << "all particles to the grid!" << std::endl;
      }
      assert(success[0]);
   }

   // DEBUG
   //{
   //   LoggingThread &logger(LoggingThread::getRef());
   //   cudaThreadSynchronize();
   //   logger.pushLogItem(
   //      new LogParticlesAscii(1000, partLoc, partVel,
   //      partLoc, partVel,
   //      partLoc, partVel,
   //      partLoc, partVel,
   //      numParticles, numParticles,
   //      numParticles, numParticles));
   //   logger.flush();
   //}
   // END DEBUG
}

