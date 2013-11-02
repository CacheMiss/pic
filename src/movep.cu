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
#include "simulation_state.h"

// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> texEx;
texture<float, 2, cudaReadModeElementType> texEy;
texture<float, 2, cudaReadModeElementType> texBxm;
texture<float, 1, cudaReadModeElementType> texBym;

__global__
void calcBxm(PitchedPtr_t<float> bxm,
             const float b0, 
             const float xMax, 
             const float yMax)
{
   const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
   const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

   if(x >= xMax || y >= yMax)
   {
      return;
   }
   // bxm = 2 * b0 * ((NY1 / (NY1 + y))^2 * (1 / (NY1 + y)) * (x - NX1/2)
   float result;
   float yMaxY = yMax + static_cast<float>(y);
   result = 2 * b0;
   result *= (yMax / yMaxY) * (yMax / yMaxY);
   result /= yMaxY;
   result *= static_cast<float>(x) - (xMax / 2);

   resolvePitchedPtr(bxm, x, y) = result;
}

__global__
void calcBym(float *bym, 
             const float b0, 
             const float yMax)
{
   unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
   if(y < yMax)
   {
      // bym = b0 * (yf / (yf + y))^2
      float tmp = yMax / (yMax + static_cast<float>(y));
      bym[y] = b0 * tmp * tmp;
   }
}

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
                   const unsigned int numParticles,
                   const float mass,
                   const unsigned int NX1, const unsigned int NY1,
                   const float OOB_PARTICLE)
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
   float bxm;
   float bym;

   if(threadX < numParticles)
   {
      if(!(pLoc.y > D_DY * (NY1-1) || pLoc.y < D_DY))
      {
         const int nii = static_cast<int>(pLoc.y/D_DY);  // Y Position of pBuficle
         const int nj = static_cast<int>((pLoc.x)/D_DX); // X Position of pBuficle
         // I can't simply add 1 to nj because if I wrap off the edge of the grid I
         // need to account for that. In that case, nj + 1 should actually be 0.
         const int nj1 = (nj == NX1-1) ? 0 : nj + 1;

         dela = (float)(fabs((float) (pLoc.x-(D_DX * nj))));
         delb = (float)(fabs((float) (pLoc.y-(D_DY * nii))));
         a1 = dela*delb;
         a2 = D_DX * delb - a1;
         a3 = D_DY * dela - a1;
         a4 = D_DX * D_DY-(a1+a2+a3);
         local_expf = (a1*tex2D(texEx, nj1, nii+1) + 
            a2*tex2D(texEx, nj, nii+1) +
            a3*tex2D(texEx, nj1, nii) + 
            a4*tex2D(texEx, nj, nii))/D_TOTA;
         local_eypf = (a1*tex2D(texEy, nj1, nii+1) +
            a2*tex2D(texEy, nj, nii+1) +
            a3*tex2D(texEy, nj1, nii) +
            a4*tex2D(texEy, nj, nii))/D_TOTA;
         bxm = a1 * tex2D(texBxm, nj+1, nii+1) +
            a2 * tex2D(texBxm, nj, nii+1) +
            a3 * tex2D(texBxm, nj+1, nii) +
            a4 * tex2D(texBxm, nj, nii) / D_TOTA;
         bym = (a1 * tex1D(texBym, nii+1) +
            a2 * tex1D(texBym, nii+1) +
            a3 * tex1D(texBym, nii) +
            a4 * tex1D(texBym, nii)) / D_TOTA;
         c1 = D_DELT * mass;
         c2 = 0.5f;
         c3 = c1*0.5f;
         c4 = c1*c2;
         a[0] = 1.0f;         // a[0, 0]
         a[1] = -c3*D_BZM;    // a[0, 1]
         a[2] = c3*bym;       // a[0, 2]
         a[3] = c3*D_BZM;     // a[1, 0]
         a[4] = 1.0f;         // a[1, 1]
         a[5] = -c3*bxm;      // a[1, 2]
         a[6] = -c3*bym;      // a[2, 0]
         a[7] = c3*bxm;       // a[2, 1]
         a[8] = 1.0f;         // a[2, 2]
         b[0] = pVel.x + pVel.y*c4*D_BZM - 
            pVel.z*c4*bym + c1*local_expf;
         b[1] = pVel.y + pVel.z*c4*bxm - 
            pVel.x*c4*D_BZM + c1*local_eypf;
         b[2] = pVel.z + pVel.x*c4*bym - 
            pVel.y*c4*bxm;
         det = a[0]*(a[4]*a[8] - 
            a[5]*a[7])-
            a[1]*(a[3]*a[8] - 
            a[5]*a[6])+
            a[2]*(a[3]*a[7] - 
            a[4]*a[6]);
         det1 = b[0]*(a[4]*a[8] - 
            a[5]*a[7]) -
            a[1]*(b[1]*a[8] - 
            b[2]*a[5]) +
            a[2]*(b[1]*a[7] - 
            b[2]*a[4]);
         det2 = a[0]*(b[1]*a[8] - 
            b[2]*a[5]) -
            b[0]*(a[3]*a[8] - 
            a[5]*a[6]) +
            a[2]*(b[2]*a[3] - 
            b[1]*a[6]);
         det3 = a[0]*(a[4]*b[2] - 
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
         pLoc.x = pLoc.x - (D_DX * NX1);
         while(pLoc.x > D_DX * NX1)
         {
            pLoc.x = pLoc.x - (D_DX * NX1);
         }
      }
      else if (pLoc.x <  0)
      { 
         pLoc.x = pLoc.x + (D_DX * NX1);
         while(pLoc.x <  0)
         {
            pLoc.x = pLoc.x + (D_DX * NX1);
         }
      }

      if(pLoc.y > D_DY * (NY1-1) || pLoc.y < D_DY)
      {
         pLoc.y = OOB_PARTICLE;
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

////////////////////////////////////////////////////////////////////////////////
/// @brief
/// Find the indices of all of the particles which are out of bounds. I very
/// important assumption here is that MAX_OOB_BUFFER is equal to the number of
/// particles which are out of bounds. This has to be inforced by the caller.
///
/// @param[in] d_partLoc
///    The locations of the particles being examined
/// @param[in] numParticles
///    The size of d_partLoc
/// @param[in,out] oobIdx
///    Aa 4 byte segment of global memory initially initialized to zero. It is
///    used by the kernel to write to oobArry in an orderly fashion
/// @param[in] oobArry
///    An array storing the indices of all of the particles which are out
///    of bounds
/// @param[in] NX1
///    The width of the grid
/// @param[in] NY1
///    The height of the grid
/// @param[in] MAX_OOB_BUFFER
///    The size of oobArry and the number of out of bound particles
////////////////////////////////////////////////////////////////////////////////
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
   __shared__ int blockOobStartIdx;
   if(threadIdx.x == 0) blockOobStartIdx = 0;
   __syncthreads();
   extern __shared__ char sharedBeg[];
   volatile float2 *pLoc = (float2*)sharedBeg;
   //volatile unsigned int *sharedWork = 
   //   (volatile unsigned int*)(sharedBeg + ((sizeof(float2) + sizeof(float3)) * blockDim.x));
   //volatile unsigned int *warpBegIdx = 
   //   const_cast<unsigned int*>(&sharedWork[blockDim.x * 16] + warpId);
   volatile unsigned int *warpOobStartIdx = 
      (volatile unsigned int*)(sharedBeg + sizeof(float2) * blockDim.x) + warpId;
   // End Shared Memory

   // Load the particles into shared memory
   {
      float2 tmpLoc = d_partLoc[threadX];
      pLoc[threadIdx.x].x = tmpLoc.x;
      pLoc[threadIdx.x].y = tmpLoc.y;
   }

   if(threadX < numParticles)
   {
      const unsigned int oobBallot = 
         __ballot(threadX < (numParticles - MAX_OOB_BUFFER) &&
                  (pLoc[threadIdx.x].y > D_DY * (NY1-1) ||
                   pLoc[threadIdx.x].y < D_DY));
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
   }
}

//******************************************************************************
// Function: findGoodIndices
// Code Type: Device
// Block Structure: One thread for each particle that needs to be removed
// Purpose: If there are 1000 particles and 100 need to be removed, this kernel
//          finds which indices in the last 100 elements of the array don't need
//          to be removed. These particles are later moved lower into the array
//
// WARNING: This assumes device has been synchronized prior to entry
//
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
      unsigned int dstIndex = oobArry[threadX];
      unsigned int srcIndex = moveCandidates[threadX];
      partLoc[dstIndex] = partLoc[srcIndex];
      partVel[dstIndex] = partVel[srcIndex];
   }
}

__global__
void countOobParticles(float2 position[], 
                       unsigned int* numOob, 
                       const unsigned int numParticles,
                       const unsigned int NY1)
{
   const unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;
   const unsigned int threadInWarp = threadIdx.x % warpSize;
   unsigned int ballotResults;
   float2 pos;

   if(threadX < numParticles)
   {
      pos = position[threadX];
      // Find all the threads with particles out of bounds
      ballotResults = __ballot(pos.y > D_DY * (NY1-1) || pos.y < D_DY);

      if(threadInWarp == 0)
      {
         // Find the Hamming weight of the ballot
         unsigned int localNumOob = __popc(ballotResults);
         if(localNumOob)
         {
            atomicAdd(numOob, localNumOob);
         }
      }
   }
}

/**************************************************************
       movep Move particle routine
**************************************************************/
void movep(DevMem<float2> &partLoc, DevMem<float3> &partVel,
           unsigned int &numParticles, float mass,
           const PitchedPtr<float> &ex, const PitchedPtr<float> &ey,
           DevStream &stream, bool updateField)
{
   static bool first = true;

   int numThreads;
   dim3 blockSize;
   dim3 numBlocks;

   DeviceStats &dev(DeviceStats::getRef());
   SimulationState &simState(SimulationState::getRef());

   //DevMem<unsigned int, DevMemReuse> dev_oobIdx;
   //assert(dev_oobIdx.getPtr() != NULL);
   //dev_oobIdx.zeroMem();
   //DevMem<unsigned int, DevMemReuse> dev_moveCandIdx;
   //dev_moveCandIdx.zeroMem();
   //static HostMem<unsigned int> oobIdx(1);
   //static HostMem<unsigned int> moveCandIdx(1);

   // DEBUG
   //{
   //   LoggingThread &logger(LoggingThread::getRef());
   //   stream.synchronize();
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

   // Allocate array and copy image data
   cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
   static cudaArray *cuArrayEx = NULL;
   static cudaArray *cuArrayEy = NULL;
   static cudaArray *cuArrayBxm = NULL;
   static cudaArray *cuArrayBym = NULL;
   if(first)
   {
      checkCuda(cudaMallocArray(&cuArrayEx,
         &channelDesc,
         NX1,
         NY+1));
      checkCuda(cudaMallocArray(&cuArrayEy,
         &channelDesc,
         NX1,
         NY+1));
      checkCuda(cudaMallocArray(&cuArrayBxm,
         &channelDesc,
         NX1,
         NY+1));
      checkCuda(cudaMallocArray(&cuArrayBym,
         &channelDesc,
         NY+1));
   }

   if(first)
   {
      PitchedPtr<float> d_bxm(NX1, NY+1);
      DevMem<float> d_bym(NY+1);

      resizeDim3(blockSize, 256);
      resizeDim3(numBlocks, calcNumBlocks(256, static_cast<unsigned int>(d_bxm.getX())),
         calcNumBlocks(1, d_bxm.getY()));
      calcBxm<<<numBlocks, blockSize, 0, *stream>>>(
         d_bxm.getPtr(), 
         B0, 
         static_cast<float>(d_bxm.getX()), 
         static_cast<float>(d_bxm.getY()));
      checkForCudaError("calcBxm");

      resizeDim3(blockSize, 256);
      resizeDim3(numBlocks, calcNumBlocks(256, d_bym.size()));
      calcBym<<<numBlocks, blockSize, 0, *stream>>>(
         d_bym.getPtr(), 
         B0, 
         static_cast<float>(d_bym.size()));
      checkForCudaError("calcBxm");

      stream.synchronize();
      checkCuda(cudaMemcpy2DToArray(cuArrayBxm,
         0,
         0,
         d_bxm.getPtr().ptr,
         d_bxm.getPtr().pitch,
         d_bxm.getPtr().widthBytes,
         d_bxm.getPtr().y,
         cudaMemcpyDeviceToDevice));
      checkCuda(cudaMemcpyToArray(cuArrayBym,
         0,
         0,
         d_bym.getPtr(),
         d_bym.size() * sizeof(float),
         cudaMemcpyDeviceToDevice));

      texBxm.addressMode[0] = cudaAddressModeWrap;
      texBxm.addressMode[1] = cudaAddressModeClamp;
      texBxm.filterMode = cudaFilterModePoint;
      texBxm.normalized = false;

      texBym.addressMode[0] = cudaAddressModeClamp;
      texBym.filterMode = cudaFilterModePoint;
      texBym.normalized = false;

      // Bind the array to the texture
      checkCuda(cudaBindTextureToArray(texBxm, cuArrayBxm, channelDesc));
      checkCuda(cudaBindTextureToArray(texBym, cuArrayBym, channelDesc));
   }

   if(updateField)
   {
      checkCuda(cudaMemcpy2DToArray(cuArrayEx,
         0,
         0,
         ex.getPtr().ptr,
         ex.getPtr().pitch,
         ex.getPtr().widthBytes,
         ex.getPtr().y,
         cudaMemcpyDeviceToDevice));
      checkCuda(cudaMemcpy2DToArray(cuArrayEy,
         0,
         0,
         ey.getPtr().ptr,
         ey.getPtr().pitch,
         ey.getPtr().widthBytes,
         ey.getPtr().y,
         cudaMemcpyDeviceToDevice));
      stream.synchronize();
   }

   if(first)
   {
      // Set texture parameters
      texEx.addressMode[0] = cudaAddressModeWrap;
      texEx.addressMode[1] = cudaAddressModeClamp;
      texEx.filterMode = cudaFilterModePoint;
      texEx.normalized = false;

      texEy.addressMode[0] = cudaAddressModeWrap;
      texEy.addressMode[1] = cudaAddressModeClamp;
      texEy.filterMode = cudaFilterModePoint;
      texEy.normalized = false;

      // Bind the array to the texture
      checkCuda(cudaBindTextureToArray(texEx, cuArrayEx, channelDesc));
      checkCuda(cudaBindTextureToArray(texEy, cuArrayEy, channelDesc));
   }

   numThreads = dev.maxThreadsPerBlock / 2;
   resizeDim3(blockSize, numThreads);
   resizeDim3(numBlocks, calcNumBlocks(numThreads, numParticles));
   checkForCudaError("Before moveParticles");
   moveParticles<<<numBlocks, blockSize, 0, *stream>>>(
      partLoc.getPtr(), partVel.getPtr(),
      numParticles, mass, NX1, NY1, OOB_PARTICLE);
   checkForCudaError("moveParticles");

   /*
   numThreads = dev.maxThreadsPerBlock / 2;
   resizeDim3(blockSize, numThreads);
   resizeDim3(numBlocks, calcNumBlocks(numThreads, numParticles));
   stream.synchronize();
   checkForCudaError("Before countOobParticles");
   countOobParticles<<<numBlocks, blockSize, 0, *stream>>>(
      partLoc.getPtr(), dev_oobIdx.getPtr(), numParticles, NY1);
   checkForCudaError("countOobParticles");

   stream.synchronize();
   // Get the number of particles that are outside of the y bounds
   oobIdx = dev_oobIdx;
   dev_oobIdx.zeroMem();
   if(numParticles == oobIdx[0])
   {
      printf("WARNING: %d of %d particles eliminated\n", oobIdx[0], numParticles);
      numParticles = 0;
   }
   // There are no out of bounds particles
   else if(oobIdx[0] == 0)
   {
      printf("WARNING: No out of bounds particles were detected.\n");
   }
   else
   {
      assert(numParticles >= oobIdx[0]);
      DevMem<unsigned int, ParticleAllocator> dev_oobArry(oobIdx[0]);

      numThreads = dev.maxThreadsPerBlock / 4;
      resizeDim3(blockSize, numThreads);
      resizeDim3(numBlocks, calcNumBlocks(numThreads, numParticles));
      sharedMemoryBytes = numThreads * sizeof(float2) + 
         sizeof(unsigned int) * numThreads/dev.warpSize;
      DevMem<unsigned int, ParticleAllocator> dev_moveCandidates(oobIdx[0]);
      stream.synchronize();
      checkForCudaError("Before findOobParticles");
      findOobParticles<<<numBlocks, blockSize, sharedMemoryBytes, *stream>>>(
         partLoc.getPtr(), 
         numParticles, dev_oobIdx.getPtr(), dev_oobArry.getPtr(),
         NX1, NY1, 
         static_cast<unsigned int>(dev_oobArry.size()));
      checkForCudaError("moveParticles");

      // DEBUG
      //{
      //   LoggingThread &logger(LoggingThread::getRef());
      //   stream.synchronize();
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

      stream.synchronize();
      unsigned int alignedStart = ((numParticles - oobIdx[0]) / (16)) * 16;

      numThreads = MAX_THREADS_PER_BLOCK / 4;
      resizeDim3(blockSize, numThreads);
      resizeDim3(numBlocks, calcNumBlocks(numThreads, 
         numParticles-(alignedStart)));
      sharedMemoryBytes = numThreads * sizeof(float2);
      // If there are good particles in the top portion of the array,
      // find them so they can be moved down
      findGoodIndicies<<<numBlocks, blockSize, sharedMemoryBytes, *stream>>>(
         partLoc.getPtr(), numParticles,
         dev_moveCandIdx.getPtr(), dev_moveCandidates.getPtr(),
         alignedStart, oobIdx[0], NY1);
      checkForCudaError("findGoodIndices");

      stream.synchronize();
      checkForCudaError("Before sorting oobArry");
      moveCandIdx = dev_moveCandIdx;

      if(moveCandIdx[0] > 0)
      {
         //picSort(dev_oobArry, oobIdx[0]);
         //picSort(dev_moveCandidates, moveCandIdx[0]);

         numThreads = MAX_THREADS_PER_BLOCK / 4;
         resizeDim3(blockSize, numThreads);
         resizeDim3(numBlocks, calcNumBlocks(numThreads, moveCandIdx[0]));
         stream.synchronize();
         checkForCudaError("Before killParticles");
         killParticles<<<numBlocks, blockSize, 0, *stream>>>(
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
         stream.synchronize();
         checkBoundaryConditions<<<numBlocks, numThreads, 0, *stream>>>(
            partLoc.getPtr(), 
            numParticles, 
            NY1, NX1, 
            dev_success.getPtr());
         stream.synchronize();
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
      //   stream.synchronize();
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
   */

   first = false;
}

