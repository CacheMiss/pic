#include "field.h"

#include "global_variables.h"
#include "pic_utils.h"

namespace Field
{
   __global__
   void calcEy(PitchedPtr_t<float> ey, const float phi[],
               const unsigned int NX1, const unsigned int NY1,
               const float DY)
   {
      float topPhi;
      float bottomPhi;
      
      unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;
      unsigned int threadY = blockDim.y * blockIdx.y + threadIdx.y;

      if(threadX >= NX1 || threadY >= NY1-1)
      {
         return;
      }

      topPhi = phi[NX1 * (threadY + 2) + threadX];
      bottomPhi = phi[NX1 * threadY + threadX];

      // for 2^126 <= y <= 2^128, __fdividef(x,y) delivers a result of zero,
      resolvePitchedPtr(ey, threadX, threadY + 1) =
         __fdividef(-(topPhi - bottomPhi), (float)2.0 * DY);
   }

   __global__
   void calcEx(PitchedPtr_t<float> ex, const float phi[], const unsigned int NX1,
               const unsigned int NY, const float DX)
   {
      extern __shared__ float begShared[];
      float *sharedPhi = begShared;

      unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;
      unsigned int threadY = blockDim.y * blockIdx.y + threadIdx.y;
      bool hasWork = true;
      float fieldVal = 0;

      if(threadX >= NX1 || threadY >= NY)
      {
         hasWork = false;
      }

      unsigned int gIndex = NX1 * threadY + threadX;

      if(hasWork)
      {
         sharedPhi[threadIdx.x + 1] = phi[gIndex];
         if(threadIdx.x == 0)
         {
            // If there is a phi element to the right, get it
            if(threadX + blockDim.x != NX1)
            {
               sharedPhi[blockDim.x + 1] = phi[gIndex + blockDim.x];
            }
            // Wrap back to the beginning
            else
            {
               sharedPhi[blockDim.x + 1] = phi[NX1 * threadY];
            }
            // If there is a phi element to the left, get it
            if(blockIdx.x != 0)
            {
               sharedPhi[threadIdx.x] = phi[gIndex - 1];
            }
         }
      }

      __syncthreads();

      if(!hasWork)
      {
         return;
      }
      if(threadX != 0)
      {
         // for 2^126 <= y <= 2^128, __fdividef(x,y) delivers a result of zero,
         fieldVal = 
            -__fdividef((sharedPhi[threadIdx.x + 2] - sharedPhi[threadIdx.x]),
            (float) 2 * DX);
      }
      else
      {
         // for 2^126 <= y <= 2^128, __fdividef(x,y) delivers a result of zero,
         fieldVal = -__fdividef((sharedPhi[2] - sharedPhi[1]), DX);
      }

      resolvePitchedPtr(ex, threadX, threadY) = fieldVal;
   }

   __global__
   void fixEyBoundaries(PitchedPtr_t<float> ey, const float phi[], 
                        const unsigned int NX1, const unsigned int NY1, 
                        const float DY)
   {
      unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;

      float phiTop;
      float phiBottom;

      phiTop = phi[NX1 + threadX];
      phiBottom = phi[threadX];
      // for 2^126 <= y <= 2^128, __fdividef(x,y) delivers a result of zero,
      resolvePitchedPtr(ey, threadX, 0) = __fdividef(-(phiTop - phiBottom), DY);

      phiTop = phi[NX1 * NY1 + threadX];
      phiBottom = phi[NX1 * (NY1 - 1) + threadX];

      // for 2^126 <= y <= 2^128, __fdividef(x,y) delivers a result of zero,
      resolvePitchedPtr(ey, threadX, NY1) = __fdividef(-(phiTop - phiBottom), DY);
   }

}

/****************************************************************
         subroutine field
 ***************************************************************/
void field(PitchedPtr<float> &ex,
           PitchedPtr<float> &ey,
           const DevMemF &phi)
{
   unsigned int numThreads;
   unsigned int sharedMemSizeBytes;
   dim3 blockSize;
   dim3 numBlocks;

   numThreads = MAX_THREADS_PER_BLOCK / 2;
   blockSize.x = numThreads;
   resizeDim3(numBlocks, calcNumBlocks(numThreads, NX1), NY1-1);
   cudaThreadSynchronize();
   checkForCudaError("Before calcEy");
   Field::calcEy<<<numBlocks, blockSize>>>(ey.getPtr(), phi.getPtr(),
      NX1, NY1, DY);
   checkForCudaError("calcEy");

   numThreads = MAX_THREADS_PER_BLOCK / 2;
   blockSize.x = numThreads;
   resizeDim3(numBlocks, calcNumBlocks(numThreads, NX1), NY1);
   sharedMemSizeBytes = (numThreads + 2) * sizeof(float);
   Field::calcEx<<<numBlocks, blockSize, sharedMemSizeBytes>>>(
      ex.getPtr(), phi.getPtr(), NX1, NY, DX);
   checkForCudaError("calcEx");

   numThreads = MAX_THREADS_PER_BLOCK / 2;
   blockSize.x = numThreads;
   resizeDim3(numBlocks, calcNumBlocks(numThreads, NX1));
   cudaThreadSynchronize();
   checkForCudaError("Before fixEyBoundaries");
   Field::fixEyBoundaries<<<numBlocks, blockSize>>>(ey.getPtr(), phi.getPtr(),
      NX1, NY1, DY);
   checkForCudaError("fixEyBoundaries");
}
