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
         // There is nothing to the left here. sharedPhi[1] is phi where x=0
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
           const DevMemF &phi,
           DevStream &stream1,
           DevStream &stream2)
{
   unsigned int numThreads;
   unsigned int sharedMemSizeBytes;
   dim3 blockSize;
   dim3 numBlocks;

   numThreads = MAX_THREADS_PER_BLOCK / 2;
   blockSize.x = numThreads;
   resizeDim3(numBlocks, calcNumBlocks(numThreads, NX1), NY1-1);
   stream1.synchronize();
   checkForCudaError("Before calcEy");
   Field::calcEy<<<numBlocks, blockSize, 0, *stream1>>>(
      ey.getPtr(), phi.getPtr(),
      NX1, NY1, DY);
   checkForCudaError("calcEy");

   numThreads = MAX_THREADS_PER_BLOCK / 2;
   blockSize.x = numThreads;
   resizeDim3(numBlocks, calcNumBlocks(numThreads, NX1), NY1);
   sharedMemSizeBytes = (numThreads + 2) * sizeof(float);
   Field::calcEx<<<numBlocks, blockSize, sharedMemSizeBytes, *stream2>>>(
      ex.getPtr(), phi.getPtr(), NX1, NY, DX);
   checkForCudaError("calcEx");

   numThreads = MAX_THREADS_PER_BLOCK / 2;
   blockSize.x = numThreads;
   resizeDim3(numBlocks, calcNumBlocks(numThreads, NX1));
   stream1.synchronize();
   checkForCudaError("Before fixEyBoundaries");
   Field::fixEyBoundaries<<<numBlocks, blockSize, 0, *stream1>>>(
      ey.getPtr(), phi.getPtr(),
      NX1, NY1, DY);
   checkForCudaError("fixEyBoundaries");
}
