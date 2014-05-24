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
#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H

#include "d_global_variables.h"

#include <assert.h>
#include <cuda.h>

#ifndef NO_THRUST
#include <thrust/device_vector.h>
#endif

#include "cuda_allocator.h"
#include "pic_utils.h"
#include "global_variables.h"

#include <thrust/sort.h>

#ifdef DEVEMU
#include <vector>
#endif

template<class KeyType, class AllocatorType>
void picSort(DevMem<KeyType, AllocatorType> &keys, int size=-1)
{
   assert(size != 0);
   if(size == -1)
   {
      size = keys.size();
   }

   thrust::sort(keys.getThrustPtr(), keys.getThrustPtr() + size);
   checkForCudaError("Radix sort failed");
}

template<class KeyType, class KeyAllocatorType, class ValType, class ValAllocatorType>
void picSort(DevMem<KeyType, KeyAllocatorType> &keys, DevMem<ValType, ValAllocatorType> &values, int size=0)
{
   if(size == 0)
   {
      size = static_cast<unsigned int>(keys.size());
   }

   thrust::sort_by_key(keys.getThrustPtr(),
      keys.getThrustPtr() + size, values.getThrustPtr());
   //thrust::stable_sort_by_key(keys.getThrustPtr(),
   //   keys.getThrustPtr() + size, values.getThrustPtr());
   checkForCudaError("Radix sort failed");
}

template<class Type>
__global__
void setValuesKernel(Type *t, unsigned int size, const Type val)
{
   if(blockDim.x * blockIdx.x + threadIdx.x >= size)
   {
      return;
   }
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   t[index] = val;
}

template<class Type>
void setDeviceArray(Type *devArray, unsigned int size, const Type &val)
{
   DeviceStats &dev(DeviceStats::getRef());
   int numThreads = dev.maxThreadsPerBlock / 2;
   dim3 blockSize(numThreads);
   dim3 numBlocks(calcNumBlocks(numThreads, size));
   setValuesKernel<<<numBlocks, blockSize>>>(devArray, size, val);
   checkForCudaError("setDeviceArray has failed");
}

template<class Type>
__global__
void multVectorKernel(Type *t, unsigned int size, const Type val)
{
   if(blockDim.x * blockIdx.x + threadIdx.x >= size)
   {
      return;
   }
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   t[index] = t[index] * val;
}

template<class Type>
void multVector(Type *devArray, unsigned int size, const Type val)
{
   int numThreads = MAX_THREADS_PER_BLOCK / 2;
   dim3 blockSize(numThreads);
   dim3 numBlocks(calcNumBlocks(numThreads, size));
   multVectorKernel<Type> <<<numBlocks, numThreads>>>(devArray, size, val);
   checkForCudaError("multVector");
}

template<class Type>
__global__
void divVectorKernel(Type *t, unsigned int size, const Type val)
{
   if(blockDim.x * blockIdx.x + threadIdx.x >= size)
   {
      return;
   }
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   t[index] /= val;
}

template<class Type>
void divVector(Type *devArray, unsigned int size, const Type val)
{
   unsigned int numThreads = MAX_THREADS_PER_BLOCK / 2;
   dim3 blockSize(numThreads);
   dim3 numBlocks(calcNumBlocks(numThreads, size));
   divVectorKernel<Type> <<<numBlocks, numThreads>>>(devArray, size, val);
   checkForCudaError("divVector");
}

#ifndef NO_THRUST
template<class Type>
inline void divVector(thrust::device_vector<Type> &devArray, 
                       const Type val)
{
   divVector<Type>(thrust::raw_pointer_cast(&devArray[0]), 
      devArray.size(), val);
}
#endif

template<class Type>
inline void divVector(DevMem<Type> &devArray, 
                       const Type val)
{
   divVector<Type>(devArray.getPtr(), 
      static_cast<unsigned int>(devArray.size()), 
      val);

}

template<class Type>
__global__
void subVectorKernel(const Type *a, const Type *b, Type *result, unsigned int size)
{
   if(blockDim.x * blockIdx.x + threadIdx.x >= size)
   {
      return;
   }
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   result[index] = a[index] - b[index];
}

template<class Type>
void subVector(Type *a, Type *b, Type *result, unsigned int size)
{
   int numThreads = MAX_THREADS_PER_BLOCK / 2;
   dim3 blockSize(numThreads);
   dim3 numBlocks(calcNumBlocks(numThreads, size));
   subVectorKernel<<<numBlocks, numThreads>>>(a, b, result, size);
   checkForCudaError("divVector");
}

#endif
