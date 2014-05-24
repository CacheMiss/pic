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
#include "device_stats.h"

#include <cuda_runtime_api.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "error_check.h"

DeviceStats *DeviceStats::m_ref = NULL;
DeviceStats & DeviceStats::getRef(int dev)
{
   if(m_ref == NULL)
   {
      try
      {
         m_ref = new DeviceStats();
         int deviceCount;
         cudaGetDeviceCount(&deviceCount);
         if (dev == -1 && deviceCount == 0) 
         {
            fprintf(stderr, "cutil error: no devices supporting CUDA.\n");
            exit(EXIT_FAILURE);
         }
         if(dev == -1)
         {
            dev = 0;
         }
         checkCuda(cudaGetDeviceProperties(static_cast<cudaDeviceProp*>(m_ref), dev));
         checkCuda(cudaSetDevice(dev));
         if (m_ref->major < 1) {
            fprintf(stderr, "Major cuda veresion supported is %d\n", m_ref->major);
            fprintf(stderr, "cutil error: device does not support CUDA.\n");
            exit(EXIT_FAILURE);
         }
         fprintf(stdout, "Using device %d: %s\n", dev, m_ref->name);
      }
      catch (CudaRuntimeError &e)
      {
         fprintf(stderr, "ERROR: Unable to initialize a CUDA capable device!\n");
         fprintf(stderr, "%s", e.what());
         exit(1);
      }
   }

    return *m_ref;
}

std::size_t DeviceStats::getTotalMemBytes() const
{
   std::size_t totalMem;
   std::size_t freeMem;
   checkCuda(cudaMemGetInfo(&freeMem, &totalMem));

   return totalMem;
}

std::size_t DeviceStats::getFreeMemBytes() const
{
   std::size_t totalMem;
   std::size_t freeMem;
   checkCuda(cudaMemGetInfo(&freeMem, &totalMem));

   return freeMem;
}

std::size_t DeviceStats::getTotalMemMb() const
{
   return getTotalMemBytes() / 1048576;
}

std::size_t DeviceStats::getFreeMemMb() const
{
   return getFreeMemBytes() / 1048576;
}

double DeviceStats::getPercentFreeMem() const
{
   std::size_t totalMem;
   std::size_t freeMem;
   checkCuda(cudaMemGetInfo(&freeMem, &totalMem));

   double percentFree = static_cast<double>(freeMem);
   percentFree /= totalMem;

   return percentFree;
}
