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
