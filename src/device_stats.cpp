#include "device_stats.h"

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

DeviceStats *DeviceStats::m_ref = NULL;
DeviceStats & DeviceStats::getRef()
{
   if(m_ref == NULL)
   {
      m_ref = new DeviceStats();
      int deviceCount;
      cudaGetDeviceCount(&deviceCount);
      if (deviceCount == 0) {
         fprintf(stderr, "cutil error: no devices supporting CUDA.\n");
         exit(EXIT_FAILURE);
      }
      int dev = 0;
      cudaGetDeviceProperties(static_cast<cudaDeviceProp*>(m_ref), dev);
      //cudaDeviceProp deviceProp;
      //cudaGetDeviceProperties(&deviceProp, dev);
      if (m_ref->major < 1) {
         fprintf(stderr, "cutil error: device does not support CUDA.\n");
         exit(EXIT_FAILURE);
      }
      cudaError_t err = cudaSetDevice(dev);
      if(err != cudaSuccess)
      {
         fprintf(stderr,"ERROR Initializing CUDA Device: %s: %s\n", 
            cudaGetErrorString(err), m_ref->name);
         exit(1);
      }
      fprintf(stderr, "Using device %d: %s\n", dev, m_ref->name);

      //cudaGetDeviceProperties((cudaDeviceProp*)&m_ref, dev);
   }

    return *m_ref;
}
