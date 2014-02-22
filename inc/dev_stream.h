#pragma once
#include <cuda_runtime_api.h>

#include "error_check.h"

class DevStream
{
public:
   DevStream()
   {
      checkCuda(cudaStreamCreate(&m_stream));
   }
   ~DevStream()
   {
      checkCuda(cudaStreamDestroy(m_stream));
   }
   cudaStream_t operator*()
   {
      return m_stream;
   }
   void synchronize()
   {
      checkCuda(cudaStreamSynchronize(m_stream));
   }
private:
   cudaStream_t m_stream;
};
