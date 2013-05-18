#include <cassert>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_functions.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include "error_check.h"

void checkCuda(cudaError_t error)
{
   if(error != cudaSuccess)
   {
      std::stringstream s;
      s << "ERROR: Cuda call failed! ";
      s << cudaGetErrorString(error);
      s << std::endl;
      std::cerr << s.str() << std::endl;
      throw CudaRuntimeError(s.str());
   }
}
