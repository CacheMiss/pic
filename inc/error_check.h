#pragma once

#include <driver_types.h>
#include <stdexcept>

class CudaRuntimeError : public std::runtime_error
{
public:
   CudaRuntimeError(const std::string &message) 
      : std::runtime_error(message)
   {}
};

void checkCuda(cudaError_t error);

