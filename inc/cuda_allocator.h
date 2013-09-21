#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>

class CudaAllocator
{
public:
   static CudaAllocator& getRef();

   template<class T>
   void allocate(T* &ptr, std::size_t nElements);
   template<class T>
   void free(T* m);

private:
   static CudaAllocator *m_ref;
};

template<class T>
void CudaAllocator::allocate(T* &ptr, std::size_t nElements)
{
   checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * nElements));
}

template<class T>
void CudaAllocator::free(T* m)
{
   cudaFree(m);
}
