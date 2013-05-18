#pragma once

#include <cuda_runtime_api.h>

#include "pitched_ptr.h"

class CudaAllocator
{
public:
   static CudaAllocator& getRef();

   template<class T>
   void allocate(T* &ptr, std::size_t nElements);
   template<class T>
   void allocatePitched(PitchedPtr<T> &ptr, std::size_t sizeX, std::size_t sizeY);
   template<class T>
   void free(T* m);
   template<class T>
   void free(PitchedPtr<T> &m);

private:
   static CudaAllocator *m_ref;
};

template<class T>
void CudaAllocator::allocate(T* &ptr, std::size_t nElements)
{
   checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * nElements));
}

template<class T>
void CudaAllocator::allocatePitched(PitchedPtr<T> &ptr, std::size_t sizeX, std::size_t sizeY)
{
   allocatePitchedPtr(ptr, sizeX, sizeY);
}

template<class T>
void CudaAllocator::free(T* m)
{
   cudaFree(m);
}

template<class T>
void CudaAllocator::free(PitchedPtr<T> &m)
{
   cudaFree(m.ptr);
}
