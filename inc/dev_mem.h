#ifndef DEV_MEM_H
#define DEV_MEM_H

#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <vector>

#include "d_global_variables.h"
#include "device_stats.h"
#include "simulation_state.h"

#ifndef NO_THRUST
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#endif

template<class Type>
class DevMem
{
public:
   DevMem();
   DevMem(unsigned int size);
#ifndef NO_THRUST
   DevMem(unsigned int size, Type val);
#endif
   DevMem(Type *ptr, unsigned int size);
   ~DevMem();
   void copyToHost(Type &hostType) const;
   void copyArrayToHost(Type hostType[]) const;
   void copyArrayToHost(Type hostType[], unsigned int size) const;
   void copyArrayToDev(Type h_array[], unsigned int numElements);
   const Type* getPtr() const;
   Type* getPtr();
   void setPtr(Type *newPtr);
#ifndef NO_THRUST
   const thrust::device_ptr<Type> getThrustPtr() const;
   thrust::device_ptr<Type> getThrustPtr();
   void fill(Type val);
#endif
   void freeMem();
   unsigned int size() const;
   unsigned int sizeBytes() const;
   void resize(unsigned int newSize);
   void zeroMem();
   void operator=(const Type &rhs);
#ifndef NO_THRUST
   void operator=(const thrust::host_vector<Type> &rhs);
#endif
   void operator=(const std::vector<Type> &rhs);
   void operator=(const Type * const rhs);

private:
   unsigned int sizeAllocated;
   Type *ptr;
   bool manageMem;
   void allocateMem(unsigned int s);
   void checkForCudaError(cudaError_t error) const;
};


template<class Type>
DevMem<Type>::DevMem()
{
   manageMem = true;
   allocateMem(1);
}

template<class Type>
DevMem<Type>::DevMem(unsigned int size)
{
   assert(size > 0);
   manageMem = true;
   allocateMem(size);
}

#ifndef NO_THRUST
template<class Type>
DevMem<Type>::DevMem(unsigned int size, Type val)
{
   assert(size > 0);
   manageMem= true;
   allocateMem(size);
   fill(val);
}
#endif

template<class Type>
DevMem<Type>::DevMem(Type *ptr, unsigned int size)
{
   this->ptr = ptr;
   sizeAllocated = size;
   manageMem= false;
}

template<class Type>
DevMem<Type>::~DevMem()
{
   if(manageMem)
   {
      freeMem();
   }
}

#ifndef NO_THRUST
template<class Type>
void DevMem<Type>::fill(Type val)
{
   thrust::fill(getThrustPtr(),
      getThrustPtr() + size(),
      val);
}
#endif

template<class Type>
void DevMem<Type>::copyToHost(Type &hostType) const
{
   cudaError_t error;
   error = cudaMemcpy((void*)&hostType, ptr, sizeof(Type), cudaMemcpyDeviceToHost);
   checkForCudaError(error);
}

template<class Type>
void DevMem<Type>::copyArrayToHost(Type hostType[]) const
{
   cudaError_t error;
   error = cudaMemcpy((void*)hostType, ptr, sizeAllocated * sizeof(Type), 
      cudaMemcpyDeviceToHost);
   checkForCudaError(error);
}

template<class Type>
void DevMem<Type>::copyArrayToHost(Type hostType[], unsigned int size) const
{
   if(size == 0)
   {
      return;
   }
   assert(size < sizeAllocated);
   cudaError_t error;
   error = cudaMemcpy((void*)hostType, ptr, size * sizeof(Type), 
      cudaMemcpyDeviceToHost);
   checkForCudaError(error);
}

template<class Type>
void DevMem<Type>::copyArrayToDev(Type h_array[], unsigned int numElements)
{
   assert(numElements <= sizeAllocated);
   cudaError_t error;
   error = cudaMemcpy((void*)ptr, (void*)h_array, numElements * sizeof(Type), 
      cudaMemcpyHostToDevice);
   checkForCudaError(error);
}

#ifndef NO_THRUST
template<class Type>
thrust::device_ptr<Type> DevMem<Type>::getThrustPtr()
{
   return thrust::device_ptr<Type>(ptr);
}

template<class Type>
const thrust::device_ptr<Type> DevMem<Type>::getThrustPtr() const
{
   return thrust::device_ptr<Type>(ptr);
}
#endif

template<class Type>
Type* DevMem<Type>::getPtr()
{
   return ptr;
}

template<class Type>
const Type* DevMem<Type>::getPtr() const
{
   return ptr;
}

template<class Type>
void DevMem<Type>::setPtr(Type *newPtr)
{
   ptr = newPtr;
}

template<class Type>
unsigned int DevMem<Type>::size() const
{
   return sizeAllocated;
}

template<class Type>
unsigned int DevMem<Type>::sizeBytes() const
{
   return sizeAllocated * sizeof(Type);
}

template<class Type>
void DevMem<Type>::freeMem()
{
   if(ptr != NULL)
   {
      cudaFree(ptr);
   }
   ptr = NULL;
   sizeAllocated = 0;
}

template<class Type>
void DevMem<Type>::resize(unsigned int newSize)
{
   if(newSize > sizeAllocated)
   {
      cudaFree(ptr);
      allocateMem(newSize);
   }
}

template<class Type>
void DevMem<Type>::zeroMem()
{
#ifdef NO_THRUST
   cudaMemset((void*)ptr, 0, sizeAllocated * sizeof(Type));
#else
   thrust::fill(getThrustPtr(),
      getThrustPtr() + size(),
      (Type)0);
#endif
}

template<class Type>
void DevMem<Type>::operator=(const Type &rhs)
{
   cudaError_t error;

   error = cudaMemcpy(ptr, &rhs, sizeof(Type), cudaMemcpyHostToDevice);
   checkForCudaError(error);
}

#ifndef NO_THRUST
template<class Type>
void DevMem<Type>::operator=(const thrust::host_vector<Type> &rhs)
{
   cudaError_t error;
   assert(sizeAllocated == rhs.size());

   error = cudaMemcpy(ptr, &rhs[0], sizeof(Type) * sizeAllocated, 
                      cudaMemcpyHostToDevice);
   checkForCudaError(error);
}
#endif

template<class Type>
void DevMem<Type>::operator=(const std::vector<Type> &rhs)
{
   cudaError_t error;
   assert(sizeAllocated == rhs.size());

   error = cudaMemcpy(ptr, &rhs[0], sizeof(Type) * sizeAllocated, 
      cudaMemcpyHostToDevice);
   checkForCudaError(error);
}

template<class Type>
void DevMem<Type>::operator=(const Type * const rhs)
{
   cudaError_t error;

   error = cudaMemcpy(ptr, rhs, sizeof(Type) * sizeAllocated, 
      cudaMemcpyHostToDevice);
   checkForCudaError(error);
}

template<class Type>
void DevMem<Type>::allocateMem(unsigned int s)
{
   cudaError_t error;

   sizeAllocated = s;
   error = cudaMalloc(&ptr, sizeof(Type) * sizeAllocated);
   checkForCudaError(error);
}

template<class Type>
void DevMem<Type>::checkForCudaError(cudaError_t error) const
{
   if(error != cudaSuccess)
   {
      SimulationState &simState(SimulationState::getRef());
      fprintf(stderr,"ERROR on iteration %u: %s\n", simState.iterationNum,
         cudaGetErrorString(error) );
      assert(error == cudaSuccess);
      exit(1);
   }
}

#endif

