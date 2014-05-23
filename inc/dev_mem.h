////////////////////////////////////////////////////////////////////////////////
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
// 
//   http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
////////////////////////////////////////////////////////////////////////////////
class CudaAllocator;
template<class T, class Allocator=CudaAllocator>
class DevMem;

#pragma once

#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <vector>

#include "cuda_allocator.h"
#include "device_stats.h"
#include "error_check.h"
#include "host_mem.h"

#ifndef NO_THRUST
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#endif

template<class T>
class HostMem;

template<class T, class Allocator>
class DevMem
{
public:
   DevMem();
   DevMem(std::size_t size);
#ifndef NO_THRUST
   DevMem(std::size_t size, T val);
#endif
   DevMem(T *ptr, std::size_t size);
   DevMem(const HostMem<T> &rhs);
   DevMem(const std::vector<T> &rhs);
   DevMem(const DevMem &rhs);
   ~DevMem();
   void copyToHost(T &hostType) const;
   void copyToVector(std::vector<T> &v) const;
   void copyArrayToHost(T hostType[]) const;
   void copyArrayToHost(T hostType[], std::size_t size) const;
   void copyArrayToDev(T h_array[], std::size_t numElements);
   void copyArrayToDev(const HostMem<T> &h_array);
   const T* getPtr() const;
   T* getPtr();
   const T* getPtrUnsafe() const;
   T* getPtrUnsafe();
   void setPtr(T *newPtr);
#ifndef NO_THRUST
   const thrust::device_ptr<T> getThrustPtr() const;
   thrust::device_ptr<T> getThrustPtr();
   std::size_t getElementSize() const;
   void fill(T val);
#endif
   void freeMem();
   std::size_t size() const;
   std::size_t sizeBytes() const;
   void resize(std::size_t newSize);
   void clear();
   void setPadding(std::size_t p);
   void zeroMem();
   const DevMem<T, Allocator>& operator=(const T &rhs);
#ifndef NO_THRUST
   const DevMem<T, Allocator>& operator=(const thrust::host_vector<T> &rhs);
#endif
   const DevMem<T, Allocator>& operator=(const std::vector<T> &rhs);
   const DevMem<T, Allocator>& operator=(const T * const rhs);
   const DevMem<T, Allocator>& operator=(const HostMem<T> &rhs);

private:
   std::size_t m_size;
   std::size_t m_reserved;
   std::size_t m_padding;
   T *m_ptr;
   // If true, free the mem in the destructor. If false, someone else is freeing it
   bool m_manageMem;

   void allocateMem(std::size_t s);
};


template<class T, class Allocator>
DevMem<T, Allocator>::DevMem()
  :m_ptr(NULL)
  ,m_size(0)
  ,m_reserved(0)
  ,m_padding(0)
{
   m_manageMem = true;
   allocateMem(1);
}

template<class T, class Allocator>
DevMem<T, Allocator>::DevMem(std::size_t size)
  :m_ptr(NULL)
  ,m_size(0)
  ,m_reserved(0)
  ,m_padding(0)
{
   m_manageMem = true;
   allocateMem(size);
}

#ifndef NO_THRUST
template<class T, class Allocator>
DevMem<T, Allocator>::DevMem(std::size_t size, T val)
  :m_ptr(NULL)
  ,m_size(0)
  ,m_reserved(0)
  ,m_padding(0)
{
   assert(size > 0);
   m_manageMem = true;
   allocateMem(size);
   fill(val);
}
#endif

template<class T, class Allocator>
DevMem<T, Allocator>::DevMem(T *ptr, std::size_t size)
  :m_ptr(ptr)
  ,m_size(size)
  ,m_reserved(size)
  ,m_padding(0)
  ,m_manageMem(false)
{
}

template<class T, class Allocator>
DevMem<T, Allocator>::DevMem(const HostMem<T> &rhs)
  :m_ptr(NULL)
  ,m_size(0)
  ,m_reserved(0)
  ,m_padding(0)
{
   allocateMem(rhs.size());
   checkCuda(cudaMemcpy(m_ptr, rhs.getPtr(), sizeof(T) * rhs.size(), cudaMemcpyHostToDevice));
}

template<class T, class Allocator>
DevMem<T, Allocator>::DevMem(const DevMem<T, Allocator> &rhs)
  :m_ptr(NULL)
  ,m_size(0)
  ,m_reserved(0)
  ,m_padding(rhs.m_padding)
{
   allocateMem(rhs.size());
   checkCuda(cudaMemcpy(m_ptr, rhs.getPtr(), sizeof(T) * rhs.size(), cudaMemcpyDeviceToDevice));
}

template<class T, class Allocator>
DevMem<T, Allocator>::DevMem(const std::vector<T> &rhs)
  :m_ptr(NULL)
  ,m_size(0)
  ,m_reserved(0)
  ,m_padding(0)
{
   allocateMem(rhs.size());
   checkCuda(cudaMemcpy(m_ptr, &rhs[0], sizeof(T) * m_size, cudaMemcpyHostToDevice));
}

template<class T, class Allocator>
DevMem<T, Allocator>::~DevMem()
{
   if(m_manageMem)
   {
      freeMem();
   }
}

#ifndef NO_THRUST
template<class T, class Allocator>
void DevMem<T, Allocator>::fill(T val)
{
   thrust::fill(getThrustPtr(),
      getThrustPtr() + size(),
      val);
}
#endif

template<class T, class Allocator>
void DevMem<T, Allocator>::copyToHost(T &hostType) const
{
   checkCuda(cudaMemcpy((void*)&hostType, m_ptr, sizeof(T), cudaMemcpyDeviceToHost));
}

template<class T, class Allocator>
void DevMem<T, Allocator>::copyToVector(std::vector<T> &v) const
{
   v.resize(m_size);
   checkCuda(cudaMemcpy((void*)&v[0], m_ptr, sizeof(T) * m_size, cudaMemcpyDeviceToHost));
}

template<class T, class Allocator>
void DevMem<T, Allocator>::copyArrayToHost(T hostType[]) const
{
   checkCuda(cudaMemcpy((void*)hostType, m_ptr, m_size * sizeof(T), 
                        cudaMemcpyDeviceToHost));
}

template<class T, class Allocator>
void DevMem<T, Allocator>::copyArrayToHost(T hostType[], std::size_t size) const
{
   if(size == 0)
   {
      return;
   }
   assert(size <= m_size);
   checkCuda(cudaMemcpy((void*)hostType, m_ptr, size * sizeof(T), 
                        cudaMemcpyDeviceToHost));
}

template<class T, class Allocator>
void DevMem<T, Allocator>::copyArrayToDev(const HostMem<T> &h_array)
{
   if(m_size < h_array.size())
   {
      resize(h_array.size());
   }
   checkCuda(cudaMemcpy((void*)m_ptr, (void*)&h_array[0], h_array.size() * sizeof(T), 
                        cudaMemcpyHostToDevice));
}

template<class T, class Allocator>
void DevMem<T, Allocator>::copyArrayToDev(T h_array[], std::size_t numElements)
{
   if(m_size < numElements)
   {
      resize(numElements);
   }
   checkCuda(cudaMemcpy((void*)m_ptr, (void*)h_array, numElements * sizeof(T), 
                        cudaMemcpyHostToDevice));
}

#ifndef NO_THRUST
template<class T, class Allocator>
thrust::device_ptr<T> DevMem<T, Allocator>::getThrustPtr()
{
   if(m_ptr == NULL)
   {
      throw CudaRuntimeError("DevMem<T, Allocator>::getThrustPtr() was called after the device memory was freed!");
   }
   return thrust::device_ptr<T>(m_ptr);
}

template<class T, class Allocator>
std::size_t DevMem<T, Allocator>::getElementSize() const
{
   return sizeof(T);
}

template<class T, class Allocator>
const thrust::device_ptr<T> DevMem<T, Allocator>::getThrustPtr() const
{
   if(m_ptr == NULL)
   {
      throw CudaRuntimeError("DevMem<T, Allocator>::getThrustPtr() was called after the device memory was freed!");
   }
   return thrust::device_ptr<T>(m_ptr);
}
#endif

template<class T, class Allocator>
T* DevMem<T, Allocator>::getPtr()
{
   if(m_ptr == NULL)
   {
      throw CudaRuntimeError("DevMem<T, Allocator>::getPtr() was called after the device memory was freed!");
   }
   return m_ptr;
}

// This won't throw an exception is m_ptr is NULL
template<class T, class Allocator>
const T* DevMem<T, Allocator>::getPtrUnsafe() const
{
   return m_ptr;
}

// This won't throw an exception is m_ptr is NULL
template<class T, class Allocator>
T* DevMem<T, Allocator>::getPtrUnsafe()
{
   return m_ptr;
}

template<class T, class Allocator>
const T* DevMem<T, Allocator>::getPtr() const
{
   if(m_ptr == NULL)
   {
      throw CudaRuntimeError("DevMem<T, Allocator>::getPtr() was called after the device memory was freed!");
   }
   return m_ptr;
}

template<class T, class Allocator>
void DevMem<T, Allocator>::setPtr(T *newPtr)
{
   m_ptr = newPtr;
}

template<class T, class Allocator>
std::size_t DevMem<T, Allocator>::size() const
{
   return m_size;
}

template<class T, class Allocator>
std::size_t DevMem<T, Allocator>::sizeBytes() const
{
   return m_size * sizeof(T);
}

template<class T, class Allocator>
void DevMem<T, Allocator>::freeMem()
{
   Allocator &a(Allocator::getRef());
   if(m_ptr != NULL)
   {
      a.free(m_ptr);
   }
   m_ptr = NULL;
   m_size = 0;
   m_reserved = 0;
}

template<class T, class Allocator>
void DevMem<T, Allocator>::resize(std::size_t newSize)
{
   allocateMem(newSize);
}

template<class T, class Allocator>
void DevMem<T, Allocator>::clear()
{
   m_size = 0;
}

template<class T, class Allocator>
void DevMem<T, Allocator>::zeroMem()
{
   cudaMemset((void*)m_ptr, 0, m_size * sizeof(T));
}

template<class T, class Allocator>
const DevMem<T, Allocator>& DevMem<T, Allocator>::operator=(const T &rhs)
{
   resize(sizeof(T));
   checkCuda(cudaMemcpy(m_ptr, &rhs, sizeof(T), cudaMemcpyHostToDevice));
   return *this;
}

#ifndef NO_THRUST
template<class T, class Allocator>
const DevMem<T, Allocator>& DevMem<T, Allocator>::operator=(const thrust::host_vector<T> &rhs)
{
   resize(rhs.size());
   checkCuda(cudaMemcpy(m_ptr, &rhs[0], sizeof(T) * m_size, 
                        cudaMemcpyHostToDevice));
   return *this;
}
#endif

template<class T, class Allocator>
const DevMem<T, Allocator>& DevMem<T, Allocator>::operator=(const std::vector<T> &rhs)
{
   resize(rhs.size());
   checkCuda(cudaMemcpy(m_ptr, &rhs[0], sizeof(T) * m_size, 
                        cudaMemcpyHostToDevice));
   return *this;
}

template<class T, class Allocator>
const DevMem<T, Allocator>& DevMem<T, Allocator>::operator=(const T * const rhs)
{
   checkCuda(cudaMemcpy(m_ptr, rhs, sizeof(T) * m_size, 
                        cudaMemcpyHostToDevice));
   return *this;
}

template<class T, class Allocator>
const DevMem<T, Allocator>& DevMem<T, Allocator>::operator=(const HostMem<T> &rhs)
{
   resize(rhs.size());
   cudaMemcpy(m_ptr, &rhs[0], sizeof(T) * m_size, cudaMemcpyHostToDevice);
   return *this;
}

template<class T, class Allocator>
void DevMem<T, Allocator>::allocateMem(std::size_t s)
{
   Allocator &a(Allocator::getRef());
   m_manageMem = true;
   m_size = s;
   if(m_size > m_reserved)
   {
      a.free(m_ptr);
      m_reserved = m_size + m_padding;
      a.allocate(m_ptr, m_reserved);
   }
}

template<class T, class Allocator>
void DevMem<T, Allocator>::setPadding(std::size_t p)
{
   m_padding = p;
}
