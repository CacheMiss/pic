#pragma once

#include <iostream>

#include <cuda_runtime_api.h>
#include <algorithm>

#include "error_check.h"
#include "dev_mem.h"
#include "pitched_ptr.h"

//class CudaAllocator;
template<class T, class Allocator>
class DevMem;

template<class T>
class HostMem
{
   public:
   HostMem(std::size_t size);
   HostMem(std::size_t size, int val);
   HostMem(const PitchedPtr<T> &rhs);
   template<class Allocator>
   HostMem(const DevMem<T, Allocator> &rhs);
   HostMem();
   ~HostMem();

   std::size_t size() const;
   void resize(std::size_t newSize);
   // Resize to match the given vector and copy the vectors data into the class
   void resize(const std::vector<T> &dataToCopy);
   void push_back(const T &rhs);
   void clear();

   T& operator[](std::size_t i);
   const T& operator[](std::size_t i) const;
   T* getPtr();
   const T* getPtr() const;

   template<class Allocator>
   const HostMem<T>& operator=(const DevMem<T, Allocator> &rhs);
   template<class Allocator>
   const HostMem<T>& operator=(const PitchedPtr<T, Allocator> &rhs);

   private:
   T* m_ptr;
   std::size_t m_size;
   std::size_t m_reserved;

   void reserve(std::size_t n);
};

template<class T>
HostMem<T>::HostMem()
  :m_ptr(NULL)
  , m_size(0)
  , m_reserved(0)
{
}

template<class T>
HostMem<T>::HostMem(std::size_t size)
  :m_size(size)
  , m_reserved(size)
{
   checkCuda(cudaMallocHost(reinterpret_cast<void**>(&m_ptr), sizeof(T) * m_size));
}

template<class T>
HostMem<T>::HostMem(std::size_t size, int val)
  :m_size(size)
  , m_reserved(size)
{
   checkCuda(cudaMallocHost(reinterpret_cast<void**>(&m_ptr), sizeof(T) * m_size));
   memset(m_ptr, val, sizeof(T) * m_size);
}

template<class T>
HostMem<T>::HostMem(const PitchedPtr<T> &rhs)
{
   const PitchedPtr_t<T> &rhsPtr = rhs.getPtr();
   m_size = rhsPtr.x * rhsPtr.y;
   m_reserved = m_size;
   checkCuda(cudaMallocHost(reinterpret_cast<void**>(&m_ptr), sizeof(T) * m_size));
   checkCuda(cudaMemcpy2D(m_ptr, rhsPtr.widthBytes, 
                          rhsPtr.ptr, rhsPtr.pitch, 
                          rhsPtr.widthBytes, rhsPtr.y, 
                          cudaMemcpyDeviceToHost));
}

template<class T>
template<class Allocator>
HostMem<T>::HostMem(const DevMem<T, Allocator> &rhs)
  :m_ptr(NULL)
  , m_size(0)
  , m_reserved(0)
{
   operator=(rhs);
}

template<class T>
HostMem<T>::~HostMem()
{
   //checkCuda(cudaFreeHost(m_ptr));
   cudaFreeHost(m_ptr);
}

template<class T>
std::size_t HostMem<T>::size() const
{
   return m_size;
}

template<class T>
void HostMem<T>::resize(std::size_t newSize)
{
   if(newSize > m_reserved)
   {
      checkCuda(cudaFreeHost(m_ptr));
      m_size = newSize;
      m_reserved = m_size;
      checkCuda(cudaMallocHost(reinterpret_cast<void**>(&m_ptr), sizeof(T) * m_reserved));
   }
   else
   {
      m_size = newSize;
   }
}

template<class T>
void HostMem<T>::resize(const std::vector<T> &dataToCopy)
{
   resize(dataToCopy.size());
   memcpy(m_ptr, &dataToCopy[0], sizeof(T) * m_size);
}

template<class T>
T& HostMem<T>::operator[](std::size_t i)
{
   assert(i < m_size);
   return m_ptr[i];
}

template<class T>
const T& HostMem<T>::operator[](std::size_t i) const
{
   assert(i < m_size);
   return m_ptr[i];
}

template<class T>
T* HostMem<T>::getPtr()
{
   assert(m_ptr != NULL);
   return m_ptr;
}

template<class T>
const T* HostMem<T>::getPtr() const
{
   assert(m_ptr != NULL);
   return m_ptr;
}

template<class T>
template<class Allocator>
const HostMem<T>& HostMem<T>::operator=(const DevMem<T, Allocator> &rhs)
{
   resize(rhs.size());
   checkCuda(cudaMemcpy(reinterpret_cast<void*>(m_ptr), 
             reinterpret_cast<const void*>(rhs.getPtr()), 
             sizeof(T) * m_size, cudaMemcpyDeviceToHost));

   return *this;
}

template<class T>
template<class Allocator>
const HostMem<T>& HostMem<T>::operator=(const PitchedPtr<T, Allocator> &rhs)
{
   const PitchedPtr_t<T> &rhsPtr = rhs.m_ptr;
   resize(rhsPtr.x * rhsPtr.y);
   checkCuda(cudaMemcpy2D(m_ptr, rhsPtr.widthBytes, 
                          rhsPtr.ptr, rhsPtr.pitch, 
                          rhsPtr.widthBytes, rhsPtr.y, 
                          cudaMemcpyDeviceToHost));

   return *this;
}

template<class T>
void HostMem<T>::reserve(std::size_t n)
{
   //std::cout << "reserve(" << n <<") called. Current reserve = " << m_reserved << std::endl;
   if(n <= m_reserved)
   {
      return;
   }

   T* tmp = m_ptr;

   m_reserved = n;
   checkCuda(cudaMallocHost(reinterpret_cast<void**>(&m_ptr), sizeof(T) * m_reserved));

   if(tmp != NULL)
   {
      if(m_size != 0)
      {
         memcpy(m_ptr, tmp, sizeof(T) * m_size);
      }
      cudaFreeHost(tmp);
   }
}

template<class T>
void HostMem<T>::push_back(const T &rhs)
{
   if(m_size == m_reserved)
   {
      reserve(std::max(static_cast<std::size_t>(100), m_size + 200));
   }
   m_ptr[m_size] = rhs;
   m_size++;
}

template<class T>
void HostMem<T>::clear()
{
   m_size = 0;
}

