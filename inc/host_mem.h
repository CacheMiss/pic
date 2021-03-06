////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014, Stephen C. Sewell
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////
#pragma once


#include <algorithm>
#include <boost/iterator/iterator_facade.hpp>
#include <cuda_runtime_api.h>
#include <iostream>

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

   template<class Value>
   class iterator_template
      : public boost::iterator_facade<
         iterator_template<Value>, 
         Value, 
         boost::random_access_traversal_tag 
         >
   {
      public:
      iterator_template() : m_data(NULL){}
      explicit iterator_template(Value* data, std::size_t index=0)
         :m_data(data+index){}

      template<class OtherValue>
      iterator_template(iterator_template<OtherValue> const& other)
         :m_data(other.m_data)
      {}

      Value& operator[](std::size_t n) const
      {
         return m_data[n];
      }

      private:
      friend class boost::iterator_core_access;
      template <class> friend class iterator_template;

      Value& dereference() const
      {
         return *m_data;
      }
      template<class OtherValue>
      bool equal(iterator_template<OtherValue> const& i) const
      {
         return m_data == i.m_data;
      }
      void increment()
      {
         m_data++;
      }
      void decrement()
      {
         m_data--;
      }
      void advance(std::size_t n)
      {
         m_data += n;
      }
      std::ptrdiff_t distance_to(iterator_template<Value> const& j) const
      {
         return j.m_data - m_data;
      }

      Value* m_data;
   };

   typedef iterator_template<T> iterator;
   typedef iterator_template<T const> const_iterator;

   HostMem(std::size_t size);
   HostMem(std::size_t size, int val);
   HostMem(const PitchedPtr<T> &rhs);
   template<class Allocator>
   HostMem(const DevMem<T, Allocator> &rhs);
   HostMem();
   ~HostMem();

   iterator begin();
   iterator end();

   std::size_t size() const;
   void resize(std::size_t newSize);
   // Resize to match the given vector and copy the vectors data into the class
   void resize(const std::vector<T> &dataToCopy);
   void setPadding(std::size_t newPadding);
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
   std::size_t m_padding;

   void reserve(std::size_t n);
   void allocate(std::size_t size, bool preserve=false);
};

template<class T>
HostMem<T>::HostMem()
  : m_ptr(NULL)
  , m_size(0)
  , m_reserved(0)
  , m_padding(0)
{
}

template<class T>
HostMem<T>::HostMem(std::size_t size)
  : m_ptr(NULL)
  , m_size(size)
  , m_reserved(0)
  , m_padding(0)
{
   allocate(m_size);
}

template<class T>
HostMem<T>::HostMem(std::size_t size, int val)
  : m_ptr(NULL)
  , m_size(size)
  , m_reserved(0)
  , m_padding(0)
{
   allocate(m_size);
   memset(m_ptr, val, sizeof(T) * m_size);
}

template<class T>
HostMem<T>::HostMem(const PitchedPtr<T> &rhs)
  : m_ptr(NULL)
  , m_size(0)
  , m_reserved(0)
  , m_padding(0)
{
   const PitchedPtr_t<T> &rhsPtr = rhs.getPtr();
   m_size = rhsPtr.x * rhsPtr.y;
   allocate(m_size);
   checkCuda(cudaMemcpy2D(m_ptr, rhsPtr.widthBytes, 
                          rhsPtr.ptr, rhsPtr.pitch, 
                          rhsPtr.widthBytes, rhsPtr.y, 
                          cudaMemcpyDeviceToHost));
}

template<class T>
template<class Allocator>
HostMem<T>::HostMem(const DevMem<T, Allocator> &rhs)
  : m_ptr(NULL)
  , m_size(0)
  , m_reserved(0)
  , m_padding(0)
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
typename HostMem<T>::iterator HostMem<T>::begin()
{
   return HostMem<T>::iterator(m_ptr, 0);
}

template<class T>
typename HostMem<T>::iterator HostMem<T>::end()
{
   return HostMem<T>::iterator(m_ptr, m_size);
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
      m_size = newSize;
      allocate(m_size);
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

   m_reserved = n;
   allocate(m_reserved, true);
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

template<class T>
void HostMem<T>::setPadding(std::size_t newPadding)
{
   m_padding = newPadding;
}

template<class T>
void HostMem<T>::allocate(std::size_t size, bool preserve)
{
   if(!preserve)
   {
      checkCuda(cudaFreeHost(m_ptr));
      checkCuda(cudaMallocHost(reinterpret_cast<void**>(&m_ptr), (size + m_padding) * sizeof(T)));
      m_reserved = size + m_padding;
   }
   else
   {
      T* tmp = m_ptr;

      checkCuda(cudaMallocHost(reinterpret_cast<void**>(&m_ptr), sizeof(T) * (size + m_padding)));
      m_reserved = size + m_padding;

      if(tmp != NULL)
      {
         if(m_size != 0)
         {
            memcpy(m_ptr, tmp, sizeof(T) * m_size);
         }
         cudaFreeHost(tmp);
      }
   }
}