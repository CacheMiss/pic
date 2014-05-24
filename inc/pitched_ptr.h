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

#include <sstream>

////////////////////////////////////////////////////////////////////////////////
//
// Note: Please #define _DEBUG for additional debug safety checking
//
////////////////////////////////////////////////////////////////////////////////

#include "error_check.h"

template<class T>
struct PitchedPtr_t
{
   T* ptr;
   std::size_t pitch;
   std::size_t x;
   std::size_t y;
   std::size_t widthBytes;
};

template<class T>
#ifdef __CUDACC__
__host__
__device__
#endif
T& resolvePitchedPtr(PitchedPtr_t<T> &p, std::size_t x, std::size_t y)
{
#ifdef _DEBUG
   if(p.x <= x || p.y <= y)
   {
      printf("ERROR: resolvePitchedPtr attempted to resolve x=%u y=%u when size_x=%u and size_y=%u\n",
         x, y, p.x, p.y);
   }
#endif
   return reinterpret_cast<T*>(&reinterpret_cast<char*>(p.ptr)[y * p.pitch])[x];
}

template<class T>
#ifdef __CUDACC__
__host__
__device__
#endif
T const & resolvePitchedPtr(const PitchedPtr_t<T> &p, std::size_t x, std::size_t y)
{
#ifdef _DEBUG
   if(p.x <= x || p.y <= y)
   {
      printf("ERROR: resolvePitchedPtr(const) attempted to resolve x=%u y=%u when size_x=%u and size_y=%u\n",
         x, y, p.x, p.y);
   }
#endif
   return reinterpret_cast<T*>(&reinterpret_cast<char*>(p.ptr)[y * p.pitch])[x];
}

class CudaPitchedAllocator
{
public:
   static CudaPitchedAllocator& getRef();

   template<class T>
   void allocate(PitchedPtr_t<T> &ptr, std::size_t sizeX, std::size_t sizeY);
   template<class T>
   void free(T* m);
   template<class T>
   void free(PitchedPtr_t<T> &m);

private:
   static CudaPitchedAllocator *m_ref;
};

template<class T>
void CudaPitchedAllocator::allocate(PitchedPtr_t<T> &ptr, size_t sizeX, size_t sizeY)
{
   ptr.widthBytes = sizeX * sizeof(T);
   checkCuda(cudaMallocPitch(reinterpret_cast<void**>(&ptr.ptr), &ptr.pitch, ptr.widthBytes, sizeY));
   ptr.x = sizeX;
   ptr.y = sizeY;
}

template<class T>
void CudaPitchedAllocator::free(T *m)
{
   cudaFree(m);
}

template<class T>
void CudaPitchedAllocator::free(PitchedPtr_t<T> &m)
{
   cudaFree(m.ptr);
}

template<class T>
class HostMem;

template<class T, class Allocator=CudaPitchedAllocator>
class PitchedPtr
{
public:
   PitchedPtr(std::size_t x, std::size_t y);
   template<class rhsAlloc>
   PitchedPtr(const PitchedPtr<T, rhsAlloc> &rhs);
   ~PitchedPtr();

   void memset(int val);
   std::size_t getX() const;
   std::size_t getY() const;
   std::size_t getPitch() const;
   std::size_t getWidthBytes() const;

   PitchedPtr_t<T>& getPtr();
   const PitchedPtr_t<T>& getPtr() const;

   template<class rhsAlloc>
   const PitchedPtr<T, Allocator>& operator=(const PitchedPtr<T, rhsAlloc> &rhs);

   friend class HostMem<T>;

private:
   PitchedPtr_t<T> m_ptr;

   void free();
   void alloc(std::size_t x, std::size_t y);
   void resize(std::size_t x, std::size_t y);
   template<class rhsAlloc>
   void copy(const PitchedPtr<T, rhsAlloc> &rhs);
};

template<class T, class Allocator>
PitchedPtr<T, Allocator>::PitchedPtr(std::size_t x, std::size_t y)
{
   alloc(x, y);
}

template<class T, class Allocator>
template<class rhsAlloc>
PitchedPtr<T, Allocator>::PitchedPtr(const PitchedPtr<T, rhsAlloc> &rhs)
{
   alloc(rhs.m_ptr.x, rhs.m_ptr.y);
   copy(rhs);
}

template<class T, class Allocator>
PitchedPtr<T, Allocator>::~PitchedPtr()
{
   free();
}

template<class T, class Allocator>
void PitchedPtr<T, Allocator>::memset(int val)
{
   cudaMemset2D(m_ptr.ptr, m_ptr.pitch, val, m_ptr.widthBytes, m_ptr.y);
}

template<class T, class Allocator>
std::size_t PitchedPtr<T, Allocator>::getX() const
{
   return m_ptr.x;
}

template<class T, class Allocator>
std::size_t PitchedPtr<T, Allocator>::getY() const
{
   return m_ptr.y;
}

template<class T, class Allocator>
std::size_t PitchedPtr<T, Allocator>::getPitch() const
{
   return m_ptr.pitch;
}

template<class T, class Allocator>
std::size_t PitchedPtr<T, Allocator>::getWidthBytes() const
{
   return m_ptr.widthBytes;
}

template<class T, class Allocator>
PitchedPtr_t<T>& PitchedPtr<T, Allocator>::getPtr()
{
   return m_ptr;
}

template<class T, class Allocator>
const PitchedPtr_t<T>& PitchedPtr<T, Allocator>::getPtr() const
{
   return m_ptr;
}

template<class T, class Allocator>
template<class rhsAlloc>
const PitchedPtr<T, Allocator>& PitchedPtr<T, Allocator>::operator=(const PitchedPtr<T, rhsAlloc> &rhs)
{
   resize(rhs.m_ptr.x, rhs.m_ptr.y);
   copy(rhs);
   
   return *this;
}

template<class T, class Allocator>
void PitchedPtr<T, Allocator>::free()
{
   Allocator::getRef().free(m_ptr);
}

template<class T, class Allocator>
void PitchedPtr<T, Allocator>::alloc(std::size_t x, std::size_t y)
{
   Allocator::getRef().allocate(m_ptr, x, y);

#ifdef _DEBUG
   checkCuda(cudaMemset2D(m_ptr.ptr, m_ptr.pitch, 0, m_ptr.widthBytes, m_ptr.y));
#endif
}

template<class T, class Allocator>
void PitchedPtr<T, Allocator>::resize(std::size_t x, std::size_t y)
{
   if(m_ptr.x != x || m_ptr.y != y)
   {
      free();
      alloc(x, y);
   }
}

template<class T, class Allocator>
template<class rhsAlloc>
void PitchedPtr<T, Allocator>::copy(const PitchedPtr<T, rhsAlloc> &rhs)
{
   assert(m_ptr.ptr != m_ptr.ptr);
   assert(m_ptr.x == rhs.m_ptr.x);
   assert(m_ptr.y == rhs.m_ptr.y);
   assert(m_ptr.widthBytes == rhs.m_ptr.widthBytes);
   checkCuda(cudaMemcpy2D(m_ptr.ptr, m_ptr.pitch, 
             rhs.m_ptr.ptr, rhs.m_ptr.pitch, 
             rhs.m_ptr.widthBytes, rhs.m_ptr.y, 
             cudaMemcpyDeviceToDevice));
}
