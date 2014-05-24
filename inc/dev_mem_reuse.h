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

#include <deque>
#include <map>

#include "pitched_ptr.h"

class DevMemReuse
{
public:
   static DevMemReuse& getRef();
   ~DevMemReuse();

   void setSizeX(std::size_t s);
   void setSizeY(std::size_t s);
   void setDataSize(std::size_t s);
   std::size_t getSizeX() const;
   std::size_t getSizeY() const;
   std::size_t getDataSize() const;

   template<class T>
   void allocate(T* &ptr, std::size_t nElements);
   template<class T>
   void allocatePitched(PitchedPtr_t<T> &ptr, std::size_t sizeX, std::size_t sizeY);
   template<class T>
   void free(T* m);
   template<class T>
   void free(PitchedPtr_t<T> &m);

   void cleanup();

private:
   DevMemReuse();
   static DevMemReuse *m_ref;
   PitchedPtr_t<void> getBlock();

   std::size_t m_sizeX;
   std::size_t m_sizeY;
   std::size_t m_dataSize;
   typedef std::deque<PitchedPtr_t<void> >MemPool_t;
   MemPool_t m_memPool;
   typedef std::map<void*, PitchedPtr_t<void> > UsedPool_t;
   UsedPool_t m_usedPool;
};

template<class T>
void DevMemReuse::allocate(T* &ptr, std::size_t nElements)
{
   if(sizeof(T) > m_dataSize)
   {
      std::stringstream s;
      s << "Attempted to use DevMemReuse to allocate a data type larger than it was configured for.";
      throw std::runtime_error(s.str());
   }
   PitchedPtr_t<void> p = getBlock();

   // If more elements were asked for than have been allocated
   if(nElements > (p.pitch * (p.y -1) + p.widthBytes) / m_dataSize)
   {
      std::stringstream s;
      s << "DevMemReuse was asked to allocate something too large.";
      throw std::runtime_error(s.str());
   }

   ptr = static_cast<T*>(p.ptr);
}

template<class T>
void DevMemReuse::allocatePitched(PitchedPtr_t<T> &ptr, std::size_t sizeX, std::size_t sizeY)
{
   if(sizeof(T) > m_dataSize)
   {
      std::stringstream s;
      s << "Attempted to use DevMemReuse to allocate a data type larger than it was configured for.";
      throw std::runtime_error(s.str());
   }
   PitchedPtr_t<void> p;
   p = getBlock();

   ptr.ptr = reinterpret_cast<T*>(p.ptr);
   ptr.height = sizeY;
   ptr.width = sizeX;
   ptr.widthBytes = sizeX * sizeof(T);
   ptr.pitch = p.pitch;

   if(ptr.y > p.y)
   {
      std::stringstream s;
      s << "DevMemReuse was asked to create a pitched ptr with height " << ptr.height
        << " but it was only configured to create heights of " << m_sizeY;
      throw std::runtime_error(s.str());
   }
   if(ptr.x > p.x)
   {
      std::stringstream s;
      s << "DevMemReuse was asked to create a pitched ptr with width " << ptr.width
        << " but it was only configured to create widths of " << m_sizeX;
      throw std::runtime_error(s.str());
   }
}

template<class T>
void DevMemReuse::free(T* m)
{
   if(NULL == m)
   {
      return;
   }

   UsedPool_t::iterator i;
   i = m_usedPool.find(m);
   if(i != m_usedPool.end())
   {
      m_memPool.push_back((*i).second);
      m_usedPool.erase(i);
   }
   else
   {
      std::stringstream s;
      s << "DevMemReuse was asked to free memory it didn't allocate";
      throw std::runtime_error(s.str());
   }
}

template<class T>
void DevMemReuse::free(PitchedPtr_t<T> &m)
{
   if(NULL == m.ptr)
   {
      return;
   }

   UsedPool_t::iterator i;
   i = m_usedPool.find(m.ptr);
   if(i != m_usedPool.end())
   {
      m_memPool.push_back((*i).second);
      m_usedPool.erase(i);
   }
   else
   {
      std::stringstream s;
      s << "DevMemReuse was asked to free memory it didn't allocate";
      throw std::runtime_error(s.str());
   }
}
