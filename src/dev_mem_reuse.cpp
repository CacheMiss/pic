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
#include "dev_mem_reuse.h"

#include <cuda_runtime_api.h>
#include <cstddef>

#ifdef _DEBUG
#include <iostream>
#endif

DevMemReuse* DevMemReuse::m_ref = NULL;

DevMemReuse::DevMemReuse()
: m_sizeX(0)
, m_sizeY(0)
, m_dataSize(sizeof(void*))
{
#ifdef _DEBUG
   std::cout << "DevMemReuse initializing with dataSize of " 
             << m_dataSize << " bytes" << std::endl;
#endif
}

DevMemReuse& DevMemReuse::getRef()
{
   if(m_ref == NULL)
   {
      m_ref = new DevMemReuse();
   }

   return *m_ref;
}

std::size_t DevMemReuse::getSizeX() const
{
   return m_sizeX;
}

std::size_t DevMemReuse::getSizeY() const
{
   return m_sizeY;
}

std::size_t DevMemReuse::getDataSize() const
{
   return m_dataSize;
}

void DevMemReuse::setSizeX(std::size_t s)
{
   m_sizeX = s;
}

void DevMemReuse::setSizeY(std::size_t s)
{
   m_sizeY = s;
}

void DevMemReuse::setDataSize(std::size_t s)
{
   m_dataSize = s;
}

PitchedPtr_t<void> DevMemReuse::getBlock()
{
   PitchedPtr_t<void> p;
   if(m_memPool.empty())
   {
#ifdef _DEBUG
      std::cout << "DevMemReuse is allocating a block." << std::endl;
#endif
      checkCuda(cudaMallocPitch(&p.ptr, &p.pitch, m_sizeX * m_dataSize, m_sizeY));
      p.x = m_sizeX;
      p.y = m_sizeY;
      p.widthBytes = p.x * m_dataSize;
   }
   else
   {
      p = m_memPool.front();
      m_memPool.pop_front();
   }
   m_usedPool.insert(std::make_pair(p.ptr, p));

   return p;
}

void DevMemReuse::cleanup()
{
   for(MemPool_t::iterator i = m_memPool.begin();
       i != m_memPool.end(); i++)
   {
      cudaFree((*i).ptr);
   }

   for(UsedPool_t::iterator i = m_usedPool.begin();
       i != m_usedPool.end(); i++)
   {
      cudaFree((*i).first);
   }
}
