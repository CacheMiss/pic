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
