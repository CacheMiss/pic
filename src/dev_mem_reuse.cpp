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

PitchedPtr<void> DevMemReuse::getBlock()
{
   PitchedPtr<void> p;
   if(m_memPool.empty())
   {
#ifdef _DEBUG
      std::cout << "DevMemReuse is allocating a block." << std::endl;
#endif
      checkCuda(cudaMallocPitch(&p.ptr, &p.pitch, m_sizeX * m_dataSize, m_sizeY));
      p.width = m_sizeX;
      p.height = m_sizeY;
      p.widthBytes = p.width * m_dataSize;
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
