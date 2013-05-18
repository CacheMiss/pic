#pragma once

#include <algorithm>
#include <cuda_runtime_api.h>
#include <list>
#include <map>
#include <sstream>

#ifdef _DEBUG
#include <iostream>
#endif

#include "error_check.h"

class ParticleAllocator
{
public:
   static ParticleAllocator& getRef();

   template<class T>
   void allocate(T* &ptr, std::size_t nElements);
   template<class T>
   void free(T* m);

private:
   ParticleAllocator();
   ~ParticleAllocator();

   static ParticleAllocator *m_ref;

   typedef std::list<std::pair<void*, std::size_t> > MemList_t;
   typedef std::map<void*, std::size_t> MemMap_t;
   MemList_t m_freePool;
   MemMap_t m_usedPool;
   std::size_t m_maxNumConcurrentAlloc;
};

template<class T>
void ParticleAllocator::allocate(T* &ptr, std::size_t nElements)
{
   std::size_t requiredBytes = sizeof(T) * nElements;
   MemList_t::iterator bestFit = m_freePool.end();
   MemList_t::iterator largestAllocation = m_freePool.begin();
   for(MemList_t::iterator i = m_freePool.begin(); i != m_freePool.end(); i++)
   {
      if((*i).second > (*largestAllocation).second)
      {
         largestAllocation = i;
      }
      if((*i).second >= requiredBytes)
      {
         // I have at least one block of memory which is large enough
         if(bestFit == m_freePool.end())
         {
            bestFit = i;
         }
         // I don't want to settle for just any block. I won't ever have much allocated,
         // so search them all to find the best one
         else if((*i).second < (*bestFit).second)
         {
            bestFit = i;
         }
      }
   }

   // I've found memory and don't need to allocate anything else
   if(bestFit != m_freePool.end())
   {
      m_usedPool.insert(*bestFit);
      ptr = reinterpret_cast<T*>((*bestFit).first);
      m_freePool.erase(bestFit);
   }
   else
   {
      // I don't really want to trend all my allocations to be too large. Always getting rid of the largest one that I
      // dont' use gives me the chance to keep small allocations for when they're necessary.
      if(!m_freePool.empty() && m_usedPool.size() + m_freePool.size() >= m_maxNumConcurrentAlloc)
      {
         checkCuda(cudaFree((*largestAllocation).first));
         m_freePool.erase(largestAllocation);
      }
      const std::size_t ONE_MB = 1048576;
      // Always allocate a megabyte. If you don't, the cuda allocator from 2013/05/17 will.
      // Pad the allocation if it's over 1MB
      std::size_t allocatedBytes = requiredBytes > ONE_MB ? requiredBytes + ONE_MB : ONE_MB;
      checkCuda(cudaMalloc(&ptr, allocatedBytes));
      m_usedPool.insert(std::make_pair(ptr, allocatedBytes));
      m_maxNumConcurrentAlloc = std::max(m_maxNumConcurrentAlloc, m_usedPool.size());

#ifdef _DEBUG
      std::cout << "ParticleAllocator allocating " << (double)allocatedBytes / 1048576 << " megabytes." << std::endl;
      std::cout << "There are " << m_freePool.size() + m_usedPool.size() << " blocks allocated." << std::endl;
#endif
   }
}

template<class T>
void ParticleAllocator::free(T* m)
{
   if(NULL == m)
   {
      return;
   }

   MemMap_t::iterator i;
   i = m_usedPool.find(m);
   if(i != m_usedPool.end())
   {
      m_freePool.push_back(*i);
      m_usedPool.erase(i);
   }
   else
   {
      std::stringstream s;
      s << "DevMemReuse was asked to free memory it didn't allocate";
      throw std::runtime_error(s.str());
   }
}
