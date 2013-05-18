#include "particle_allocator.h"

ParticleAllocator* ParticleAllocator::m_ref = NULL;

ParticleAllocator::ParticleAllocator()
: m_maxNumConcurrentAlloc(0)
{
}

ParticleAllocator::~ParticleAllocator()
{
   for(MemList_t::iterator i = m_freePool.begin(); i != m_freePool.end(); i++)
   {
      cudaFree((*i).first);
   }
}

ParticleAllocator& ParticleAllocator::getRef()
{
   if(m_ref == NULL)
   {
      m_ref = new ParticleAllocator;
   }

   return *m_ref;
}