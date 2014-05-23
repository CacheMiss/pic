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
#include "particle_allocator.h"

ParticleAllocator* ParticleAllocator::m_ref = NULL;

ParticleAllocator::ParticleAllocator()
: m_maxNumConcurrentAlloc(0)
{
}

ParticleAllocator::~ParticleAllocator()
{
   cleanup();
}

ParticleAllocator& ParticleAllocator::getRef()
{
   if(m_ref == NULL)
   {
      m_ref = new ParticleAllocator;
   }

   return *m_ref;
}

void ParticleAllocator::cleanup()
{
   for(MemList_t::iterator i = m_freePool.begin();
       i != m_freePool.end(); i++)
   {
      cudaFree((*i).first);
   }

   for(MemMap_t::iterator i = m_usedPool.begin();
       i != m_usedPool.end(); i++)
   {
      cudaFree((*i).first);
   }
}
