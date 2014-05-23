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
#include "host_mem.h"
#include "sort_thread.h"

#include <algorithm>
#include <boost/iterator/iterator_facade.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

SortThread::SortThread(float oobValue)
: m_readThread(NULL)
, m_writeThread(NULL)
, m_numSortThreads(1)
, m_keepRunning(false)
, m_oobValue(oobValue)
, m_padding(100000)
, m_enableParticleElimination(true)
{
   m_sortThread.resize(m_numSortThreads);
   memset(&m_sortThread[0], 0, sizeof(m_sortThread[0])* m_sortThread.size());
   m_part[0].setPadding(m_padding);
   m_part[1].setPadding(m_padding);
   m_part[2].setPadding(m_padding);
   m_memPool.push_back(Job(&m_part[0], NULL, NULL, 0));
   m_memPool.push_back(Job(&m_part[1], NULL, NULL, 0));
   m_memPool.push_back(Job(&m_part[2], NULL, NULL, 0));
}

void SortThread::setNumSortThreads(std::size_t n)
{
   m_sortThread.resize(m_numSortThreads);
   memset(&m_sortThread[0], 0, sizeof(m_sortThread[0])* m_sortThread.size());
}

void SortThread::enableParticleElimination()
{
   m_enableParticleElimination = true;
}

void SortThread::disableParticleElimination()
{
   m_enableParticleElimination = false;
}

// Create all the threads necessary for this class
void SortThread::run()
{
   m_keepRunning = true;
   if(m_readThread == NULL)
   {
      m_readThread = new boost::thread(&SortThread::readMain, this);
   }
   if(m_writeThread == NULL)
   {
      m_writeThread = new boost::thread(&SortThread::writeMain, this);
   }
   for(std::size_t i = 0; i < m_sortThread.size(); i++)
   {
      if(m_sortThread[i] == NULL)
      {
         m_sortThread[i] = new boost::thread(&SortThread::sortMain, this);
      }
   }
}

// Halt and join all the threads in this class
void SortThread::join()
{
   Job emptyJob;
   CopyRequest emptyRequest;
   m_keepRunning = false;
   if(m_readThread != NULL)
   {
      m_newRequests.push_back(emptyRequest);
      m_readThread->join();
      delete m_readThread;
      m_readThread = NULL;
   }
   if(m_writeThread != NULL)
   {
      m_writeJobs.push_back(emptyJob);
      m_writeThread->join();
      delete m_writeThread;
      m_writeThread = NULL;
   }
   for(std::size_t i = 0; i < m_sortThread.size(); i++)
   {
      if(m_sortThread[i] != NULL)
      {
         m_sortJobs.push_back(emptyJob);
      }
   }
   for(std::size_t i = 0; i < m_sortThread.size(); i++)
   {
      if(m_sortThread[i] != NULL)
      {
         m_sortJobs.push_back(emptyJob);
         m_sortThread[i]->join();
         delete m_sortThread[i];
         m_sortThread[i] = NULL;
      }
   }
}

// This thread reads data from the device if there is any and queues it
// up to be sorted by the sorting thread
void SortThread::readMain()
{
   while(m_keepRunning)
   {
      // Get the copy request
      CopyRequest request;
      request = m_newRequests.atomicPop();
      // I use this to bail out if I need to join
      if(request.d_pos == NULL && request.d_vel == NULL)
      {
         continue;
      }

      // When a unit in mem pool is free, secure it
      Job newJob = m_memPool.atomicPop();
      // I use this to bail out if I need to join
      if(newJob.part == NULL &&
         newJob.d_srcPos == NULL && newJob.d_srcVel == NULL)
      {
         continue;
      }
      newJob = request;
      checkCuda(cudaMemcpy2DAsync(&(*newJob.part)[0].pos, sizeof(Particle),
                newJob.d_srcPos, sizeof(float2),
                sizeof(float2),
                newJob.numPart, 
                cudaMemcpyDeviceToHost, 
                *m_readStream));
      checkCuda(cudaMemcpy2DAsync(&(*newJob.part)[0].vel, sizeof(Particle),
                newJob.d_srcVel, sizeof(float3),
                sizeof(float3),
                newJob.numPart, 
                cudaMemcpyDeviceToHost, 
                *m_readStream));
      m_readStream.synchronize();
      m_sortJobs.push_back(newJob);
   }
}

// This thread writes sorted data back to the device as it becomes available
void SortThread::writeMain()
{
   while(m_keepRunning)
   {
      Job newJob = m_writeJobs.atomicPop();
      if(newJob.part == NULL &&
         newJob.d_srcPos == NULL && newJob.d_srcVel == NULL)
      {
         continue;
      }
      checkCuda(cudaMemcpy2DAsync(newJob.d_srcPos, sizeof(float2),
                &(*newJob.part)[0].pos, sizeof(Particle),
                sizeof(float2),
                newJob.numPart, 
                cudaMemcpyHostToDevice, 
                *m_writeStream));
      checkCuda(cudaMemcpy2DAsync(newJob.d_srcVel, sizeof(float3),
                &(*newJob.part)[0].vel, sizeof(Particle),
                sizeof(float3),
                newJob.numPart, 
                cudaMemcpyHostToDevice, 
                *m_writeStream));
      std::size_t numOob = 0;
      if(m_enableParticleElimination)
      {
         for(int i = static_cast<int>(newJob.part->size()) - 1; i >= 0; i--)
         {
            if((*newJob.part)[i].pos.y != m_oobValue)
            {
               numOob = (newJob.part->size() - 1) - i;
               break;
            }
         }
      }
      m_writeStream.synchronize();
      {
         boost::unique_lock<boost::mutex> lock(m_finishedCopiesLock);
         m_finishedCopies.insert(std::make_pair(newJob.d_srcPos, numOob));
         m_finishedCopiesCond.notify_one();
      }
      m_memPool.push_back(newJob);
   }
}

// This thread sorts the data acquired by the read thread and passes the
// sorted data on to the write thread
void SortThread::sortMain()
{
   while(m_keepRunning)
   {
      Job newJob = m_sortJobs.atomicPop();
      if(newJob.part == NULL &&
         newJob.d_srcPos == NULL && newJob.d_srcVel == NULL)
      {
         continue;
      }

      // Sort
      tbb::parallel_sort(newJob.part->begin(), newJob.part->end());

      // Write results
      m_writeJobs.push_back(newJob);
   }
}

void SortThread::sortAsync(DevMem<float2> &devPos, DevMem<float3> &devVel, std::size_t numPart)
{
   m_newRequests.push_back(CopyRequest(devPos.getPtr(), devVel.getPtr(), numPart));
}

std::size_t SortThread::waitForSort(DevMem<float2> &devPos, DevMem<float3> &devVel)
{
   FinishedCopies::iterator it;
   boost::unique_lock<boost::mutex> lock(m_finishedCopiesLock);
   do
   {
      it = m_finishedCopies.find(devPos.getPtr());
      if(it != m_finishedCopies.end())
      {
         break;
      }
      m_finishedCopiesCond.wait(lock);
   }while(it == m_finishedCopies.end());
   std::size_t numOob = it->second;
   m_finishedCopies.erase(it);

   return numOob;
}
