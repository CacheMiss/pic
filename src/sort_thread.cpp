#include "host_mem.h"
#include "sort_thread.h"

#include <algorithm>
#include <boost/iterator/iterator_facade.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

SortThread::SortThread()
: m_readThread(NULL)
, m_writeThread(NULL)
, m_numSortThreads(2)
, m_keepRunning(false)
, m_padding(100000)
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
   if(m_readThread == NULL)
   {
      m_keepRunning = false;
      m_readThread->join();
      delete m_readThread;
      m_readThread = NULL;
   }
   if(m_writeThread == NULL)
   {
      m_keepRunning = false;
      m_writeThread->join();
      delete m_writeThread;
      m_writeThread = NULL;
   }
   for(std::size_t i = 0; i < m_sortThread.size(); i++)
   {
      if(m_sortThread[i] == NULL)
      {
         m_keepRunning = false;
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
      m_writeStream.synchronize();
      {
         boost::unique_lock<boost::mutex> lock(m_finishedSetLock);
         m_finishedSet.insert(newJob.d_srcPos);
         m_finishedSetCond.notify_one();
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

void SortThread::waitForSort(DevMem<float2> &devPos, DevMem<float3> &devVel)
{
   std::set<float2*>::iterator it;
   boost::unique_lock<boost::mutex> lock(m_finishedSetLock);
   do
   {
      it = m_finishedSet.find(devPos.getPtr());
      if(it != m_finishedSet.end())
      {
         break;
      }
      m_finishedSetCond.wait(lock);
   }while(it == m_finishedSet.end());
   m_finishedSet.erase(it);
}
