#include "sort_thread.h"

SortThread::SortThread()
: m_readThread(NULL)
, m_writeThread(NULL)
, m_sortThread(NULL)
, m_keepRunning(false)
, m_padding(100000)
{
   m_pos[0].setPadding(m_padding);
   m_vel[0].setPadding(m_padding);
   m_pos[1].setPadding(m_padding);
   m_vel[1].setPadding(m_padding);
   m_memPool.push_back(Job(&m_pos[0], &m_vel[0], NULL, NULL));
   m_memPool.push_back(Job(&m_pos[1], &m_vel[2], NULL, NULL));
}

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
   if(m_readThread == NULL)
   {
      m_sortThread = new boost::thread(&SortThread::sortMain, this);
   }
}

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
   if(m_sortThread == NULL)
   {
      m_keepRunning = false;
      m_sortThread->join();
      delete m_sortThread;
      m_sortThread = NULL;
   }
}

void SortThread::readMain()
{
   while(m_keepRunning)
   {
      std::pair<DevMem<float2>*, DevMem<float3>*> src;
      src = m_newRequests.atomicPop();
      // I use this to bail out if I need to join
      if(src.first == NULL && src.second == NULL)
      {
         continue;
      }
      Job newJob = m_memPool.atomicPop();
      if(newJob.pos == NULL && newJob.vel == NULL &&
         newJob.d_srcPos == NULL && newJob.d_srcVel == NULL)
      {
         continue;
      }
      newJob.pos->resize(newJob.d_srcPos->size());
      newJob.pos->resize(newJob.d_srcPos->size());
      checkCuda(cudaMemcpyAsync(&(*newJob.pos)[0], 
                newJob.d_srcPos, 
                sizeof(float2) * newJob.d_srcPos->size(), 
                cudaMemcpyDeviceToHost, 
                *m_readStream));
      checkCuda(cudaMemcpyAsync(&(*newJob.vel)[0], 
                newJob.d_srcVel, 
                sizeof(float3) * newJob.d_srcVel->size(), 
                cudaMemcpyDeviceToHost, 
                *m_readStream));
      m_readStream.synchronize();
      m_sortJobs.push_back(newJob);
   }
}

void SortThread::writeMain()
{
   while(m_keepRunning)
   {
      Job newJob = m_writeJobs.atomicPop();
      if(newJob.pos == NULL && newJob.vel == NULL &&
         newJob.d_srcPos == NULL && newJob.d_srcVel == NULL)
      {
         continue;
      }
      checkCuda(cudaMemcpyAsync(newJob.d_srcPos, 
                &(*newJob.pos)[0], 
                sizeof(float2) * newJob.d_srcPos->size(), 
                cudaMemcpyHostToDevice, 
                *m_writeStream));
      checkCuda(cudaMemcpyAsync(newJob.d_srcVel, 
                &(*newJob.vel)[0], 
                sizeof(float3) * newJob.d_srcVel->size(), 
                cudaMemcpyHostToDevice, 
                *m_writeStream));
      m_writeStream.synchronize();
      m_memPool.push_back(newJob);
      {
         boost::unique_lock<boost::mutex> lock(m_finishedSetLock);
         m_finishedSet.insert(newJob.d_srcPos);
         m_finishedSetCond.notify_one();
      }
   }
}

void SortThread::sortMain()
{
   while(m_keepRunning)
   {
   }
}

void SortThread::sortAsync(DevMem<float2> *devPos, DevMem<float3> *devVel)
{
   m_newRequests.push_back(std::make_pair(devPos, devVel));
}

void SortThread::waitForSort(DevMem<float2> *devPos, DevMem<float3> *devVel)
{
   boost::unique_lock<boost::mutex> lock(m_finishedSetLock);
   std::set<DevMem<float2>*>::iterator it;
   do
   {
      it = m_finishedSet.find(devPos);
      m_finishedSetCond.wait(lock);
   }while(it == m_finishedSet.end());
   m_finishedSet.erase(it);
}