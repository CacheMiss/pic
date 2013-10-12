#pragma once

#include <boost/thread.hpp>
#include <boost/thread/locks.hpp>
#include <deque>
#include <set>
#include <utility>

#include "dev_stream.h"
#include "host_mem.h"

class SortThread
{
public:
   SortThread();
   void run();
   void join();
   void sortAsync(DevMem<float2> &devPos, DevMem<float3> &devVel, std::size_t numPart);
   void waitForSort(DevMem<float2> &devPos, DevMem<float3> &devVel);
   struct ParticleList;
private:
   void readMain();
   void writeMain();
   void sortMain();
   //boost::mutex *m_stackLock;
   //boost::condition_variable *m_cond;
   boost::thread *m_readThread;
   boost::thread *m_writeThread;
   boost::thread *m_sortThread;

   struct CopyRequest
   {
      CopyRequest()
         :pos(NULL), vel(NULL), numPart(0)
      {}
      CopyRequest(float2* p, float3* v, std::size_t n)
         :pos(p), vel(v), numPart(n)
      {}
      float2* pos;
      float3* vel;
      std::size_t numPart;
   };

   struct Job
   {
      Job()
        : pos(NULL), vel(NULL), d_srcPos(NULL), d_srcVel(NULL)
      {}
      Job(HostMem<float2> *p, HostMem<float3> *v, 
          float2* d_p, float3* d_v,
          std::size_t n)
        : pos(p), vel(v), d_srcPos(d_p), d_srcVel(d_v), numPart(n)
      {}
      Job& operator=(const CopyRequest &rhs)
      {
         d_srcPos = rhs.pos;
         d_srcVel = rhs.vel;
         numPart = rhs.numPart;
         pos->resize(numPart);
         vel->resize(numPart);

         return *this;
      }
      Job& operator=(Job const& rhs)
      {
         if(this != &rhs)
         {
            d_srcPos = rhs.d_srcPos;
            d_srcVel = rhs.d_srcVel;
            numPart = rhs.numPart;
            pos = rhs.pos;
            vel = rhs.vel;
         }
         return *this;
      }

      HostMem<float2> *pos;
      HostMem<float3> *vel;
      float2* d_srcPos;
      float3* d_srcVel;
      std::size_t numPart;
   };

   template<class T>
   class SafeDeque
   {
   public:
      void push_back(T val)
      {
         boost::unique_lock<boost::mutex> l(m_mutex);
         m_deque.push_back(val);
         m_cond.notify_one();
      }
      const T& front() const
      {
         boost::unique_lock<boost::mutex> l(m_mutex);
         while(m_deque.empty())
         {
            m_cond.wait(l);
         }
         return m_deque.front();
      }
      T& front()
      {
         boost::unique_lock<boost::mutex> l(m_mutex);
         while(m_deque.empty())
         {
            m_cond.wait(l);
         }
         return m_deque.front();
      }
      void pop_front()
      {
         boost::unique_lock<boost::mutex> l(m_mutex);
         while(m_deque.empty())
         {
            m_cond.wait(l);
         }
         m_deque.pop_front();
      }
      T atomicPop()
      {
         boost::unique_lock<boost::mutex> l(m_mutex);
         while(m_deque.empty())
         {
            m_cond.wait(l);
         }
         T ret = m_deque.front();
         m_deque.pop_front();
         return ret;
      }
   private:
      std::deque<T> m_deque;
      mutable boost::mutex m_mutex;
      mutable boost::condition_variable m_cond;
   };

   std::size_t m_padding;
   HostMem<float2> m_pos[2];
   HostMem<float3> m_vel[2];

   SafeDeque<Job> m_memPool;
   SafeDeque<Job> m_writeJobs;
   SafeDeque<Job> m_sortJobs;
   SafeDeque<CopyRequest> m_newRequests;

   DevStream m_readStream;
   DevStream m_writeStream;

   std::set<float2*> m_finishedSet;
   boost::mutex m_finishedSetLock;
   boost::condition_variable m_finishedSetCond;

   bool m_keepRunning;
};