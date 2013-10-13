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
private:
   void readMain();
   void writeMain();
   void sortMain();
   //boost::mutex *m_stackLock;
   //boost::condition_variable *m_cond;
   boost::thread *m_readThread;
   boost::thread *m_writeThread;
   std::vector<boost::thread*> m_sortThread;
   std::size_t m_numSortThreads;

   class Particle
   {
      public:
      Particle()
      {
         pos.x = 0;
         pos.y = 0;
         vel.x = 0;
         vel.y = 0;
         vel.z = 0;
      }
      Particle(float2 p, float3 v)
         : pos(p), vel(v)
      {}
      bool operator<(Particle const& rhs) const
      {
         if(pos.y < rhs.pos.y)
         {
            return true;
         }
         else if(pos.y == rhs.pos.y && pos.x < rhs.pos.x)
         {
            return true;
         }
         return false;
      }
      bool operator==(Particle const& rhs) const
      {
         return pos.y == rhs.pos.y && pos.x == rhs.pos.x;
      }
      float2 pos;
      float3 vel;
   };

   struct CopyRequest
   {
      CopyRequest()
         :d_pos(NULL), d_vel(NULL), numPart(0)
      {}
      CopyRequest(float2* p, float3* v, std::size_t n)
         :d_pos(p), d_vel(v), numPart(n)
      {}
      float2* d_pos;
      float3* d_vel;
      std::size_t numPart;
   };

   struct Job
   {
      Job()
        : part(NULL), d_srcPos(NULL), d_srcVel(NULL)
      {}
      Job(HostMem<Particle> *p,
          float2* d_p, float3* d_v,
          std::size_t n)
        : part(p), d_srcPos(d_p), d_srcVel(d_v), numPart(n)
      {}
      Job& operator=(const CopyRequest &rhs)
      {
         d_srcPos = rhs.d_pos;
         d_srcVel = rhs.d_vel;
         numPart = rhs.numPart;
         assert(part != NULL);
         part->resize(numPart);

         return *this;
      }
      Job& operator=(Job const& rhs)
      {
         if(this != &rhs)
         {
            d_srcPos = rhs.d_srcPos;
            d_srcVel = rhs.d_srcVel;
            numPart = rhs.numPart;
            part = rhs.part;
         }
         return *this;
      }

      HostMem<Particle> *part;
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
   HostMem<Particle> m_part[3];

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