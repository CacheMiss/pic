////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014, Stephen C. Sewell
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <boost/thread.hpp>
#include <boost/thread/locks.hpp>
#include <deque>
#include <map>
#include <utility>

#include "dev_stream.h"
#include "host_mem.h"

class SortThread
{
public:
   SortThread(float oobValue);
   void run();
   void join();
   void setNumSortThreads(std::size_t n); // Default 1
   void enableParticleElimination(); // Default
   void disableParticleElimination();
   void sortAsync(DevMem<float2> &devPos, DevMem<float3> &devVel, std::size_t numPart);
   // Wait for a sort to finish. Returns the number of particles which were out of bounds.
   std::size_t waitForSort(DevMem<float2> &devPos, DevMem<float3> &devVel);

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
        : part(NULL), d_srcPos(NULL), d_srcVel(NULL), numOob(0)
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
         numOob = 0;
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
      // Num out of bounds particles
      std::size_t numOob;
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

private:
   void readMain();
   void writeMain();
   void sortMain();
   boost::thread *m_readThread;
   boost::thread *m_writeThread;
   std::vector<boost::thread*> m_sortThread;
   std::size_t m_numSortThreads;
   std::size_t m_padding;
   bool m_enableParticleElimination;
   float m_oobValue;
   HostMem<Particle> m_part[3];

   SafeDeque<Job> m_memPool;
   SafeDeque<Job> m_writeJobs;
   SafeDeque<Job> m_sortJobs;
   SafeDeque<CopyRequest> m_newRequests;

   DevStream m_readStream;
   DevStream m_writeStream;

   typedef std::map<float2*, std::size_t> FinishedCopies;
   FinishedCopies m_finishedCopies;
   boost::mutex m_finishedCopiesLock;
   boost::condition_variable m_finishedCopiesCond;

   bool m_keepRunning;
};