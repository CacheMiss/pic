#include "host_mem.h"
#include "sort_thread.h"

#include <algorithm>
#include <boost/iterator/iterator_facade.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

struct SortThread::ParticleList
{
public:
   class Particle
   {
   public:
      Particle()
         : m_pos(NULL), m_vel(NULL)
      {}
      Particle(float2* pos, float3* vel)
         : m_pos(pos), m_vel(vel)
      {}
      Particle(const Particle& rhs)
      {
         std::cout << "Copy Constructor called!" << std::endl;
      }
      bool operator<(Particle const& rhs) const
      {
         if(m_pos->y < rhs.m_pos->y)
         {
            return true;
         }
         else if(m_pos->y == rhs.m_pos->y && m_pos->x < rhs.m_pos->x)
         {
            return true;
         }
         return false;
      }
      bool operator==(Particle const& rhs) const
      {
         return m_pos->y == rhs.m_pos->y && m_pos->x == rhs.m_pos->x;
      }
      Particle& operator=(Particle const &rhs)
      {
         if(&rhs != this)
         {
            *m_pos = *rhs.m_pos;
            *m_vel = *rhs.m_vel;
         }
         return *this;
      }
      static void swap(Particle& a, Particle& b)
      {
         std::swap(*a.m_pos, *b.m_pos);
         std::swap(*a.m_vel, *b.m_vel);
      }

      template<class T>
      friend void std::swap(T& a, T&b);
      float2* m_pos;
      float3* m_vel;
   };

   class ParticleLess
   {
   public:
      bool operator()(const Particle &lhs, const Particle &rhs) const
      {
         std::cout << "Hi" << std::endl;
         return lhs < rhs;
      }
   };

   class PartVectorInitializer
   {
   public:
      PartVectorInitializer(std::vector<Particle> &vec, float2* pos, float3* vel)
         : m_vec(&vec), m_pos(pos), m_vel(vel)
      {}
      void operator()(const tbb::blocked_range<std::size_t>& range) const
      {
         for(std::size_t i = range.begin(); i != range.end(); ++i)
         {
            (*m_vec)[i].m_pos = m_pos+i;
            (*m_vec)[i].m_vel = m_vel+i;
         }
      }
   private:
      std::vector<Particle> *m_vec;
      float2* m_pos;
      float3* m_vel;
   };

   private:
   template<class Value>
   class iterator_template
      : public boost::iterator_facade<
         iterator_template<Value>, 
         Value, 
         boost::random_access_traversal_tag 
         >
   {
      public:
      iterator_template() : m_data(NULL){}
      explicit iterator_template(Value* data)
         :m_data(data){}

      template<class OtherValue>
      iterator_template(iterator_template<OtherValue> const& other)
         :m_data(other.m_data)
      {}

      Value& operator[](std::size_t n) const
      {
         return m_data[n];
      }

      private:
      friend class boost::iterator_core_access;
      template <class> friend class iterator_template;

      Value& dereference() const
      {
         return *m_data;
      }
      template<class OtherValue>
      bool equal(iterator_template<OtherValue> const& i) const
      {
         return m_data == i.m_data;
      }
      void increment()
      {
         m_data++;
      }
      void decrement()
      {
         m_data--;
      }
      void advance(std::size_t n)
      {
         m_data += n;
      }
      std::ptrdiff_t distance_to(iterator_template<Value> const& j) const
      {
         return j.m_data - m_data;
      }

      Value* m_data;
   };
   public:
   typedef iterator_template<Particle> iterator;
   typedef iterator_template<Particle const> const_iterator;

   ParticleList()
      : m_pos(NULL), m_vel(NULL), m_particles(), m_numPart(0)
   {}

   void initialize(HostMem<float2>& pos, HostMem<float3>& vel, std::size_t numPart)
   {
      m_pos = &pos;
      m_vel = &vel;
      m_numPart = numPart;
      m_particles.resize(m_numPart);

      PartVectorInitializer initializer(m_particles, &(*m_pos)[0], &(*m_vel)[0]);
      tbb::parallel_for(tbb::blocked_range<std::size_t>(0, m_numPart), initializer);
   }

   iterator begin()
   {
      return iterator(&m_particles[0]);
   }
   iterator end()
   {
      return iterator(&m_particles[0] + m_numPart);
   }
   template<class T> friend void std::swap(T& a, T&b);

   Particle& operator[](std::size_t n)
   {
      return m_particles[n];
   }
   const Particle& operator[](std::size_t n) const
   {
      return m_particles[n];
   }

   private:
   HostMem<float2>* m_pos;
   HostMem<float3>* m_vel;
   std::vector<Particle> m_particles;
   std::size_t m_numPart;

};

template<>
void std::swap(SortThread::ParticleList::Particle& a, SortThread::ParticleList::Particle& b)
{
   SortThread::ParticleList::Particle::swap(a, b);
}

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
   m_memPool.push_back(Job(&m_pos[0], &m_vel[0], NULL, NULL, 0));
   m_memPool.push_back(Job(&m_pos[1], &m_vel[1], NULL, NULL, 0));
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
   if(m_sortThread == NULL)
   {
      m_sortThread = new boost::thread(&SortThread::sortMain, this);
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
   if(m_sortThread == NULL)
   {
      m_keepRunning = false;
      m_sortThread->join();
      delete m_sortThread;
      m_sortThread = NULL;
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
      if(request.pos == NULL && request.vel == NULL)
      {
         continue;
      }

      // When a unit in mem pool is free, secure it
      Job newJob = m_memPool.atomicPop();
      // I use this to bail out if I need to join
      if(newJob.pos == NULL && newJob.vel == NULL &&
         newJob.d_srcPos == NULL && newJob.d_srcVel == NULL)
      {
         continue;
      }
      newJob = request;
      checkCuda(cudaMemcpyAsync(&(*newJob.pos)[0], 
                newJob.d_srcPos, 
                sizeof(float2) * newJob.numPart, 
                cudaMemcpyDeviceToHost, 
                *m_readStream));
      checkCuda(cudaMemcpyAsync(&(*newJob.vel)[0], 
                newJob.d_srcVel, 
                sizeof(float3) * newJob.numPart, 
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
      if(newJob.pos == NULL && newJob.vel == NULL &&
         newJob.d_srcPos == NULL && newJob.d_srcVel == NULL)
      {
         continue;
      }
      checkCuda(cudaMemcpyAsync(newJob.d_srcPos, 
                &(*newJob.pos)[0], 
                sizeof(float2) * newJob.numPart, 
                cudaMemcpyHostToDevice, 
                *m_writeStream));
      checkCuda(cudaMemcpyAsync(newJob.d_srcVel, 
                &(*newJob.vel)[0], 
                sizeof(float3) * newJob.numPart, 
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

// This thread sorts the data acquired by the read thread and passes the
// sorted data on to the write thread
void SortThread::sortMain()
{
   ParticleList pList;
   while(m_keepRunning)
   {
      Job newJob = m_sortJobs.atomicPop();
      if(newJob.pos == NULL && newJob.vel == NULL &&
         newJob.d_srcPos == NULL && newJob.d_srcVel == NULL)
      {
         continue;
      }

      // Sort
      newJob.numPart = 4;
      pList.initialize(*newJob.pos, *newJob.vel, newJob.numPart);
      for(std::size_t i = 0; i < newJob.numPart; i++)
      {
         assert(pList[i].m_pos == &(*newJob.pos)[i]);
      }
      ParticleList::iterator b = pList.begin();
      ParticleList::iterator e = pList.end();
      std::cout << "e-b = " << e-b << std::endl;

      std::set<float2*> tmpSet;
      std::pair<std::set<float2*>::iterator, bool> setIt;
      for(b; b != e; b++)
      {
         setIt = tmpSet.insert((*b).m_pos);
         if(setIt.second == false)
         {
            std::cout << "Duplicate Ptr!" << std::endl;
         }
      }

      tbb::parallel_sort(pList.begin(), pList.end());

      for(std::size_t i = 0; i < std::min(static_cast<std::size_t>(20), newJob.numPart); i++)
      {
         std::cout << "y=" << (*newJob.pos)[i].y << " x=" << (*newJob.pos)[i].x << std::endl;
         assert(pList[i].m_pos == &(*newJob.pos)[i]);
         if(i > 0)
         {
            assert(pList[i-1] < pList[i] || pList[i-1] == pList[i]);
         }
      }

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
   boost::unique_lock<boost::mutex> lock(m_finishedSetLock);
   std::set<float2*>::iterator it;
   do
   {
      it = m_finishedSet.find(devPos.getPtr());
      m_finishedSetCond.wait(lock);
   }while(it == m_finishedSet.end());
   m_finishedSet.erase(it);
}
