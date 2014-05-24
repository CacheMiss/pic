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
#include "logging_thread.h"

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <stack>
#include <string>

#include "array2d.h"
#include "global_variables.h"
#include "logging_types.h"
#include "pic_utils.h"

struct LoggerThread
{
    LoggerThread(LoggingThread *threadRef) : dataQueue(threadRef) {}

    void operator()()
    {
       LoggingBase *work;

       while(1)
       {
          work = dataQueue->popLogItem();
          if(work == NULL)
          {
             break;
          }
          work->logData();
          delete work;
       }

       return;
    }

    LoggingThread *dataQueue;
};

LoggingThread *LoggingThread::m_ref = NULL;

LoggingThread::LoggingThread()
   :m_jobQueue()
{
   m_stackLock = new boost::mutex;
   m_threadHandle = new boost::thread(LoggerThread(this));
   m_cond = new boost::condition_variable();
   m_condEmpty = new boost::condition_variable();
}

LoggingThread::~LoggingThread()
{
   stopLogging();
   m_threadHandle->join();
   delete m_threadHandle;
   delete m_stackLock;
   delete m_cond;
   delete m_condEmpty;
}

LoggingThread & LoggingThread::getRef()
{
   if(m_ref == NULL)
   {
      m_ref = new LoggingThread();
   }
   return *m_ref;
}


void LoggingThread::logRhoAscii(const int index, const Array2dF *rho,
              const Array2dF *rhoe, const Array2dF *rhoi)
{
   LoggingBase *tmp;
   tmp = new LogRhoAscii(index, rho, rhoe, rhoi);
   pushLogItem(tmp);
}

void LoggingThread::logPhiAscii(const int index, const Array2dF *phi)
{
   LoggingBase *tmp;
   tmp = new LogPhiAscii(index, phi);
   pushLogItem(tmp);
}

void LoggingThread::logRhoBinary(const int index, const Array2dF *rho,
              const Array2dF *rhoe, const Array2dF *rhoi)
{
   LoggingBase *tmp;
   tmp = new LogRhoBinary(index, rho, rhoe, rhoi);
   pushLogItem(tmp);
}

void LoggingThread::logPhiBinary(const int index, const Array2dF *phi)
{
   LoggingBase *tmp;
   tmp = new LogPhiBinary(index, phi);
   pushLogItem(tmp);
}

void LoggingThread::logInfo(unsigned int idx, float sTime, 
                            unsigned int nmElectrons, unsigned int nmIons,
                            bool resumeRun)
{
   LogInfo *tmp = new LogInfo(idx, sTime, nmElectrons, nmIons, resumeRun);
   pushLogItem(tmp);
}

void LoggingThread::pushLogItem(LoggingBase *item)
{
   boost::unique_lock<boost::mutex> lock(*m_stackLock);
   m_jobQueue.push(item);
   m_cond->notify_one();
}

LoggingBase* LoggingThread::popLogItem()
{
   boost::unique_lock<boost::mutex> lock(*m_stackLock);
   LoggingBase *tmp;

   while(m_jobQueue.empty())
   {
       m_cond->wait(lock);
   }

   tmp = m_jobQueue.front();
   m_jobQueue.pop();
   
   if(m_jobQueue.empty())
   {
      m_condEmpty->notify_all();
   }

   return tmp;
}

void LoggingThread::flush()
{
   pushLogItem(new FlushLog());
   boost::unique_lock<boost::mutex> lock(*m_stackLock);
   while(!m_jobQueue.empty())
   {
       m_condEmpty->wait(lock);
   }
}

void LoggingThread::stopLogging()
{
   boost::mutex::scoped_lock lock(*m_stackLock);
   m_jobQueue.push(NULL);
   m_cond->notify_one();
}
