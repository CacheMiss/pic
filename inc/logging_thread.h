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
#ifndef LOGGING_THREAD_H
#define LOGGING_THREAD_H

//#include <pthread.h>
#include <queue>
#include <stdio.h>

#include "array2d.h"
#include "typedefs.h"

//void* loggerThread(void *ptr);

class LoggingBase;

namespace boost
{
    class thread;
    class mutex;
    class condition_variable;
};

class LoggingThread
{

   public:
   ~LoggingThread();
   static LoggingThread & getRef();
   void logRhoAscii(const int index, const Array2dF *rho,
           const Array2dF *rhoe, const Array2dF *rhoi);
   void logPhiAscii(const int index, const Array2dF *phi);
   void logRhoBinary(const int index, const Array2dF *rho,
           const Array2dF *rhoe, const Array2dF *rhoi);
   void logPhiBinary(const int index, const Array2dF *phi);
   void logInfo(unsigned int idx, float sTime, 
           unsigned int nmElectrons, unsigned int nmIons,
           bool resumeRun);
   void pushLogItem(LoggingBase *item);
   void flush();
   LoggingBase* popLogItem();

   private:
   LoggingThread();
   void stopLogging();

   static LoggingThread *m_ref;
   boost::mutex *m_stackLock;
   boost::condition_variable *m_cond;
   boost::condition_variable *m_condEmpty;
   std::queue<LoggingBase*> m_jobQueue;
   boost::thread *m_threadHandle;

};


#endif
