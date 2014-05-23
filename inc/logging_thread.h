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
