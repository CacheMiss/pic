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
   void logForPerformance(unsigned int iter, float sTime,
           unsigned int nEleHot, unsigned int nEleCold,
           unsigned int nIonHot, unsigned int nIonCold,
           unsigned int iterTInMs, unsigned int injectTInMs,
           unsigned int densTInMs, unsigned int potent2TInMs,
           unsigned int fieldTInMs, unsigned int movepTInMs,
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
