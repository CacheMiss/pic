#ifndef _PRECISIONTIMER_H_
#define _PRECISIONTIMER_H_

#ifndef WIN32
#include <boost/date_time/posix_time/posix_time_types.hpp>
#else
namespace boost
{
   namespace posix_time
   {
      class ptime;
   };
};
#endif

class PrecisionTimer
{
  boost::posix_time::ptime  *start_time;
  boost::posix_time::ptime  *end_time;

public:
  PrecisionTimer();
  ~PrecisionTimer();
#ifdef WIN32
  void start();
  void stop();
#else
  inline void start();
  inline void stop();
#endif
  long intervalInS();
  long intervalInMilliS();
  long intervalInMicroS();
  long intervalInNanoS();
};

#ifndef WIN32
void PrecisionTimer::start()
{
   *start_time = boost::posix_time::microsec_clock::local_time();
}

void PrecisionTimer::stop()
{
   *end_time = boost::posix_time::microsec_clock::local_time();
}
#endif

#endif // _PRECISIONTIMER_H_
