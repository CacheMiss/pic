#ifndef _PRECISIONTIMER_H_
#define _PRECISIONTIMER_H_

#include <boost/timer/timer.hpp>

class PrecisionTimer
{
   boost::timer::cpu_timer m_timer;

public:
  PrecisionTimer();
  ~PrecisionTimer();
  inline void start();
  inline void stop();
  boost::timer::nanosecond_type intervalInS();
  boost::timer::nanosecond_type intervalInMilliS();
  boost::timer::nanosecond_type intervalInMicroS();
  boost::timer::nanosecond_type intervalInNanoS();
};

void PrecisionTimer::start()
{
   m_timer.start();
}

void PrecisionTimer::stop()
{
   m_timer.stop();
}

#endif // _PRECISIONTIMER_H_
