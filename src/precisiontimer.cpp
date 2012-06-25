#include "precisiontimer.h"

#include <assert.h>
#include <boost/date_time/posix_time/posix_time_types.hpp>

PrecisionTimer::PrecisionTimer()
  :start_time(new boost::posix_time::ptime), 
   end_time(new boost::posix_time::ptime)
{
}

PrecisionTimer::~PrecisionTimer()
{
   delete start_time;
   delete end_time;
}

#ifdef WIN32
void PrecisionTimer::start()
{
   *start_time = boost::posix_time::microsec_clock::local_time();
}

void PrecisionTimer::stop()
{
   *end_time = boost::posix_time::microsec_clock::local_time();
}
#endif

long PrecisionTimer::intervalInS()
{
   boost::posix_time::time_duration interval(*end_time - *start_time);
   return interval.total_seconds();
}

long PrecisionTimer::intervalInMilliS()
{
   boost::posix_time::time_duration interval(*end_time - *start_time);
   return (long)interval.total_milliseconds();
}

long PrecisionTimer::intervalInMicroS()
{
   boost::posix_time::time_duration interval(*end_time - *start_time);
   return (long)interval.total_microseconds();
}

long PrecisionTimer::intervalInNanoS()
{
   boost::posix_time::time_duration interval(*end_time - *start_time);
   return (long)interval.total_nanoseconds();
}
