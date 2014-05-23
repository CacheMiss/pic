#include "precisiontimer.h"

#include <assert.h>
#include <boost/date_time/posix_time/posix_time_types.hpp>

PrecisionTimer::PrecisionTimer()
{
}

PrecisionTimer::~PrecisionTimer()
{
}

boost::timer::nanosecond_type PrecisionTimer::intervalInS()
{
   boost::timer::cpu_times interval = m_timer.elapsed();
   return interval.wall / 1000000000;
}

boost::timer::nanosecond_type PrecisionTimer::intervalInMilliS()
{
   boost::timer::cpu_times interval = m_timer.elapsed();
   return interval.wall / 1000000;
}

boost::timer::nanosecond_type PrecisionTimer::intervalInMicroS()
{
   boost::timer::cpu_times interval = m_timer.elapsed();
   return interval.wall / 1000;
}

boost::timer::nanosecond_type PrecisionTimer::intervalInNanoS()
{
   boost::timer::cpu_times interval = m_timer.elapsed();
   return interval.wall;
}
