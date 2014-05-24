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
