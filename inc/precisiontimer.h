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
