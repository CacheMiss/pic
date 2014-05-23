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
#include "phi_avg.h"

#include <tbb/tbb.h>

PhiAvg::PhiAvg(std::size_t phiSize, unsigned int xSize, unsigned int ySize)
  : m_phiStage(phiSize)
  , m_phiTotal(phiSize)
  , m_phiSize(phiSize)
  , m_avgSize(0)
  , m_xSize(xSize)
  , m_ySize(ySize)
{}

void PhiAvg::clear()
{
   memset(&m_phiTotal[0], 0, sizeof(m_phiTotal[0]) * m_phiTotal.size());
   m_avgSize = 0;
}

void PhiAvg::addPhi(const DevMem<float> &phi)
{
   m_phiStage = phi;

   AddFunc addFunc;
   addFunc.src = &m_phiStage[0];
   addFunc.dst = &m_phiTotal[0];
   tbb::parallel_for(tbb::blocked_range<std::size_t>(0, m_phiSize), addFunc);
   m_avgSize++;
}

void PhiAvg::saveToVector(std::vector<float> &phiAvg) const
{
   if(m_avgSize == 0)
   {
      phiAvg.resize(0);
      return;
   }

   phiAvg.resize(m_phiSize);
   AvgPhi avgPhiFunction;
   avgPhiFunction.total = &m_phiTotal[0];
   avgPhiFunction.avgSize = m_avgSize;
   avgPhiFunction.dst = &phiAvg[0];
   tbb::parallel_for(tbb::blocked_range<std::size_t>(0, m_phiSize), avgPhiFunction);
}

unsigned int PhiAvg::getXSize() const
{
   return m_xSize;
}

unsigned int PhiAvg::getYSize() const
{
   return m_ySize;
}
