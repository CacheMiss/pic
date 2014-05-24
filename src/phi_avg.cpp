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
