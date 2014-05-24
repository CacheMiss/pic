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
#ifndef PHI_AVG_H
#define PHI_AVG_H

#include <vector>
#include <tbb/tbb.h>

#include "host_mem.h"

class PhiAvg
{
public:
   PhiAvg(std::size_t phiSize, unsigned int xSize, unsigned int ySize);
   void clear();
   void addPhi(const DevMem<float> &phi);
   void saveToVector(std::vector<float> &phiAvg) const;

   unsigned int getXSize() const;
   unsigned int getYSize() const;

private:
   HostMem<float> m_phiStage;
   std::vector<double> m_phiTotal;
   std::size_t m_phiSize;
   double m_avgSize;
   unsigned int  m_xSize;
   unsigned int  m_ySize;

	struct AddFunc
	{
		float* src;
		double *dst;

		void operator()(const tbb::blocked_range<std::size_t>& range) const
		{
			for(std::size_t i = range.begin(); i != range.end(); i++)
			{
				dst[i] += src[i];
			}
		}
	};

	struct AvgPhi
	{
		const double* total;
		double avgSize;
		float* dst;

		void operator()(const tbb::blocked_range<std::size_t>& range) const
		{
			for(std::size_t i = range.begin(); i != range.end(); i++)
			{
				dst[i] = static_cast<float>(total[i] / avgSize);
			}
		}
	};

};

#endif
