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
