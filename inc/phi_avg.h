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
