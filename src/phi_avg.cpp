#include "phi_avg.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

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
   AddFunc addFunc;
   addFunc.src = &m_phiStage[0];
   addFunc.dst = &m_phiTotal[0];
   tbb::parallel_for(tbb::blocked_range<std::size_t>(0, m_phiSize), addFunc);
   m_avgSize++;
}

void PhiAvg::saveToVector(std::vector<float> &phiAvg) const
{
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
