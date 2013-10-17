#ifndef LOGGING_TYPES_H
#define LOGGING_TYPES_H

#include "array2d.h"
#include "dev_mem.h"
#include "typedefs.h"

class LoggingBase
{
   public:
      LoggingBase(){}
      virtual ~LoggingBase(){}
      virtual void logData()=0;
};

class FlushLog : public LoggingBase
{
   virtual void logData(){};
};

class LogParticlesAscii : public LoggingBase
{
   int m_index;
   std::vector<float2> m_eleHotLoc;
   std::vector<float3> m_eleHotVel;
   std::vector<float2> m_eleColdLoc;
   std::vector<float3> m_eleColdVel;
   std::vector<float2> m_ionHotLoc;
   std::vector<float3> m_ionHotVel;
   std::vector<float2> m_ionColdLoc;
   std::vector<float3> m_ionColdVel;

   int m_numEleHot;
   int m_numEleCold;
   int m_numIonHot;
   int m_numIonCold;
   void logParticles(const char *fileName, 
                     const float2 hotLoc[], const float3 hotVel[],
                     const float2 coldLoc[], const float3 coldVel[],
                     const int numHot, const int numCold);

   public:
   LogParticlesAscii(const int i,
                const DevMemF2 &eHLoc, const DevMemF3 &eHVel, 
                const DevMemF2 &eCLoc, const DevMemF3 &eCVel, 
                const DevMemF2 &iHLoc, const DevMemF3 &iHVel, 
                const DevMemF2 &iCLoc, const DevMemF3 &iCVel, 
                const int nEleH, const int nEleC,
                const int nIonH, const int nIonC);
   ~LogParticlesAscii()
   {}

   virtual void logData();
};

class LogParticlesBinary : public LoggingBase
{
   int m_index;
   std::vector<float2> m_eleHotLoc;
   std::vector<float3> m_eleHotVel;
   std::vector<float2> m_eleColdLoc;
   std::vector<float3> m_eleColdVel;
   std::vector<float2> m_ionHotLoc;
   std::vector<float3> m_ionHotVel;
   std::vector<float2> m_ionColdLoc;
   std::vector<float3> m_ionColdVel;

   int m_numEleHot;
   int m_numEleCold;
   int m_numIonHot;
   int m_numIonCold;
   void logParticles(const char *fileName, 
                     const float2 hotLoc[], const float3 hotVel[],
                     const float2 coldLoc[], const float3 coldVel[],
                     int maxSizeHot, int maxSizeCold);

   public:
   LogParticlesBinary(const int i,
                const DevMemF2 &eHLoc, const DevMemF3 &eHVel, 
                const DevMemF2 &eCLoc, const DevMemF3 &eCVel, 
                const DevMemF2 &iHLoc, const DevMemF3 &iHVel, 
                const DevMemF2 &iCLoc, const DevMemF3 &iCVel, 
                const int nEleH, const int nEleC,
                const int nIonH, const int nIonC);
   ~LogParticlesBinary()
   {}


   virtual void logData();
};

class LogRhoAscii : public LoggingBase
{
   private:
   int m_index;
   const Array2dF *m_rho;
   const Array2dF *m_rhoe;
   const Array2dF *m_rhoi;

   public:
   LogRhoAscii(const int i, const Array2dF *r,
               const Array2dF *rE, const Array2dF *rI)
     :LoggingBase(), m_index(i),
      m_rho(r), m_rhoe(rE), m_rhoi(rI)
   {}
   ~LogRhoAscii()
   {
      delete m_rho;
      delete m_rhoe;
      delete m_rhoi;
   }
   virtual void logData();
};

class LogRhoBinary : public LoggingBase
{
   private:
   int m_index;
   const Array2dF *m_rho;
   const Array2dF *m_rhoe;
   const Array2dF *m_rhoi;

   public:
   LogRhoBinary(const int i,
                const Array2dF *r,
                const Array2dF *rE, const Array2dF *rI)
     :LoggingBase(), m_index(i),
      m_rho(r), m_rhoe(rE), m_rhoi(rI)
   {}
   ~LogRhoBinary()
   {
      delete m_rho;
      delete m_rhoe;
      delete m_rhoi;
   }
   virtual void logData();
};

class LogPhiAscii : public LoggingBase
{
   private:
   int m_index;
   const Array2dF *m_phi;

   public:
   LogPhiAscii(const int i, const Array2dF *p)
     :LoggingBase(), m_index(i), m_phi(p)
   {}
   ~LogPhiAscii()
   {
      delete m_phi;
   }

   virtual void logData();
};

class LogPhiBinary : public LoggingBase
{
   private:
   int m_index;
   const Array2dF *m_phi;

   public:
   LogPhiBinary(const int i, const Array2dF *p)
     :LoggingBase(), m_index(i), m_phi(p)
   {}
   ~LogPhiBinary()
   {
      delete m_phi;
   }

   virtual void logData();
};

class LogInfo : public LoggingBase
{
   private:
   unsigned int index;
   float simTime;
   unsigned int numElectrons;
   unsigned int numIons;
   bool resume;
   static bool first;

   public:
   LogInfo(unsigned int idx, float sTime, 
           unsigned int nmElectrons, unsigned int nmIons,
           bool resumeRun=false)
     :LoggingBase()
     ,index(idx)
     ,simTime(sTime)
     ,numElectrons(nmElectrons)
     ,numIons(nmIons)
     ,resume(resumeRun)
   {
      if(resumeRun)
      {
         first = false;
      }
   }

   virtual void logData();
};

class LogForPerformance : public LoggingBase
{
   private:
   unsigned int iteration;
   float        simTime;
   unsigned int numEleHot;
   unsigned int numEleCold;
   unsigned int numIonHot;
   unsigned int numIonCold;
   unsigned int iterTimeInMs;
   unsigned int injectTimeInMs;
   unsigned int densTimeInMs;
   unsigned int potent2TimeInMs;
   unsigned int fieldTimeInMs;
   unsigned int movepTimeInMs;
   bool         resume;
   static bool first;

   public:
   LogForPerformance(unsigned int iter, float sTime,
      unsigned int nEleHot, unsigned int nEleCold,
      unsigned int nIonHot, unsigned int nIonCold,
      unsigned int iterTInMs, unsigned int injectTInMs,
      unsigned int densTInMs, unsigned int potent2TInMs,
      unsigned int fieldTInMs, unsigned int movepTInMs,
      bool resumeRun=false)
     :LoggingBase()
     ,iteration(iter)
     ,simTime(sTime)
     ,numEleHot(nEleHot)
     ,numEleCold(nEleCold)
     ,numIonHot(nIonHot)
     ,numIonCold(nIonCold)
     ,iterTimeInMs(iterTInMs)
     ,injectTimeInMs(injectTInMs)
     ,densTimeInMs(densTInMs)
     ,potent2TimeInMs(potent2TInMs)
     ,fieldTimeInMs(fieldTInMs)
     ,movepTimeInMs(movepTInMs)
     ,resume(resumeRun)
   {
   }

   virtual void logData();
};

#endif
