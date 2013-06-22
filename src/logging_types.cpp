#include "logging_types.h"

#include "array2d.h"
#include "global_variables.h"
#include "pic_utils.h"

LogParticlesAscii::LogParticlesAscii(const int i,
             const DevMemF2 &eHLoc, const DevMemF3 &eHVel, 
             const DevMemF2 &eCLoc, const DevMemF3 &eCVel, 
             const DevMemF2 &iHLoc, const DevMemF3 &iHVel, 
             const DevMemF2 &iCLoc, const DevMemF3 &iCVel, 
             const int nEleH, const int nEleC,
             const int nIonH, const int nIonC)
  :LoggingBase(), m_index(i), 
   m_eleHotLoc(nEleH > 0 ? nEleH : 1), m_eleHotVel(nEleH > 0 ? nEleH : 1),
   m_eleColdLoc(nEleC > 0 ? nEleC : 1), m_eleColdVel(nEleC > 0 ? nEleC : 1),
   m_ionHotLoc(nIonH > 0 ? nIonH : 1), m_ionHotVel(nIonH > 0 ? nIonH : 1),
   m_ionColdLoc(nIonC > 0 ? nIonC : 1), m_ionColdVel(nIonC > 0 ? nIonC : 1),
   m_numEleHot(nEleH), m_numEleCold(nEleC), 
   m_numIonHot(nIonH), m_numIonCold(nIonC)
{
   if(m_numEleHot > 0)
   {
      eHLoc.copyArrayToHost(&m_eleHotLoc[0], m_numEleHot);
      eHVel.copyArrayToHost(&m_eleHotVel[0], m_numEleHot);
   }
   if(m_numEleCold > 0)
   {
      eCLoc.copyArrayToHost(&m_eleColdLoc[0], m_numEleCold);
      eCVel.copyArrayToHost(&m_eleColdVel[0], m_numEleCold);
   }
   if(m_numIonHot > 0)
   {
      iHLoc.copyArrayToHost(&m_ionHotLoc[0], m_numIonHot);
      iHVel.copyArrayToHost(&m_ionHotVel[0], m_numIonHot);
   }
   if(m_numIonCold > 0)
   {
      iCLoc.copyArrayToHost(&m_ionColdLoc[0], m_numIonCold);
      iCVel.copyArrayToHost(&m_ionColdVel[0], m_numIonCold);
   }
}

void LogParticlesAscii::logParticles(const char *fileName, 
                                     const float2 hotLoc[], const float3 hotVel[],
                                     const float2 coldLoc[], const float3 coldVel[],
                                     const int numHot, const int numCold)
{
   FILE *f = fopen(fileName, "w");
   for(int i = 0; i < numHot; i++)
   {
      fprintf(f, "%12.3f %12.3f %12.3f %12.3f %12.3f %12.3f\n", 
         hotLoc[i].x, hotLoc[i].y, hotVel[i].x, hotVel[i].y, hotVel[i].z, 1.0);
   }
   for(int i = 0; i < numCold; i++)
   {
      fprintf(f, "%12.3f %12.3f %12.3f %12.3f %12.3f %12.3f\n", 
         coldLoc[i].x, coldLoc[i].y, coldVel[i].x, coldVel[i].y, coldVel[i].z, 0.0);
   }
   fclose(f);
}

void LogParticlesAscii::logData()
{
   char name[100];
   assert(m_index < 1000000);

   sprintf(name,"%s/%s_%04d",outputDir.c_str(), "ele", m_index);
   logParticles(name, &m_eleHotLoc[0], &m_eleHotVel[0], &m_eleColdLoc[0], &m_eleColdVel[0], 
      m_numEleHot, m_numEleCold);
   sprintf(name,"%s/%s_%04d",outputDir.c_str(), "ion", m_index);
   logParticles(name, &m_ionHotLoc[0], &m_ionHotVel[0], &m_ionColdLoc[0], &m_ionColdVel[0], 
      m_numIonHot, m_numIonCold);
}

void LogRhoAscii::logData()
{
    out2dr("rhoi",m_index,NY,NX1,*m_rhoi, true);
    out2dr("rhoe",m_index,NY,NX1,*m_rhoe, true);
    out2dr("rho_",m_index,NY,NX1,*m_rho, true);
}

void LogPhiAscii::logData()
{
    out2dr("phi_",m_index,NY,NX1,*m_phi, true);
}

LogParticlesBinary::LogParticlesBinary(const int i,
             const DevMemF2 &eHLoc, const DevMemF3 &eHVel, 
             const DevMemF2 &eCLoc, const DevMemF3 &eCVel, 
             const DevMemF2 &iHLoc, const DevMemF3 &iHVel, 
             const DevMemF2 &iCLoc, const DevMemF3 &iCVel, 
             const int nEleH, const int nEleC,
             const int nIonH, const int nIonC)
  :LoggingBase(), m_index(i), 
   m_eleHotLoc(nEleH > 0 ? nEleH : 1), m_eleHotVel(nEleH > 0 ? nEleH : 1),
   m_eleColdLoc(nEleC > 0 ? nEleC : 1), m_eleColdVel(nEleC > 0 ? nEleC : 1),
   m_ionHotLoc(nIonH > 0 ? nIonH : 1), m_ionHotVel(nIonH > 0 ? nIonH : 1),
   m_ionColdLoc(nIonC > 0 ? nIonC : 1), m_ionColdVel(nIonC > 0 ? nIonC : 1),
   m_numEleHot(nEleH), m_numEleCold(nEleC), 
   m_numIonHot(nIonH), m_numIonCold(nIonC)
{
   if(m_numEleHot > 0)
   {
      eHLoc.copyArrayToHost(&m_eleHotLoc[0], m_numEleHot);
      eHVel.copyArrayToHost(&m_eleHotVel[0], m_numEleHot);
   }
   if(m_numEleCold > 0)
   {
      eCLoc.copyArrayToHost(&m_eleColdLoc[0], m_numEleCold);
      eCVel.copyArrayToHost(&m_eleColdVel[0], m_numEleCold);
   }
   if(m_numIonHot > 0)
   {
      iHLoc.copyArrayToHost(&m_ionHotLoc[0], m_numIonHot);
      iHVel.copyArrayToHost(&m_ionHotVel[0], m_numIonHot);
   }
   if(m_numIonCold > 0)
   {
      iCLoc.copyArrayToHost(&m_ionColdLoc[0], m_numIonCold);
      iCVel.copyArrayToHost(&m_ionColdVel[0], m_numIonCold);
   }
}

void LogParticlesBinary::logParticles(const char *fileName, 
                                      const float2 hotLoc[], const float3 hotVel[],
                                      const float2 coldLoc[], const float3 coldVel[],
                                      const int numHot, const int numCold)
{
   FILE *f = fopen(fileName, "wb");
   int numPart = numHot + numCold;

   fwrite(&numPart, sizeof(int), 1, f);
   fwrite(&numHot, sizeof(int), 1, f);
   fwrite(&numCold, sizeof(int), 1, f);
   for(int i = 0; i < numHot; i++)
   {
      fwrite(hotLoc + i, sizeof(float2), 1, f);
      fwrite(hotVel + i, sizeof(float3), 1, f);
   }
   for(int i = 0; i < numCold; i++)
   {
      fwrite(coldLoc + i, sizeof(float2), 1, f);
      fwrite(coldVel + i, sizeof(float3), 1, f);
   }
   fclose(f);
}

void LogParticlesBinary::logData()
{
   char name[100];
   assert(m_index < 1000000);

   sprintf(name,"%s/%s_%04d",outputDir.c_str(), "ele_", m_index);
   logParticles(name, &m_eleHotLoc[0], &m_eleHotVel[0], &m_eleColdLoc[0], &m_eleColdVel[0], 
      m_numEleHot, m_numEleCold);
   sprintf(name,"%s/%s_%04d",outputDir.c_str(), "ion_", m_index);
   logParticles(name, &m_ionHotLoc[0], &m_ionHotVel[0], &m_ionColdLoc[0], &m_ionColdVel[0], 
      m_numIonHot, m_numIonCold);
}

void LogRhoBinary::logData()
{
    out2drBin("rhoi",m_index,NY,NX1,*m_rhoi, true);
    out2drBin("rhoe",m_index,NY,NX1,*m_rhoe, true);
    out2drBin("rho",m_index,NY,NX1,*m_rho, true);
}

void LogPhiBinary::logData()
{
    out2drBin("phi",m_index,NY,NX1,*m_phi, true);
}

void LogInfo::logData()
{
   outinfo("info", index, simTime, numElectrons, numIons);
}

void LogForPerformance::logData()
{
   std::string fname = outputDir + "/performance.csv";
   std::string keyFname = outputDir + "/performance_key.csv";
   FILE *fp;

   if(first)
   {
      first = false;
      fp = fopen(keyFname.c_str(), "w");
      fprintf(fp, "Iteration Number,Sim Time,Num Electrons Hot,"
              "NumElectrons Cold,Num Ions Hot,Num IonsCold,"
              "Iteration Time (ms)");
      fclose(fp);
      fp = fopen(fname.c_str(), resume ? "a" : "w");
   }
   else
   {
      fp = fopen(fname.c_str(), "a");
   }
   fprintf(fp, "%u,%f,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\n", iteration, simTime, numEleHot,
      numEleCold, numIonHot, numIonCold, iterTimeInMs, injectTimeInMs, densTimeInMs,
      potent2TimeInMs, fieldTimeInMs, movepTimeInMs);
   fclose(fp);
}

