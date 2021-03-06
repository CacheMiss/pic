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
#include "logging_types.h"

#include <fstream>
#include <iomanip>
#include <sstream>

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
   assert(m_index < 1000000);

   std::stringstream s;
   s << outputPath << "/ele_" << std::setw(D_LOG_IDX_WIDTH) << std::setfill('0') << m_index;
   logParticles(s.str().c_str(), &m_eleHotLoc[0], &m_eleHotVel[0], &m_eleColdLoc[0], &m_eleColdVel[0], 
      m_numEleHot, m_numEleCold);
   s.str("");
   s << outputPath << "/ion_" << std::setw(D_LOG_IDX_WIDTH) << std::setfill('0') << m_index;
   logParticles(s.str().c_str(), &m_ionHotLoc[0], &m_ionHotVel[0], &m_ionColdLoc[0], &m_ionColdVel[0], 
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

// The file format here is as follows
// 4 byte int - Total number of particles
// 4 byte int - Number of hot particles
// 4 byte int - Number of cold particles
// The following sections repeat until all hot particles are written
// 4 byte float - posX
// 4 byte float - posY
// 4 byte float - velX
// 4 byte float - velY
// 4 byte float - velZ
// The following sections repeat until all cold particles are written
// 4 byte float - posX
// 4 byte float - posY
// 4 byte float - velX
// 4 byte float - velY
// 4 byte float - velZ
void LogParticlesBinary::logParticles(const char *fileName, 
                                      const float2 hotLoc[], const float3 hotVel[],
                                      const float2 coldLoc[], const float3 coldVel[],
                                      int maxSizeHot, int maxSizeCold)
{
   std::fstream f;
   // Create the file
   f.open(fileName, std::ios::out | std::ios::binary);
   f.close();
   // Reopen the file for writing
   f.open(fileName, std::ios::in | std::ios::out | std::ios::binary);
   assert(f.good());
   std::ios::streampos numPartLoc;
   std::ios::streampos numHotLoc;
   std::ios::streampos numColdLoc;
   //FILE *f = fopen(fileName, "wb");
   //int numPart = numHot + numCold;
   unsigned int numPart = 0;
   unsigned int numHot = 0;
   unsigned int numCold = 0;

   //fwrite(&numPart, sizeof(int), 1, f);
   //fwrite(&numHot, sizeof(int), 1, f);
   //fwrite(&numCold, sizeof(int), 1, f);
   numPartLoc = f.tellg();
   f.write(reinterpret_cast<const char*>(&numPart), sizeof(numPart));
   numHotLoc = f.tellg();
   f.write(reinterpret_cast<const char*>(&numHot), sizeof(numHot));
   numColdLoc = f.tellg();
   f.write(reinterpret_cast<const char*>(&numCold), sizeof(numCold));
   for(int i = 0; i < maxSizeHot; i++)
   {
      //fwrite(hotLoc + i, sizeof(float2), 1, f);
      //fwrite(hotVel + i, sizeof(float3), 1, f);
      if(hotLoc[i].y != OOB_PARTICLE)
      {
         f.write(reinterpret_cast<const char*>(hotLoc+i), sizeof(hotLoc[0]));
         f.write(reinterpret_cast<const char*>(hotVel+i), sizeof(hotVel[0]));
         numPart++;
         numHot++;
      }
   }
   for(int i = 0; i < maxSizeCold; i++)
   {
      //fwrite(coldLoc + i, sizeof(float2), 1, f);
      //fwrite(coldVel + i, sizeof(float3), 1, f);
      if(coldLoc[i].y != OOB_PARTICLE)
      {
         f.write(reinterpret_cast<const char*>(coldLoc+i), sizeof(coldLoc[0]));
         f.write(reinterpret_cast<const char*>(coldVel+i), sizeof(coldVel[0]));
         numPart++;
         numCold++;
      }
   }
   f.seekg(numPartLoc);
   f.write(reinterpret_cast<const char*>(&numPart), sizeof(numPart));
   f.seekg(numHotLoc);
   f.write(reinterpret_cast<const char*>(&numHot), sizeof(numHot));
   f.seekg(numColdLoc);
   f.write(reinterpret_cast<const char*>(&numCold), sizeof(numCold));
   //fclose(f);
}

void LogParticlesBinary::logData()
{
   assert(m_index < 1000000);

   std::stringstream s;
   s << outputPath << "/ele_" << std::setw(D_LOG_IDX_WIDTH) << std::setfill('0') << m_index;
   logParticles(s.str().c_str(), &m_eleHotLoc[0], &m_eleHotVel[0], 
      &m_eleColdLoc[0], &m_eleColdVel[0],
      m_numEleHot, m_numEleCold);
   s.str("");
   s << outputPath << "/ion_" << std::setw(D_LOG_IDX_WIDTH) << std::setfill('0') << m_index;
   logParticles(s.str().c_str(), &m_ionHotLoc[0], &m_ionHotVel[0], 
      &m_ionColdLoc[0], &m_ionColdVel[0],
      m_numIonHot, m_numIonCold);
}

void LogRhoBinary::logData()
{
    out2drBin("rhoi",m_index,NY,NX1,*m_rhoi, true);
    out2drBin("rhoe",m_index,NY,NX1,*m_rhoe, true);
    out2drBin("rho",m_index,NY,NX1,*m_rho, true);
}

void LogFieldBinary::logData()
{
   out2drBin("ex", m_index, NY, NX1, *m_ex, true);
   out2drBin("ey", m_index, NY, NX1, *m_ey, true);
}

void LogPhiBinary::logData()
{
    out2drBin("phi",m_index,NY,NX1,*m_phi, true);
}

bool LogInfo::first = true;
void LogInfo::logData()
{
   outinfo("info", index, simTime, numElectrons, numIons, first);
   first = false;
}

bool LogForPerformance::first = true;
void LogForPerformance::logData()
{
   std::string fname = outputPath + "/performance.csv";
   std::string keyFname = outputPath + "/performance_key.csv";

   std::ofstream logFile;
   if(first)
   {
      first = false;
      std::ofstream keyFile(keyFname.c_str());
      keyFile << "Iteration Number,Sim Time,Num Electrons Hot,"
                 "NumElectrons Cold,Num Ions Hot,Num IonsCold,"
                 "Iteration Time (ms),Inject Time (ms),Dens Time (ms),"
                 "Potent2 Time (ms),Field Time(ms),Movep Time (ms)";

      if(resume)
      {
         logFile.open(fname.c_str(), std::ios::out | std::ios::app);
      }
      else
      {
         logFile.open(fname.c_str(), std::ios::out);
      }
   }
   else
   {
      logFile.open(fname.c_str(), std::ios::out | std::ios::app);
   }
   logFile << iteration << "," << simTime << "," << numEleHot << "," << numEleCold << ","
      << numIonHot << "," << numIonCold << "," << iterTimeInMs << "," << injectTimeInMs << ","
      << densTimeInMs << "," << potent2TimeInMs << "," << fieldTimeInMs << "," << movepTimeInMs
      << std::endl;
}

void LogAvgPhi::logData()
{
   std::stringstream s;
   s << outputPath + "/phiAvg_" << std::setw(D_LOG_IDX_WIDTH) << std::setfill('0') << idx;
   std::ofstream f;
   f.open(s.str().c_str(), std::ios::binary);
   // Write number of rows
   f.write(reinterpret_cast<char*>(&y), sizeof(y));
   // Write number of columns
   f.write(reinterpret_cast<char*>(&x), sizeof(x));
   unsigned int columnOrder = 0; // We're writing row order
   f.write(reinterpret_cast<char*>(&columnOrder), sizeof(columnOrder));
   for(std::size_t c = 0; c < x; c++)
   {
      for(std::size_t r = 0; r < y; r++)
      {
         f.write(reinterpret_cast<char*>(&data[r*x+c]), sizeof(data[0]));
      }
   }
}
