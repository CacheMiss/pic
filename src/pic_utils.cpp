#include "pic_utils.h"

#include <algorithm>
#include <assert.h>
#include <boost/filesystem.hpp>
#include <cuda_runtime_api.h>
#include <fstream>
#include <stdio.h>

#include "commandline_options.h"
#include "host_mem.h"
#include "simulation_state.h"

void checkForCudaError(const char *errorMsg)
{
   cudaError_t error = cudaGetLastError();
   if(error != cudaSuccess)
   {
      FILE *file;
      SimulationState &simState(SimulationState::getRef());
      fprintf(stderr,"ERROR on iteration %u: %s: %s\n", 
         simState.iterationNum, errorMsg, cudaGetErrorString(error) );
      file = fopen(errorLogName.c_str(), "w");
      fprintf(file,"ERROR on iteration %u: %s\n", simState.iterationNum,
         cudaGetErrorString(error) );
      fclose(file);
      assert(error == cudaSuccess);
      exit(1);
   }
}

void checkForCudaErrorSync(const char *errorMsg)
{
   cudaThreadSynchronize();
   checkForCudaError(errorMsg);
}

void checkForCudaError(const char *errorMsg, cudaError_t error)
{
   if(error != cudaSuccess)
   {
      SimulationState &simState(SimulationState::getRef());
      fprintf(stderr,"ERROR on iteration %u: %s: %s\n", simState.iterationNum, 
         errorMsg, cudaGetErrorString(error) );
      assert(error == cudaSuccess);
      exit(1);
   }
}

void errExit(const char *errorString)
{
   FILE *file;
   file = fopen(errorLogName.c_str(), "w");
   fprintf(stderr,"ERROR: %s\n", errorString);
   fprintf(file,"ERROR: %s\n", errorString);
   fclose(file);
   assert(false);
   exit(1);
}

void resizeDim3(dim3 &rhs, int x, int y, int z)
{
   rhs.x = x;
   rhs.y = y;
   rhs.z = z;
}

//*****************************************************************************
// Name: outinfo
// Purpose: Output info files that are used to periodically report
//          on the iteration number, simulation time, number of
//          electrons and the number of ions
// Parameters:
// ---------------------
// fname  - Filename to output
// idx_nm - Index to append to fname
// time   - TODO *** Find a better description ***
// need   - The number of electrons
// niid   - The number of ions
//*****************************************************************************
void outinfo(const std::string &fname,int idx_nm,float time,int need,int niid)
{
   char name[100];
   FILE *fp;
   sprintf(name,"%s/%s_%04d",outputDir.c_str(), fname.c_str(), idx_nm);
   if((fp=fopen(name,"wt"))==NULL) {
      printf("Cannot open '%s' file for writing\n",name);
      exit(1);
   }
   fprintf(fp,"%d %f %d %d\n",idx_nm,time,need,niid);
   fclose(fp);
}

void createOutputDir(const char * dir)
{
   if(!boost::filesystem::exists(dir))
   {
      boost::filesystem::create_directory(dir);
   }
}

bool fileExists(const std::string &fileName)
{
  if(boost::filesystem::exists(fileName))
  {
     return true;
  }
  else
  {
     return false;
  }
}

bool verifyParticle(const float part[], unsigned int nx1, unsigned int ny1)
{
   bool ret = true;
   if(part[0] < 0 || part[0] > nx1)
   {
      ret = false;
   }
   else if(part[1] < 0 || part[1] > ny1)
   {
      ret = false;
   }
   else if(abs(part[2]) > 1000)
   {
      ret = false;
   }
   else if(abs(part[3]) > 1000)
   {
      ret = false;
   }
   else if(abs(part[4]) > 1000)
   {
      ret = false;
   }
   else if(part[0] != part[0])
   {
      ret = false;
   }
   else if(part[1] != part[1])
   {
      ret = false;
   }
   else if(part[2] != part[2])
   {
      ret = false;
   }
   else if(part[3] != part[3])
   {
      ret = false;
   }
   else if(part[4] != part[4])
   {
      ret = false;
   }
   else if(part[0] == 0 && part[1] == 0 && 
           part[2] == 0 && part[3] == 0)
   {
      ret = false;
   }

   return ret;
}

void loadPrevSimState(unsigned int loadIndex, const std::string &loadDir,
                      DevMem<float2> &dev_eleHotLoc, DevMem<float3> &dev_eleHotVel, 
                      DevMem<float2> &dev_eleColdLoc, DevMem<float3> &dev_eleColdVel,
                      DevMem<float2> &dev_ionHotLoc, DevMem<float3> &dev_ionHotVel, 
                      DevMem<float2> &dev_ionColdLoc, DevMem<float3> &dev_ionColdVel,
                      unsigned int &numEleHot, unsigned int &numEleCold,
                      unsigned int &numIonHot, unsigned int &numIonCold)
{
   assert(loadIndex < 10000);
   const unsigned int FPPF = 6; // Floats Per Particle File
   const unsigned int FPPM = 5; // Floats Per Particle Memory

   SimulationState &simState(SimulationState::getRef());
   CommandlineOptions &options(CommandlineOptions::getRef());

   char infoName[40];
   char eleName[40];
   char ionName[40];

   // Generate file names
   sprintf(infoName, "info_%04d", loadIndex);
   sprintf(eleName, "ele__%04d", loadIndex);
   sprintf(ionName, "ion__%04d", loadIndex);
   boost::filesystem::path infoPath(loadDir + "/" + infoName);
   boost::filesystem::path elePath(loadDir + "/" + eleName);
   boost::filesystem::path ionPath(loadDir + "/" + ionName);

   if(!boost::filesystem::exists(infoPath))
   {
      std::cerr << "ERROR: " << infoPath.string() << " does not exist!" << std::endl;
      exit(1);
   }
   if(!boost::filesystem::exists(elePath))
   {
      std::cerr << "ERROR: " << elePath.string() << " does not exist!" << std::endl;
      exit(1);
   }
   if(!boost::filesystem::exists(ionPath))
   {
      std::cerr << "ERROR: " << ionPath.string() << " does not exist!" << std::endl;
      exit(1);
   }

   unsigned int numEle;
   unsigned int numIon;
   float hot;
   std::ifstream infoFile(infoPath.string().c_str());
   FILE *eleFile;
   FILE *ionFile;

   infoFile >> simState.iterationNum >> simState.simTime >> numEle >> numIon;
   infoFile.close();
   simState.iterationNum *= D_LF;

   HostMem<float2> h_eleHotLoc(dev_eleHotLoc.size());
   HostMem<float3> h_eleHotVel(dev_eleHotVel.size());
   HostMem<float2> h_eleColdLoc(dev_eleColdLoc.size());
   HostMem<float3> h_eleColdVel(dev_eleColdVel.size());
   HostMem<float2> h_ionHotLoc(dev_ionHotLoc.size());
   HostMem<float3> h_ionHotVel(dev_ionHotVel.size());
   HostMem<float2> h_ionColdLoc(dev_ionColdLoc.size());
   HostMem<float3> h_ionColdVel(dev_ionColdVel.size());

   numEleHot = 0;
   numEleCold = 0;
   numIonHot = 0;
   numIonCold = 0;

   float2 tmpLoc;
   float3 tmpVel;

   eleFile = fopen(elePath.string().c_str(), "rb");
   fread(&numEle, sizeof(unsigned int), 1, eleFile);
   for(unsigned int i = 0; i < numEle; i++)
   {
      fread(&tmpLoc, sizeof(float2), 1, eleFile);
      fread(&tmpVel, sizeof(float3), 1, eleFile);
      fread(&hot, sizeof(float), 1, eleFile);
      if(hot > 0)
      {
         assert(numEleHot < h_eleHotLoc.size());
         h_eleHotLoc[numEleHot] = tmpLoc;
         h_eleHotVel[numEleHot] = tmpVel;
         numEleHot++;
      }
      else
      {
         assert(numEleCold < h_eleColdLoc.size());
         h_eleColdLoc[numEleCold] = tmpLoc;
         h_eleColdVel[numEleCold] = tmpVel;
         numEleCold++;
      }
   }
   fclose(eleFile);

   ionFile = fopen(ionPath.string().c_str(), "rb");
   fread(&numIon, sizeof(unsigned int), 1, ionFile);
   for(unsigned int i = 0; i < numIon; i++)
   {
      fread(&tmpLoc, sizeof(float2), 1, ionFile);
      fread(&tmpVel, sizeof(float3), 1, ionFile);
      fread(&hot, sizeof(float), 1, ionFile);
      if(hot > 0)
      {
         assert(numIonHot < h_ionHotLoc.size());
         h_ionHotLoc[numIonHot] = tmpLoc;
         h_ionHotVel[numIonHot] = tmpVel;
         numIonHot++;
      }
      else
      {
         assert(numIonCold < h_ionColdLoc.size());
         h_ionColdLoc[numIonCold] = tmpLoc;
         h_ionColdVel[numIonCold] = tmpVel;
         numIonCold++;
      }
   }
   fclose(ionFile);

   dev_eleHotLoc.copyArrayToDev(h_eleHotLoc);
   dev_eleHotVel.copyArrayToDev(h_eleHotVel);
   dev_eleColdLoc.copyArrayToDev(h_eleColdLoc);
   dev_eleColdVel.copyArrayToDev(h_eleColdVel);
   dev_ionHotLoc.copyArrayToDev(h_ionHotLoc);
   dev_ionHotVel.copyArrayToDev(h_ionHotVel);
   dev_ionColdLoc.copyArrayToDev(h_ionColdLoc);
   dev_ionColdVel.copyArrayToDev(h_ionColdVel);
}
