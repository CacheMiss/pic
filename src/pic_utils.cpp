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

void resizeDim3(dim3 &rhs, std::size_t x, std::size_t y, std::size_t z)
{
   rhs.x = static_cast<unsigned int>(x);
   rhs.y = static_cast<unsigned int>(y);
   rhs.z = static_cast<unsigned int>(z);
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
   sprintf(name,"%s/%s",outputPath.c_str(), fname.c_str(), idx_nm);
   if((fp=fopen(name,"a"))==NULL) {
      printf("Cannot open '%s' file for writing\n",name);
      exit(1);
   }
   fprintf(fp,"%d\t%f\t%d\t%d\n",idx_nm,time,need,niid);
   fclose(fp);
}

void createOutputDir(const char * dir)
{
   if(!boost::filesystem::exists(dir))
   {
      std::cout << "Creating " << dir << std::endl;
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
   else if(fabs(part[2]) > 1000)
   {
      ret = false;
   }
   else if(fabs(part[3]) > 1000)
   {
      ret = false;
   }
   else if(fabs(part[4]) > 1000)
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

void loadPrevSimState(const std::string &loadDir,
                      DevMem<float2> &dev_eleHotLoc, DevMem<float3> &dev_eleHotVel, 
                      DevMem<float2> &dev_eleColdLoc, DevMem<float3> &dev_eleColdVel,
                      DevMem<float2> &dev_ionHotLoc, DevMem<float3> &dev_ionHotVel, 
                      DevMem<float2> &dev_ionColdLoc, DevMem<float3> &dev_ionColdVel,
                      unsigned int &numEleHot, unsigned int &numEleCold,
                      unsigned int &numIonHot, unsigned int &numIonCold)
{
   const unsigned int FPPF = 6; // Floats Per Particle File
   const unsigned int FPPM = 5; // Floats Per Particle Memory

   SimulationState &simState(SimulationState::getRef());
   CommandlineOptions &options(CommandlineOptions::getRef());

   const std::size_t strSize = 40;
   char infoName[strSize];
   char eleName[strSize];
   char ionName[strSize];

   // Generate file names
   snprintf(infoName, strSize, "info");
   boost::filesystem::path infoPath(loadDir + "/" + infoName);
   boost::filesystem::path elePath;
   boost::filesystem::path ionPath;

   if(!boost::filesystem::exists(infoPath))
   {
      std::cerr << "ERROR: " << infoPath.string() << " does not exist!" << std::endl;
      exit(1);
   }

   // Now I need to find the right files to load
   std::ifstream infoF(infoPath.string().c_str());
   infoF.seekg(-1, std::ios_base::end);
   std::stringstream infoLine;
   bool foundFiles = false;
   unsigned int numEle;
   unsigned int numIon;

   infoLine << getPrevLine(infoF);
   while(!foundFiles && infoLine.str() != "")
   {
      unsigned int index;
      infoLine >> simState.iterationNum >> simState.simTime >> numEle >> numIon;
      snprintf(eleName, strSize, "ele_%04d", index);
      snprintf(ionName, strSize, "ion_%04d", index);
      elePath = boost::filesystem::path(loadDir + "/" + eleName);
      ionPath = boost::filesystem::path(loadDir + "/" + ionName);

      if(boost::filesystem::exists(elePath) &&
         boost::filesystem::exists(ionPath))
      {
         foundFiles = true;
      }
      infoLine.clear();
      infoLine << getPrevLine(infoF);
   }

   if(infoLine.str() == "")
   {
      std::cerr << "ERROR: " << infoPath.string() 
         << " is either empty or does not reference any ele or ion files that exist." << std::endl;
      exit(1);
   }

   std::cout << "Restating run from index " << index << std::endl;

   float hot;
   std::ifstream infoFile(infoPath.string().c_str());
   FILE *eleFile;
   FILE *ionFile;

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

void getLastLine(const std::string fileName, std::string &lastLine)
{
   char lastChar = 0;
   std::ifstream f(fileName.c_str());
   // Seek for the last character
   f.seekg(-1, std::ios_base::end);

   lastLine = getPrevLine(f);

}

std::string getPrevLine(std::ifstream &f)
{
   std::string line;
   char lastChar = 0;

   // Loop until I'm done with the file or I've read the last non-emtpy line
   while(f && line == "")
   {
      // Check to see if I've found the start of a line
      lastChar = static_cast<char>(f.peek());
      if('\n' == lastChar || f.tellg() == static_cast<std::ifstream::streampos>(0))
      {
         std::istream::streampos pos = f.tellg();
         // If I'm back at the beginning of the file, lastChar won't be newline
         // In those cases, I don't want to skip the first character
         if('\n' == lastChar)
         {
            f.seekg(1, std::ios_base::cur);
         }

         std::getline(f, line);
         // Clear the EOF bit if its been set
         f.clear();
         // Reset the file cursor to its position before the getline
         f.seekg(pos);
      }
      // I haven't found the start of a non-emtpy line yet.
      // Read one more character back.
      f.seekg(-1, std::ios_base::cur);
   }

   return line;
}
