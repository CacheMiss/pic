#include "pic_utils.h"

#include <algorithm>
#include <assert.h>
#include <boost/filesystem.hpp>
#include <cuda_runtime_api.h>
#include <fstream>
#include <sstream>
#include <stdio.h>

#include "commandline_options.h"
#include "host_mem.h"
#include "simulation_state.h"

void checkForCudaError(const char *errorMsg)
{
   cudaError_t error = cudaGetLastError();
   if(error != cudaSuccess)
   {
      std::stringstream s;
      std::ofstream file;
      SimulationState &simState(SimulationState::getRef());
      s << "ERROR on iteration " << simState.iterationNum
        << " " << errorMsg << " " << cudaGetErrorString(error);
      std::cerr << s.str() << std::endl;;
      file.open(errorLogName.c_str());
      file << s.str();
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
// first  - Flag which governs overwriting the info file
//*****************************************************************************
void outinfo(const std::string &fname,
             int idx_nm,
             float time,
             int need,
             int niid,
             bool first)
{
   char name[100];
   FILE *fp;
   sprintf(name,"%s/%s",outputPath.c_str(), fname.c_str(), idx_nm);
   if(!first)
   {
      fp = fopen(name, "a");
   }
   else
   {
      fp = fopen(name, "w");
   }
   if(fp==NULL)
   {
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

bool verifyParticles(const float2 partLoc[], std::size_t size, 
                     unsigned int nx1, unsigned int ny1,
                     const char* whatAmI)
{
   bool ret = true;
   std::size_t numErrors = 0;
   std::size_t printLimit = 20;
   for(std::size_t i = 0; i < size; i++)
   {
      if(partLoc[i].y < DY || partLoc[i].y > DY * NY1 - 1)
      {
         if(numErrors < printLimit)
         {
            std::cout << "ERROR: " << whatAmI << " found at x=" << partLoc[i].x
               << " y=" << partLoc[i].y << std::endl;
         }
         numErrors++;
      }
   }

   return ret;
}

void saveConfiguration(const CommandlineOptions &options, const std::string& fileName)
{
   std::ofstream outFile(fileName.c_str()); 

   outFile << "--log-interval" << std::endl;
   outFile << options.getLogInterval() << std::endl;
   outFile << "--max-time" << std::endl;
   outFile << options.getMaxSimTime() << std::endl;
   outFile << "--inject-width" << std::endl;
   outFile << options.getInjectWidth() << std::endl;
   outFile << "--ascii" << std::endl;
   outFile << options.getLogInAscii() << std::endl;
   outFile << "OPTIONS_FINISHED" << std::endl;
   outFile << "X_GRD" << std::endl;
   outFile << X_GRD << std::endl;
   outFile << "Y_GRD" << std::endl;
   outFile << Y_GRD << std::endl;
   outFile << "PI" << std::endl;
   outFile << PI << std::endl;
   outFile << "TPI" << std::endl;
   outFile << TPI << std::endl;
   outFile << "ISEED" << std::endl;
   outFile << ISEED << std::endl;
   outFile << "B0" << std::endl;
   outFile << B0 << std::endl;
   outFile << "P0" << std::endl;
   outFile << P0 << std::endl;
   outFile << "UNIFORM_P0" << std::endl;
   outFile << UNIFORM_P0 << std::endl;
   outFile << "SCALE" << std::endl;
   outFile << SCALE << std::endl;
   outFile << "RATO" << std::endl;
   outFile << RATO << std::endl;
   outFile << "DELT" << std::endl;
   outFile << DELT << std::endl;
   outFile << "SIGMA_CE" << std::endl;
   outFile << SIGMA_CE << std::endl;
   outFile << "SIGMA_CI" << std::endl;
   outFile << SIGMA_CI << std::endl;
   outFile << "SIGMA_HI" << std::endl;
   outFile << SIGMA_HI << std::endl;
   outFile << "SIGMA_HE" << std::endl;
   outFile << SIGMA_HE << std::endl;
   outFile << "SIGMA_HE_PERP" << std::endl;
   outFile << SIGMA_HE_PERP << std::endl;
   outFile << "SIGMA_HI_PERP" << std::endl;
   outFile << SIGMA_HI_PERP << std::endl;
   outFile << "SIGMA_CE_SECONDARY" << std::endl;
   outFile << SIGMA_CE_SECONDARY << std::endl;
   outFile << "PERCENT_SECONDARY" << std::endl;
   outFile << PERCENT_SECONDARY << std::endl;
   outFile << "TSTART" << std::endl;
   outFile << TSTART << std::endl;
   outFile << "LF" << std::endl;
   outFile << LF << std::endl;
   outFile << "DX" << std::endl;
   outFile << DX << std::endl;
   outFile << "DX2" << std::endl;
   outFile << DX2 << std::endl;
   outFile << "DY" << std::endl;
   outFile << DY << std::endl;
   outFile << "TOTA" << std::endl;
   outFile << TOTA << std::endl;
   outFile << "NX" << std::endl;
   outFile << NX << std::endl;
   outFile << "NX1" << std::endl;
   outFile << NX1 << std::endl;
   outFile << "NX12" << std::endl;
   outFile << NX12 << std::endl;
   outFile << "NY" << std::endl;
   outFile << NY << std::endl;
   outFile << "NY1" << std::endl;
   outFile << NY1 << std::endl;
   outFile << "NY12" << std::endl;
   outFile << NY12 << std::endl;
   outFile << "NIJ" << std::endl;
   outFile << NIJ << std::endl;
   outFile << "OOB_PARTICLE" << std::endl;
   outFile << OOB_PARTICLE << std::endl;
   outFile << "SORT_INTERVAL" << std::endl;
   outFile << SORT_INTERVAL << std::endl;
   outFile << "MAX_THREADS_PER_BLOCK" << std::endl;
   outFile << MAX_THREADS_PER_BLOCK << std::endl;
   outFile << "SQUARE_BLOCK_MAX_THREADS_PER_DIM" << std::endl;
   outFile << SQUARE_BLOCK_MAX_THREADS_PER_DIM << std::endl;
   outFile << "WARPSIZE" << std::endl;
   outFile << WARPSIZE << std::endl;
   outFile << "outputPath" << std::endl;
   outFile << outputPath << std::endl;
   outFile << "errorLogName" << std::endl;
   outFile << errorLogName << std::endl;

   /*
   int X_GRD;
   int Y_GRD;
   float PI;
   float TPI;
   unsigned int ISEED;
   float B0;
   double P0;
   bool UNIFORM_P0;
   float SCALE;
   float RATO;
   float DELT;
   float SIGMA_CE;
   float SIGMA_CI;
   float SIGMA_HI;
   float SIGMA_HE;
   float SIGMA_HE_PERP;
   float SIGMA_HI_PERP;
   float SIGMA_CE_SECONDARY;
   double PERCENT_SECONDARY;
   float TSTART;
   int LF;
   float DX;
   float DX2;
   float DY;
   float TOTA;
   int NX;
   int NX1;
   int NX12;
   int NY;
   int NY1;
   int NY12;
   int NIJ;
   float OOB_PARTICLE;
   unsigned int SORT_INTERVAL;
   int MAX_THREADS_PER_BLOCK;
   int SQUARE_BLOCK_MAX_THREADS_PER_DIM;
   int WARPSIZE;
   std::string outputPath;
   std::string errorLogName;
   */
}

void loadPrevSimState(const CommandlineOptions &options,
                      DevMem<float2> &dev_eleHotLoc, DevMem<float3> &dev_eleHotVel, 
                      DevMem<float2> &dev_eleColdLoc, DevMem<float3> &dev_eleColdVel,
                      DevMem<float2> &dev_ionHotLoc, DevMem<float3> &dev_ionHotVel, 
                      DevMem<float2> &dev_ionColdLoc, DevMem<float3> &dev_ionColdVel,
                      unsigned int &numEleHot, unsigned int &numEleCold,
                      unsigned int &numIonHot, unsigned int &numIonCold,
                      const unsigned int allocIncrement)
{
   const unsigned int FPPF = 6; // Floats Per Particle File
   const unsigned int FPPM = 5; // Floats Per Particle Memory
   const std::string& loadDir = options.getRestartDir();

   SimulationState &simState(SimulationState::getRef());

   const std::size_t strSize = 40;
   char infoName[strSize];
   std::string eleName;
   std::string ionName;

   // Generate file names
   sprintf(infoName, "info");
   boost::filesystem::path infoPath(loadDir + "/" + infoName);
   boost::filesystem::path elePath;
   boost::filesystem::path ionPath;

   if(!boost::filesystem::exists(infoPath))
   {
      std::cerr << "ERROR: " << infoPath.string() << " does not exist!" << std::endl;
      exit(1);
   }

   // Now I need to find the right files to load
   std::ifstream infoF(infoPath.string().c_str(), std::ios::binary);
   infoF.seekg(-1, std::ios_base::end);
   std::stringstream infoStream;
   std::string infoLine;
   bool foundFiles = false;
   unsigned int numEle;
   unsigned int numIon;
   unsigned int index;

   infoLine = getPrevLine(infoF);
   infoStream << infoLine;
   while(!foundFiles && infoLine != "")
   {
      infoStream >> simState.iterationNum >> simState.simTime >> numEle >> numIon;
      index = simState.iterationNum;
      std::stringstream s;
      s << "ele_" << std::setw(D_LOG_IDX_WIDTH) << std::setfill('0') << index;
      eleName = s.str();
      s.str("");
      s << "ion_" << std::setw(D_LOG_IDX_WIDTH) << std::setfill('0') << index;
      ionName = s.str();
      elePath = boost::filesystem::path(loadDir) / eleName;
      ionPath = boost::filesystem::path(loadDir) / ionName;

      if(boost::filesystem::exists(elePath) &&
         boost::filesystem::exists(ionPath))
      {
         foundFiles = true;
      }
      infoLine.clear();
      infoLine = getPrevLine(infoF);
		infoStream.clear();
      infoStream << infoLine;
   }

   if(infoLine == "")
   {
      std::cerr << "ERROR: " << infoPath.string() 
         << " is either empty or does not reference any ele or ion files that exist." << std::endl;
      exit(1);
   }

   std::cout << "Restarting run from index " << index << std::endl;

   std::ifstream infoFile(infoPath.string().c_str());

   infoFile.close();
   simState.iterationNum *= D_LF;

   numEleHot = 0;
   numEleCold = 0;
   numIonHot = 0;
   numIonCold = 0;

   std::ifstream eleFile(elePath.string().c_str(), std::ios::binary);
   eleFile.read(reinterpret_cast<char*>(&numEle), sizeof(unsigned int));
   eleFile.read(reinterpret_cast<char*>(&numEleHot), sizeof(unsigned int));
   eleFile.read(reinterpret_cast<char*>(&numEleCold), sizeof(unsigned int));

   if(numEleHot > dev_eleHotLoc.size())
   {
      dev_eleHotLoc.resize(numEleHot + allocIncrement);
      dev_eleHotVel.resize(numEleHot + allocIncrement);
   }
   HostMem<float2> h_eleHotLoc(dev_eleHotLoc.size());
   HostMem<float3> h_eleHotVel(dev_eleHotVel.size());

   if(numEleCold > dev_eleColdLoc.size())
   {
      dev_eleColdLoc.resize(numEleCold + allocIncrement);
      dev_eleColdVel.resize(numEleCold + allocIncrement);
   }
   HostMem<float2> h_eleColdLoc(dev_eleColdLoc.size());
   HostMem<float3> h_eleColdVel(dev_eleColdVel.size());

   for(unsigned int i = 0; i < numEleHot; i++)
   {
      eleFile.read(reinterpret_cast<char*>(&h_eleHotLoc[i]), sizeof(float2));
      eleFile.read(reinterpret_cast<char*>(&h_eleHotVel[i]), sizeof(float3));
   }
   for(unsigned int i = 0; i < numEleCold; i++)
   {
      eleFile.read(reinterpret_cast<char*>(&h_eleColdLoc[i]), sizeof(float2));
      eleFile.read(reinterpret_cast<char*>(&h_eleColdVel[i]), sizeof(float3));
   }
   eleFile.close();

   std::ifstream ionFile(ionPath.string().c_str(), std::ios::binary);
   ionFile.read(reinterpret_cast<char*>(&numIon), sizeof(unsigned int));
   ionFile.read(reinterpret_cast<char*>(&numIonHot), sizeof(unsigned int));
   ionFile.read(reinterpret_cast<char*>(&numIonCold), sizeof(unsigned int));

   if(numIonHot > dev_ionHotLoc.size())
   {
      dev_ionHotLoc.resize(numIonHot + allocIncrement);
      dev_ionHotVel.resize(numIonHot + allocIncrement);
   }
   HostMem<float2> h_ionHotLoc(dev_ionHotLoc.size());
   HostMem<float3> h_ionHotVel(dev_ionHotVel.size());

   if(numIonCold > dev_ionColdLoc.size())
   {
      dev_ionColdLoc.resize(numIonCold + allocIncrement);
      dev_ionColdVel.resize(numIonCold + allocIncrement);
   }
   HostMem<float2> h_ionColdLoc(dev_ionColdLoc.size());
   HostMem<float3> h_ionColdVel(dev_ionColdVel.size());

   for(unsigned int i = 0; i < numIonHot; i++)
   {
      ionFile.read(reinterpret_cast<char*>(&h_ionHotLoc[i]), sizeof(float2));
      ionFile.read(reinterpret_cast<char*>(&h_ionHotVel[i]), sizeof(float3));
   }
   for(unsigned int i = 0; i < numIonCold; i++)
   {
      ionFile.read(reinterpret_cast<char*>(&h_ionColdLoc[i]), sizeof(float2));
      ionFile.read(reinterpret_cast<char*>(&h_ionColdVel[i]), sizeof(float3));
   }
   ionFile.close();

#ifdef _DEBUG
   verifyParticles(&h_eleHotLoc[0], numEleHot, NX1, NY1, "Hot ele");
   verifyParticles(&h_eleColdLoc[0], numEleCold, NX1, NY1, "Cold ele");
   verifyParticles(&h_ionHotLoc[0], numIonHot, NX1, NY1, "Hot ion");
   verifyParticles(&h_ionColdLoc[0], numIonCold, NX1, NY1, "Cold ion");
#endif

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
   // Seek for the last character; for some reason -1 is always a newline
   f.seekg(-2, std::ios_base::end);

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
