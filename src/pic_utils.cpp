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
   std::stringstream name;
   FILE *fp;
   name << outputPath << "/" << fname;
   if(!first)
   {
      fp = fopen(name.str().c_str(), "a");
   }
   else
   {
      fp = fopen(name.str().c_str(), "w");
   }
   if(fp==NULL)
   {
      printf("Cannot open '%s' file for writing\n",name.str().c_str());
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

void saveConfiguration(int argc, char* argv[], 
                       const CommandlineOptions &options, 
                       const std::string& fileName)
{
   std::ofstream outFile(fileName.c_str()); 

   // Skip the executable name
   for(int i = 1; i < argc; i++)
   {
      outFile << argv[i] << std::endl;
   }
   outFile << "ARGUMENTS FINISHED" << std::endl;
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
   outFile << "BZM" << std::endl;
   outFile << BZM << std::endl;
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

template<class T>
bool checkConfigValue(std::ifstream &inFile, const char* label, T expectedValue)
{
   bool success = true;
   std::string line;
   std::getline(inFile, line);
   T localVal;
   std::stringstream s;
   s << line;
   s >> localVal;
   // For things like floats, going to text might have hurt them.
   // For simplicity I mutate the expected value the same way by
   // going to text and back
   T softenedExpectedValue;
   s.str("");
   s.clear();
   s << expectedValue;
   s >> softenedExpectedValue;
   if(localVal != softenedExpectedValue)
   {
      std::cerr << "ERROR: You are attempting to restart with a different value for " << label << std::endl;
      success = false;
   }
   return success;
}

bool checkConfiguration(const std::string& fileName,
                        const CommandlineOptions &options)
{
   std::ifstream inFile(fileName.c_str());
   std::string line;
   bool success = true;

   // Consume the command line arguments at the front
   while(inFile)
   {
      std::getline(inFile, line);
      if(line == "ARGUMENTS FINISHED")
      {
         break;
      }
   }

   // Check things only in the commandline options class
   while(inFile)
   {
      std::getline(inFile, line);
      if(line == "--log-interval")
      {
         // Skip the log level
         // Sometimes there is good reason to use a different one
         std::getline(inFile, line);
      }
      else if(line == "--max-time")
      {
         // Skip the max time, we often want to extend it
         std::getline(inFile, line);
      }
      else if(line == "--inject-width")
      {
         success &= checkConfigValue(inFile, "--inject-width", options.getInjectWidth());
      }
      else if(line == "--ascii")
      {
         // Skip the ascii argument, Its just a logging variation
         std::getline(inFile, line);
      }
      else if(line == "OPTIONS_FINISHED")
      {
         break;
      }
   }
   // Check all the global variable values
   while(inFile)
   {
      std::getline(inFile, line);

      if(line == "X_GRD")
      {
         success &= checkConfigValue(inFile, "X_GRD", X_GRD);
      }
      else if(line == "Y_GRD")
      {
         success &= checkConfigValue(inFile, "Y_GRD", Y_GRD);
      }
      else if(line == "PI")
      {
         success &= checkConfigValue(inFile, "PI", PI);
      }
      else if(line == "TPI")
      {
         success &= checkConfigValue(inFile, "TPI", TPI);
      }
      else if(line == "ISEED")
      {
         success &= checkConfigValue(inFile, "ISEED", ISEED);
      }
      else if(line == "B0")
      {
         success &= checkConfigValue(inFile, "B0", B0);
      }
      else if(line == "P0")
      {
         success &= checkConfigValue(inFile, "P0", P0);
      }
      else if(line == "UNIFORM_P0")
      {
         success &= checkConfigValue(inFile, "UNelse ifORM_P0", UNIFORM_P0);
      }
      else if(line == "SCALE")
      {
         success &= checkConfigValue(inFile, "SCALE", SCALE);
      }
      else if(line == "RATO")
      {
         success &= checkConfigValue(inFile, "RATO", RATO);
      }
      else if(line == "BZM")
      {
         success &= checkConfigValue(inFile, "BZM", BZM);
      }
      else if(line == "DELT")
      {
         success &= checkConfigValue(inFile, "DELT", DELT);
      }
      else if(line == "SIGMA_CE")
      {
         success &= checkConfigValue(inFile, "SIGMA_CE", SIGMA_CE);
      }
      else if(line == "SIGMA_CI")
      {
         success &= checkConfigValue(inFile, "SIGMA_CI", SIGMA_CI);
      }
      else if(line == "SIGMA_HI")
      {
         success &= checkConfigValue(inFile, "SIGMA_HI", SIGMA_HI);
      }
      else if(line == "SIGMA_HE")
      {
         success &= checkConfigValue(inFile, "SIGMA_HE", SIGMA_HE);
      }
      else if(line == "SIGMA_HE_PERP")
      {
         success &= checkConfigValue(inFile, "SIGMA_HE_PERP", SIGMA_HE_PERP);
      }
      else if(line == "SIGMA_HI_PERP")
      {
         success &= checkConfigValue(inFile, "SIGMA_HI_PERP", SIGMA_HI_PERP);
      }
      else if(line == "SIGMA_CE_SECONDARY")
      {
         success &= checkConfigValue(inFile, "SIGMA_CE_SECONDARY", SIGMA_CE_SECONDARY);
      }
      else if(line == "PERCENT_SECONDARY")
      {
         success &= checkConfigValue(inFile, "PERCENT_SECONDARY", PERCENT_SECONDARY);
      }
      else if(line == "TSTART")
      {
         success &= checkConfigValue(inFile, "TSTART", TSTART);
      }
      else if(line == "LF")
      {
         success &= checkConfigValue(inFile, "LF", LF);
      }
      else if(line == "DX")
      {
         success &= checkConfigValue(inFile, "DX", DX);
      }
      else if(line == "DX2")
      {
         success &= checkConfigValue(inFile, "DX2", DX2);
      }
      else if(line == "DY")
      {
         success &= checkConfigValue(inFile, "DY", DY);
      }
      else if(line == "TOTA")
      {
         success &= checkConfigValue(inFile, "TOTA", TOTA);
      }
      else if(line == "NX")
      {
         success &= checkConfigValue(inFile, "NX", NX);
      }
      else if(line == "NX1")
      {
         success &= checkConfigValue(inFile, "NX1", NX1);
      }
      else if(line == "NX12")
      {
         success &= checkConfigValue(inFile, "NX12", NX12);
      }
      else if(line == "NY")
      {
         success &= checkConfigValue(inFile, "NY", NY);
      }
      else if(line == "NY1")
      {
         success &= checkConfigValue(inFile, "NY1", NY1);
      }
      else if(line == "NY12")
      {
         success &= checkConfigValue(inFile, "NY12", NY12);
      }
      else if(line == "NIJ")
      {
         success &= checkConfigValue(inFile, "NIJ", NIJ);
      }
      else if(line == "OOB_PARTICLE")
      {
         success &= checkConfigValue(inFile, "OOB_PARTICLE", OOB_PARTICLE);
      }
      else if(line == "SORT_INTERVAL")
      {
         success &= checkConfigValue(inFile, "SORT_INTERVAL", SORT_INTERVAL);
      }
      else if(line == "MAX_THREADS_PER_BLOCK")
      {
         success &= checkConfigValue(inFile, "MAX_THREADS_PER_BLOCK", MAX_THREADS_PER_BLOCK);
      }
      else if(line == "SQUARE_BLOCK_MAX_THREADS_PER_DIM")
      {
         success &= checkConfigValue(inFile, "SQUARE_BLOCK_MAX_THREADS_PER_DIM", SQUARE_BLOCK_MAX_THREADS_PER_DIM);
      }
      else if(line == "WARPSIZE")
      {
         success &= checkConfigValue(inFile, "WARPSIZE", WARPSIZE);
      }
   }

   return success;
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
   const std::string& loadDir = options.getRestartDir();

   SimulationState &simState(SimulationState::getRef());

   unsigned int index;
   unsigned int numEle;
   unsigned int numIon;
   std::string eleName;
   std::string ionName;

   // Generate file names
   std::string infoName = "info";
   boost::filesystem::path infoPath(loadDir + "/" + infoName);
   boost::filesystem::path elePath;
   boost::filesystem::path ionPath;

   // Generate prev config file name
   std::string configFileName = "configuration.txt";
   boost::filesystem::path configPath(loadDir);
   configPath = configPath / configFileName;

   if(!boost::filesystem::exists(configPath))
   {
      std::cerr << "WARNING: There is no configuration file for this run. There is know way"
         << " to know if these are the right settings to go with this input data" << std::endl;
   }
   else
   {
      bool matchingConfiguration = checkConfiguration(configPath.string(), options);
      if(!matchingConfiguration)
      {
         // checkConfiguration already printed errors to the screen if there were problems
         exit(1);
      }
   }

   if(options.getRestartIdx() == 0)
   {
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
   }
   else
   {
      if(options.getOutputPath() == options.getRestartDir())
      {
         std::cerr << "ERROR: When using --restart-idx, you must specify an output path which "
            << "is different from the path specified in --restart-dir." << std::endl;
         exit(1);
      }
      std::stringstream s;
      s << "ele_" << std::setw(D_LOG_IDX_WIDTH) << std::setfill('0') << options.getRestartIdx();
      eleName = s.str();
      s.str("");
      s << "ion_" << std::setw(D_LOG_IDX_WIDTH) << std::setfill('0') << options.getRestartIdx();
      ionName = s.str();
      elePath = boost::filesystem::path(loadDir) / eleName;
      ionPath = boost::filesystem::path(loadDir) / ionName;

      if(!boost::filesystem::exists(elePath))
      {
         std::cerr << "ERROR: " << eleName << " does not exist!" << std::endl;
         exit(1);
      }
      if(!boost::filesystem::exists(ionPath))
      {
         std::cerr << "ERROR: " << ionName << " does not exist!" << std::endl;
         exit(1);
      }
      index = options.getRestartIdx();
      simState.iterationNum = options.getRestartIdx();
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
