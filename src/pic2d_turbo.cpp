//*******************************************************************
// serial version of 2-d pic code 
// last modified: 7/21/2004
// comments: this version reflects changes made by  chakravaritty 
//           deverapalli in the dens subroutine (alterations to
//           rho() calcuation process)
//*******************************************************************
// cpotent2 k2----///nx-not-nx1///oct 1 02
//*******************************************************************
//       using one-dimensional fft and sm(k)
//        theta=0.05 radian bm=0.3
//       bzm = bm*cos(theta) ; bym = bm*sin(theta)   dx=dy=1  ; 
//       o: 9  oxygen ions have no  drift; delt=0.1
//       regular position distribution,   mass ratio =400  bx=1.2
//*******************************************************************
#include <assert.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <vector>

#include <cuda.h>
#include <curand.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "array2d.h"
#include "commandline_options.h"
#include "dens.h"
#include "dev_mem_reuse.h"
#include "dev_stream.h"
#include "device_stats.h"
#include "field.h"
#include "global_variables.h"
#include "inject.h"
#include "logging_thread.h"
#include "logging_types.h"
#include "movep.h"
#include "particle_allocator.h"
#include "phi_avg.h"
#include "pic_utils.h"
#include "potent2.h"
#include "precisiontimer.h"
#include "simulation_state.h"
#include "typedefs.h"

#ifdef _DEBUG
#ifndef DEBUG_TRACE
#define DEBUG_TRACE
#endif
#endif

#define ENABLE_TIMERS

void printFreeMem()
{
   DeviceStats &device(DeviceStats::getRef());

   std::cout << device.getFreeMemMb() << " Mb ("
             << static_cast<int>(device.getPercentFreeMem() * 100)
             << "%) of device memory is free." << std::endl;
}

void executePic(int argc, char *argv[])
{
   CommandlineOptions options;
   options.parseArguments(argc, argv);

   // Create output directory if necessary
   createOutputDir(outputPath.c_str());

   // Clear the old error log
   FILE *errorLog = fopen(errorLogName.c_str(), "w");
   fclose(errorLog);

   int lfint = 5; // number of info intervals between output files
   // The amount of space to get if I run out of space in the particle array
   const std::size_t ALLOC_INCREMENT = 1000000;

   lfint = options.getLogInterval();

   // Init Device
   DeviceStats &ref(DeviceStats::getRef());
   printFreeMem();

   DevStream processingStream[4];

   DevMemReuse &reuseAllocator(DevMemReuse::getRef());
   reuseAllocator.setSizeX(NX1);
   reuseAllocator.setSizeY(NY);

   PrecisionTimer iterationTimer;
#ifdef ENABLE_TIMERS
   PrecisionTimer injectTimer;
   PrecisionTimer densTimer;
   PrecisionTimer potent2Timer;
   PrecisionTimer fieldTimer;
   PrecisionTimer movepTimer;
#endif
   LoggingThread &logger(LoggingThread::getRef());
   
   time_t startTime = time(0);
   time_t stopTime;
   SimulationState &simState(SimulationState::getRef());
   float maxSimTime = options.getMaxSimTime();
   int ind;
   int lfdint;
   int lfd;
   unsigned int nit;

   const std::size_t initialAllocSize = 5000000;
   //const int neededParticles = NIJ*NX1; // Need this many particles in each array
   const int neededParticles = NIJ * options.getInjectWidth(); // Need this many particles in each array
   // 6 rands for hot electrons
   // 6 rands for cold electrons
   // 6 rands for hot ions
   // 6 rands for cold ions
   const int neededRands = neededParticles * 4 * 6;

#ifdef DEBUG_TRACE
   std::cout << "Initializing main storage..." << std::endl;
#endif
   // Device Memory Pointers
   DevMem<float2> d_eleHotLoc(initialAllocSize);
   DevMem<float3> d_eleHotVel(initialAllocSize);
   DevMem<float2> d_eleColdLoc(initialAllocSize);
   DevMem<float3> d_eleColdVel(initialAllocSize);
   DevMem<float2> d_ionHotLoc(initialAllocSize);
   DevMem<float3> d_ionHotVel(initialAllocSize);
   DevMem<float2> d_ionColdLoc(initialAllocSize);
   DevMem<float3> d_ionColdVel(initialAllocSize);
   DevMemF dev_phi(NY * NX1);
   PhiAvg phiAvg(NY * NX1, NX1, NY1);
   PitchedPtr<float> dev_ex(NX1, NY+1); // An extra row is added to pad with zeros
   dev_ex.memset(0);
   PitchedPtr<float> dev_ey(NX1, NY+1); // An extra row is added to pad with zeros
   dev_ey.memset(0);
   DevMemF dev_rho(NX1 * NY);
   DevMemF dev_rhoe(NX1 * NY);
   DevMemF dev_rhoi(NX1 * NY);
   //DevMemF dev_xx(X_GRD);
   //DevMemF dev_yy(Y_GRD);
   DevMemF dev_randTable(neededRands);
   // End Device Memory Pointers

#ifdef DEBUG_TRACE
   std::cout << "Finished main storage" << std::endl;
#endif

   int percentComplete = 0; // Used to display progress to the user
   int percentSize = 0;

#ifdef DEBUG_TRACE
   std::cout << "Initializing random number generator" << std::endl;
#endif

   // Set up the random number generator
   curandGenerator_t randGenerator;
   curandCreateGenerator (&randGenerator, CURAND_RNG_PSEUDO_MTGP32);
   curandSetPseudoRandomGeneratorSeed(randGenerator, ISEED);

   nit = static_cast<int>((maxSimTime-TSTART)/DELT + 1); // determine number of iterations

   percentSize = nit / 100;
   if(percentSize == 0)
   {
      percentSize = 1;
   }

   simState.simTime = TSTART;
   lfd=LF-1;
   ind=1;
   lfdint=0;

   if(options.getRestartDir() != "")
   {
#ifdef DEBUG_TRACE
      std::cout << "Loading previous run data..." << std::endl;
#endif
      loadPrevSimState(options,
         d_eleHotLoc, d_eleHotVel, d_eleColdLoc, d_eleColdVel,
         d_ionHotLoc, d_ionHotVel, d_ionColdLoc, d_ionColdVel,
         simState.numEleHot, simState.numEleCold,
         simState.numIonHot, simState.numIonCold,
         ALLOC_INCREMENT);
      printf("INFO: Loaded %d hot electrons\n", simState.numEleHot);
      printf("INFO: Loaded %d cold electrons\n", simState.numEleCold);
      printf("INFO: Loaded %d hot ions\n", simState.numIonHot);
      printf("INFO: Loaded %d cold ions\n", simState.numIonCold);
      simState.simTime =  (simState.iterationNum * LF + 1) * DELT;
      simState.iterationNum++;
      lfd = 0;
      lfdint = 0;
      ind = simState.iterationNum / LF + 1;
      percentComplete = (int) simState.iterationNum / percentSize;

      // Restore the random number generator state
      std::cout << "Restoring random number state... ";
      for(unsigned int i = 0; i < simState.iterationNum; i++)
      {
         curandGenerateUniform(randGenerator, dev_randTable.getPtr(), neededRands);
      }
      std::cout << "Finished!" << std::endl;
#ifdef DEBUG_TRACE
      std::cout << "previous run data loaded" << std::endl;
#endif
   }
   else if(boost::filesystem::exists(boost::filesystem::path(outputPath) / "info"))
   {
      std::cout << "ERROR: There is already run output in " << outputPath
         << ". Please resolve this before continuing." << std::endl;
      exit(1);
   }
   else
   {
      simState.iterationNum = 0;
      boost::filesystem::path fileName(outputPath);
      fileName /= "configuration.txt";
      saveConfiguration(argc, argv, options, fileName.string());
   }
   // DEBUG
   //   {
   //      Array2dF *eleHot = new Array2dF(simState.numEleHot, 5);
   //      Array2dF *eleCold = new Array2dF(simState.numEleCold, 5);
   //      Array2dF *ionHot = new Array2dF(simState.numIonHot, 5);
   //      Array2dF *ionCold = new Array2dF(simState.numIonCold, 5);
   //      eleHot->loadRows(dev_eleHot, simState.numEleHot);
   //      eleCold->loadRows(dev_eleCold, simState.numEleCold);
   //      ionHot->loadRows(dev_ionHot, simState.numIonHot);
   //      ionCold->loadRows(dev_ionCold, simState.numIonCold);
   //      logger.logParticlesBinary(ind, eleHot, eleCold, ionHot, ionCold,
   //         simState.numEleHot, simState.numEleCold,
   //         simState.numIonHot, simState.numIonCold);
   //   }
   //   logger.flush();
   // END DEBUG

   std::cout << "Free mem after initial allocations:" << std::endl;
   printFreeMem();

   printf("nit=%d\n",nit);
   for (;simState.iterationNum<nit; simState.iterationNum++) 
   {
      if(percentComplete < 100 &&
         static_cast<int>(simState.iterationNum / percentSize) > percentComplete)
      {
         percentComplete = (int) simState.iterationNum / percentSize;
         printf("%d%% Complete\n", percentComplete);
         printFreeMem();
      }

      if (lfd == 0) 
      {
         iterationTimer.start();
      }

      simState.simTime = simState.iterationNum * static_cast<double>(DELT);
      lfd++;

      // Make sure I'm not out of memory
      if(simState.numEleHot + neededParticles > d_eleHotLoc.size())
      {
         std::cout << "Adding storage for hot electrons." << std::endl;
         // Scope these to keep the total required memory size lower
         {
            HostMem<float2> pos;
            pos = d_eleHotLoc;
            d_eleHotLoc.resize(d_eleHotLoc.size() + ALLOC_INCREMENT);
            d_eleHotLoc.copyArrayToDev(pos);
         }
         {
            HostMem<float3> vel;
            vel = d_eleHotVel;
            d_eleHotVel.resize(d_eleHotVel.size() + ALLOC_INCREMENT);
            d_eleHotVel.copyArrayToDev(vel);
         }
      }
      if(simState.numEleCold + neededParticles > d_eleColdLoc.size())
      {
         std::cout << "Adding storage for cold electrons." << std::endl;
         // Scope these to keep the total required memory size lower
         {
            HostMem<float2> pos;
            pos = d_eleColdLoc;
            d_eleColdLoc.resize(d_eleColdLoc.size() + ALLOC_INCREMENT);
            d_eleColdLoc.copyArrayToDev(pos);
         }
         {
            HostMem<float3> vel;
            vel = d_eleColdVel;
            d_eleColdVel.resize(d_eleColdVel.size() + ALLOC_INCREMENT);
            d_eleColdVel.copyArrayToDev(vel);
         }
      }
      if(simState.numIonHot + neededParticles > d_ionHotLoc.size())
      {
         std::cout << "Adding storage for hot ions." << std::endl;
         // Scope these to keep the total required memory size lower
         {
            HostMem<float2> pos;
            pos = d_ionHotLoc;
            d_ionHotLoc.resize(d_ionHotLoc.size() + ALLOC_INCREMENT);
            d_ionHotLoc.copyArrayToDev(pos);
         }
         {
            HostMem<float3> vel;
            vel = d_ionHotVel;
            d_ionHotVel.resize(d_ionHotVel.size() + ALLOC_INCREMENT);
            d_ionHotVel.copyArrayToDev(vel);
         }
      }
      if(simState.numIonCold + neededParticles > d_ionColdLoc.size())
      {
         std::cout << "Adding storage for cold ions." << std::endl;
         // Scope these to keep the total required memory size lower
         {
            HostMem<float2> pos;
            pos = d_ionColdLoc;
            d_ionColdLoc.resize(d_ionColdLoc.size() + ALLOC_INCREMENT);
            d_ionColdLoc.copyArrayToDev(pos);
         }
         {
            HostMem<float3> vel;
            vel = d_ionColdVel;
            d_ionColdVel.resize(d_ionColdVel.size() + ALLOC_INCREMENT);
            d_ionColdVel.copyArrayToDev(vel);
         }
      }

#ifdef DEBUG_TRACE
      std::cout << "Inject" << std::endl;
#endif

      // Prepare to call Inject
      // Generate the random numbers inject will need
      curandGenerateUniform(randGenerator, dev_randTable.getPtr(), neededRands);

      const unsigned int injectWidth = options.getInjectWidth();
      const unsigned int injectStartX = (NX1 / 2) - (injectWidth / 2);
#ifdef ENABLE_TIMERS
      injectTimer.start();
#endif
      unsigned int neededSecondaryParticles = static_cast<unsigned int>(PERCENT_SECONDARY * neededParticles);
      // randomly inject new particles in top and bottom 
      inject(
         d_eleHotLoc, d_eleHotVel, 
         d_eleColdLoc, d_eleColdVel, 
         d_ionHotLoc, d_ionHotVel, 
         d_ionColdLoc, d_ionColdVel, 
         DX, DY,
         simState.numEleHot, simState.numEleCold, 
         simState.numIonHot, simState.numIonCold,
			neededParticles,
         neededSecondaryParticles,
         dev_randTable,
         NX1, NY1,
         SIGMA_HE, SIGMA_HI,
         SIGMA_CE, SIGMA_CI,
         SIGMA_HE_PERP, SIGMA_HI_PERP,
         SIGMA_CE_SECONDARY,
			injectWidth,
			injectStartX,
			processingStream[0]
         );

#ifdef ENABLE_TIMERS
      processingStream[0].synchronize();
      injectTimer.stop();
#endif

      // DEBUG
      //processingStream[0].synchronize();
      //logger.pushLogItem(
      //   new LogParticlesAscii(ind, d_eleHotLoc, d_eleHotVel,
      //   d_eleColdLoc, d_eleColdVel,
      //   d_ionHotLoc, d_ionHotVel,
      //   d_ionColdLoc, d_ionColdVel,
      //   simState.numEleHot, simState.numEleCold,
      //   simState.numIonHot, simState.numIonCold));
      //logger.flush();
      // END DEBUG

      unsigned int itRemainder = simState.iterationNum % SORT_INTERVAL;
      bool sortEleHot = false;
      bool sortIonHot = false;
      bool sortEleCold = false;
      bool sortIonCold = false;
      switch(itRemainder)
      {
      case 1:
         sortIonHot = true;
         break;
      case 2:
         sortIonCold = true;
         break;
      case 3:
         sortEleHot = true;
         break;
      case 4:
         sortEleCold = true;
         break;
      }

      unsigned int activeEleHot = simState.numEleHot;
      unsigned int activeEleCold = simState.numEleCold;
      unsigned int activeIonHot = simState.numIonHot;
      unsigned int activeIonCold = simState.numIonCold;
#ifdef DEBUG_TRACE
      std::cout << "Dens" << std::endl;
#endif
#ifdef ENABLE_TIMERS
      densTimer.start();
#endif
      // determine the charge density at the grid points
      dens(dev_rho, dev_rhoe,dev_rhoi, 
           d_eleHotLoc, d_eleHotVel,
           d_eleColdLoc, d_eleColdVel,
           d_ionHotLoc, d_ionHotVel,
           d_ionColdLoc, d_ionColdVel,
           activeEleHot, activeEleCold, 
           activeIonHot, activeIonCold,
           sortEleHot, sortEleCold,
           sortIonHot, sortIonCold,
           processingStream[0]);
#ifdef ENABLE_TIMERS
      processingStream[0].synchronize();
      densTimer.stop();
#endif

      // Hot Electron Count Update
      if(sortEleHot)
      {
         simState.numEleHot = activeEleHot;
      }
      // Cold Electron Count Update
      if(sortEleCold)
      {
         simState.numEleCold = activeEleCold;
      }
      // Hot Ion Count Update
      if(sortIonHot)
      {
         simState.numIonHot = activeIonHot;
      }
      // Cold Ion Count Update
      if(sortIonCold)
      {
         simState.numIonCold = activeIonCold;
      }

      // Start DEBUG
      //processingStream[0].synchronize();
      //Array2dF *rho = new Array2dF(NY, NX1);
      //Array2dF *rhoe = new Array2dF(NY, NX1);
      //Array2dF *rhoi = new Array2dF(NY, NX1);
      //*rho = dev_rho;
      //*rhoe = dev_rhoe;
      //*rhoi = dev_rhoi;
      //logger.logRhoAscii(ind, rho, rhoe, rhoi);
      //logger.pushLogItem(
      //   new LogParticlesAscii(ind, d_eleHotLoc, d_eleHotVel,
      //   d_eleColdLoc, d_eleColdVel,
      //   d_ionHotLoc, d_ionHotVel,
      //   d_ionColdLoc, d_ionColdVel,
      //   simState.numEleHot, simState.numEleCold,
      //   simState.numIonHot, simState.numIonCold));
      //logger.flush();
      // End DEBUG

#ifdef ENABLE_TIMERS
      potent2Timer.start();
#endif
      // calculate potential at Grid points
      potent2(dev_phi, dev_rho, processingStream[0]);
#ifdef ENABLE_TIMERS
      processingStream[0].synchronize();
      potent2Timer.stop();
#endif

#ifdef DEBUG_TRACE
      std::cout << "Field" << std::endl;
#endif
#ifdef ENABLE_TIMERS
      fieldTimer.start();
#endif
      //calculate E field at Grid points
      field(dev_ex,dev_ey,dev_phi,
         processingStream[0],
         processingStream[1]);
#ifdef ENABLE_TIMERS
      processingStream[0].synchronize();
      processingStream[1].synchronize();
      fieldTimer.stop();
#endif

      // DEBUG
      //processingStream[0].synchronize();
      // logger.pushLogItem(
      //    new LogParticlesAscii(ind, d_eleHotLoc, d_eleHotVel,
      //    d_eleColdLoc, d_eleColdVel,
      //    d_ionHotLoc, d_ionHotVel,
      //    d_ionColdLoc, d_ionColdVel,
      //    simState.numEleHot, simState.numEleCold,
      //    simState.numIonHot, simState.numIonCold));
      // logger.flush();
      // END DEBUG

#ifdef ENABLE_TIMERS
      movepTimer.start();
#endif
      processingStream[0].synchronize();
      // move ions
#ifdef DEBUG_TRACE
      std::cout << "MoveHi" << std::endl;
#endif
      processingStream[0].synchronize();
      processingStream[1].synchronize();
      movep(d_ionHotLoc, d_ionHotVel, simState.numIonHot, 
         RATO, dev_ex, dev_ey, processingStream[0], true);
#ifdef DEBUG_TRACE
      std::cout << "MoveCi" << std::endl;
#endif
      movep(d_ionColdLoc, d_ionColdVel, simState.numIonCold, 
         RATO, dev_ex, dev_ey, processingStream[1], false);

      // move electrons
#ifdef DEBUG_TRACE
      std::cout << "MoveHe" << std::endl;
#endif
      movep(d_eleHotLoc, d_eleHotVel, simState.numEleHot, 
         (float) -1.0, dev_ex, dev_ey, processingStream[2], false);
#ifdef DEBUG_TRACE
      std::cout << "MoveCe" << std::endl;
#endif
      movep(d_eleColdLoc, d_eleColdVel, simState.numEleCold, 
         (float) -1.0, dev_ex, dev_ey, processingStream[3], false);

#ifdef ENABLE_TIMERS
      processingStream[0].synchronize();
      processingStream[1].synchronize();
      processingStream[2].synchronize();
      processingStream[3].synchronize();
      movepTimer.stop();
#endif

      // DEBUG
      //processingStream[0].synchronize();
      //logger.pushLogItem(
      //   new LogParticlesAscii(ind, d_eleHotLoc, d_eleHotVel,
      //   d_eleColdLoc, d_eleColdVel,
      //   d_ionHotLoc, d_ionHotVel,
      //   d_ionColdLoc, d_ionColdVel,
      //   simState.numEleHot, simState.numEleCold,
      //   simState.numIonHot, simState.numIonCold));
      //logger.flush();
      // END DEBUG

      std::size_t iterationsUntilLog = simState.iterationNum % (LF * lfint) + 1;
      if(iterationsUntilLog < static_cast<std::size_t>(50.0 / DELT))
      {
         phiAvg.addPhi(dev_phi);
      }

      if (simState.iterationNum != 0 && simState.iterationNum % LF == 0)
      {
         iterationTimer.stop();
         processingStream[0].synchronize();
         processingStream[1].synchronize();
         processingStream[2].synchronize();
         processingStream[3].synchronize();
         logger.logInfo(ind, simState.simTime, 
            activeEleHot + activeEleCold,
            activeIonHot + activeIonCold,
            options.getRestartDir() != "" ? true : false);
         double iterationTime = static_cast<double>(iterationTimer.intervalInMicroS()) / (LF * 1000);
         double injectTime = static_cast<double>(injectTimer.intervalInNanoS()) / 1000000;
         double densTime = static_cast<double>(densTimer.intervalInNanoS()) / 1000000;
         double potent2Time = static_cast<double>(potent2Timer.intervalInNanoS()) / 1000000;
         double fieldTime = static_cast<double>(fieldTimer.intervalInNanoS()) / 1000000;
         double movepTime = static_cast<double>(movepTimer.intervalInNanoS()) / 1000000;
         logger.pushLogItem(new LogForPerformance(
            ind, simState.simTime, 
            activeEleHot,
            activeEleCold, 
            activeIonHot,
            activeIonCold, 
            iterationTime,
#ifdef ENABLE_TIMERS
            injectTime,
            densTime,
            potent2Time,
            fieldTime,
            movepTime,
#else
            0, 0, 0, 0, 0,
#endif
            options.getRestartDir() != "" ? true : false));
         lfdint = lfdint + 1;
         if (ind % options.getLogInterval() == 0) 
         {
            Array2dF *phi = new Array2dF(NY, NX1);
            Array2dF *ex = new Array2dF(NY+1, NX1);
            Array2dF *ey = new Array2dF(NY+1, NX1);
            Array2dF *rho = new Array2dF(NY, NX1);
            Array2dF *rhoe = new Array2dF(NY, NX1);
            Array2dF *rhoi = new Array2dF(NY, NX1);

            // Move computations back to host
            *rho = dev_rho;
            *rhoe = dev_rhoe;
            *rhoi = dev_rhoi;
            *phi = dev_phi;
            *ex = dev_ex;
            *ey = dev_ey;
      
            if(options.getLogInAscii())
            {
               logger.pushLogItem(
                  new LogParticlesAscii(ind, d_eleHotLoc, d_eleHotVel,
                  d_eleColdLoc, d_eleColdVel,
                  d_ionHotLoc, d_ionHotVel,
                  d_ionColdLoc, d_ionColdVel,
                  simState.numEleHot, simState.numEleCold,
                  simState.numIonHot, simState.numIonCold));
               logger.logRhoAscii(ind, rho, rhoe, rhoi);
               logger.logPhiAscii(ind, phi);
            }
            else
            {
               logger.pushLogItem(
                  new LogParticlesBinary(ind, d_eleHotLoc, d_eleHotVel,
                  d_eleColdLoc, d_eleColdVel,
                  d_ionHotLoc, d_ionHotVel,
                  d_ionColdLoc, d_ionColdVel,
                  simState.numEleHot, simState.numEleCold,
                  simState.numIonHot, simState.numIonCold));
               logger.logRhoBinary(ind, rho, rhoe, rhoi);
               logger.logPhiBinary(ind, phi);
            }

            logger.pushLogItem(new LogAvgPhi(ind, phiAvg));
            logger.pushLogItem(new LogFieldBinary(ind, ex, ey));
            phiAvg.clear();

            lfdint = 0;

				// Exit if we've been asked to
				boost::filesystem::path stopFile(outputPath);
				stopFile /= "stop";
				if(boost::filesystem::exists(stopFile))
				{
					boost::filesystem::remove(stopFile);
					break;
				}
         }
         lfd=0 ;
         ind=ind+1;
      }
   }

   stopTime = time(0);
   unsigned int timeMin = (unsigned int)(stopTime - startTime) / 60;
   unsigned int timeSec = (unsigned int)(stopTime - startTime) % 60;

   std::string runStatisticsFn = (boost::filesystem::path(outputPath) /= "run_statistics.txt").string();
   FILE *f = fopen(runStatisticsFn.c_str(), "w");
   fprintf(f, "nit %u reached at %u min %u sec\n", nit, timeMin, timeSec);
   fclose(f);

   logger.flush();

}

int main(int argc, char *argv[])
{
   try
   {
      DeviceStats &ref(DeviceStats::getRef());
   }
   catch(CudaRuntimeError e)
   {
      std::cout << e.what() << std::endl;
      throw;
   }
#ifndef _DEBUG
   try
   {
#endif
      executePic(argc, argv);
#ifndef _DEBUG
   }
   catch(CudaRuntimeError e)
   {
      std::cout << e.what() << std::endl;
      boost::filesystem::path fileName(outputPath);
      fileName /= "caught_exception.txt";
      std::ofstream f(fileName.c_str());
      f << e.what() << std::endl;
      f.close();
      ParticleAllocator::getRef().cleanup();
      DevMemReuse::getRef().cleanup();
      cudaDeviceReset();
      throw;
   }
#endif

   ParticleAllocator::getRef().cleanup();
   DevMemReuse::getRef().cleanup();
   cudaDeviceReset();

   return 0;
}