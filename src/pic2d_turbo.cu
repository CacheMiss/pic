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
#include "device_stats.h"
#include "device_utils.h"
#include "field.h"
#include "global_variables.h"
#include "inject.h"
#include "logging_thread.h"
#include "logging_types.h"
#include "movep.h"
#include "particle_allocator.h"
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

void printFreeMem()
{
   DeviceStats &device(DeviceStats::getRef());

   std::cout << device.getFreeMemMb() << " Mb ("
             << static_cast<int>(device.getPercentFreeMem() * 100)
             << "%) of device memory is free." << std::endl;
}

void executePic(int argc, char *argv[])
{
   // Create output directory if necessary
   createOutputDir("run_output");

   // Clear the old error log
   FILE *errorLog = fopen(errorLogName.c_str(), "w");
   fclose(errorLog);

   int lfint = 5; // number of info intervals between output files

   CommandlineOptions &options(CommandlineOptions::getRef());
   options.parseArguments(argc, argv);
   lfint = options.getLogInterval();

   // Init Device
   DeviceStats &ref(DeviceStats::getRef());
   printFreeMem();

   DevMemReuse &reuseAllocator(DevMemReuse::getRef());
   reuseAllocator.setSizeX(NX1);
   reuseAllocator.setSizeY(NY);

   PrecisionTimer iterationTimer;
   //PrecisionTimer injectTimer;
   //PrecisionTimer densTimer;
   //PrecisionTimer potent2Timer;
   //PrecisionTimer fieldTimer;
   //PrecisionTimer movepTimer;
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
   const int neededParticles = NIJ*NX1; // Need this many particles in each array
   // 6 rands for hot electrons
   // 6 rands for cold electrons
   // 6 rands for hot ions
   // 6 rands for cold ions
   const int neededRands = neededParticles * 4 * 6;

#ifdef DEBUG_TRACE
   std::cout << "Initializing main storage..." << std::endl;
#endif
   // CUDA Variables
   int sharedMemoryBytes;
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
   DevMemF dev_ex((NY+1) * NX1); // An extra row is added to pad with zeros
   dev_ex.zeroMem();
   DevMemF dev_ey((NY+1) * NX1); // An extra row is added to pad with zeros
   dev_ey.zeroMem();
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

   simState.simTime = TSTART;
   lfd=LF-1;
   ind=0;
   lfdint=0;

   if(options.getRestartPoint() > 0)
   {
#ifdef DEBUG_TRACE
      std::cout << "Loading previous run data..." << std::endl;
#endif
      loadPrevSimState(options.getRestartPoint(), options.getRestartDir(),
         d_eleHotLoc, d_eleHotVel, d_eleColdLoc, d_eleColdVel,
         d_ionHotLoc, d_ionHotVel, d_ionColdLoc, d_ionColdVel,
         simState.numEleHot, simState.numEleCold,
         simState.numIonHot, simState.numIonCold);
      printf("INFO: Loaded %d hot electrons\n", simState.numEleHot);
      printf("INFO: Loaded %d cold electrons\n", simState.numEleCold);
      printf("INFO: Loaded %d hot ions\n", simState.numIonHot);
      printf("INFO: Loaded %d cold ions\n", simState.numIonCold);
      simState.iterationNum++;
      simState.simTime += DELT;
      lfd = 0;
      lfdint = 0;
      ind = simState.iterationNum / lfint + 1;
#ifdef DEBUG_TRACE
      std::cout << "previous run data loaded" << std::endl;
#endif
   }
   else
   {
      simState.iterationNum = 0;
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

      iterationTimer.start();

      simState.simTime +=DELT;
      lfd++;

      // Make sure I'm not out of memory
      const std::size_t ALLOC_INCREMENT = 1000000;
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

      //injectTimer.start();
      const int injectThreadsPerBlock = MAX_THREADS_PER_BLOCK;
      dim3 injectNumBlocks(calcNumBlocks(injectThreadsPerBlock, neededParticles));
      dim3 injectBlockSize(injectThreadsPerBlock);
      sharedMemoryBytes = sizeof(float) * 5 * injectThreadsPerBlock;
      cudaThreadSynchronize();
      checkForCudaError("RandomGPU");
      // randomly inject new particles in top and bottom 
      inject<<<injectNumBlocks, injectBlockSize, sharedMemoryBytes>>>(
         d_eleHotLoc.getPtr(), d_eleHotVel.getPtr(), 
         d_eleColdLoc.getPtr(), d_eleColdVel.getPtr(), 
         d_ionHotLoc.getPtr(), d_ionHotVel.getPtr(), 
         d_ionColdLoc.getPtr(), d_ionColdVel.getPtr(), 
         DX, DY,
         simState.numEleHot, simState.numEleCold, 
         simState.numIonHot, simState.numIonCold,
         dev_randTable.getPtr(),
         dev_randTable.size(),
         NX1, NY1, NIJ
         );
      checkForCudaError("Inject failed");

      simState.numEleHot += neededParticles;
      simState.numEleCold += neededParticles;
      simState.numIonHot += neededParticles;
      simState.numIonCold += neededParticles;
      //cudaThreadSynchronize();
      //injectTimer.stop();

      // DEBUG
      //cudaThreadSynchronize();
      //logger.pushLogItem(
      //   new LogParticlesAscii(ind, d_eleHotLoc, d_eleHotVel,
      //   d_eleColdLoc, d_eleColdVel,
      //   d_ionHotLoc, d_ionHotVel,
      //   d_ionColdLoc, d_ionColdVel,
      //   simState.numEleHot, simState.numEleCold,
      //   simState.numIonHot, simState.numIonCold));
      //logger.flush();
      // END DEBUG

#ifdef DEBUG_TRACE
      std::cout << "Dens" << std::endl;
#endif
      //densTimer.start();
      // determine the charge density at the grid points
      dens(dev_rho, dev_rhoe,dev_rhoi, 
           d_eleHotLoc, d_eleColdLoc,
           d_ionHotLoc, d_ionColdLoc,
           simState.numEleHot, simState.numEleCold, 
           simState.numIonHot, simState.numIonCold);
      //cudaThreadSynchronize();
      //densTimer.stop();

      // Start DEBUG
      //cudaThreadSynchronize();
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

      //potent2Timer.start();
      // calculate potential at Grid points
      potent2(dev_phi, dev_rho);
      //cudaThreadSynchronize();
      //potent2Timer.stop();

#ifdef DEBUG_TRACE
      std::cout << "Field" << std::endl;
#endif
      //fieldTimer.start();
      // calculate E field at Grid points
      field(dev_ex,dev_ey,dev_phi);
      //cudaThreadSynchronize();
      //fieldTimer.stop();

      // DEBUG
      // cudaThreadSynchronize();
      // logger.pushLogItem(
      //    new LogParticlesAscii(ind, d_eleHotLoc, d_eleHotVel,
      //    d_eleColdLoc, d_eleColdVel,
      //    d_ionHotLoc, d_ionHotVel,
      //    d_ionColdLoc, d_ionColdVel,
      //    simState.numEleHot, simState.numEleCold,
      //    simState.numIonHot, simState.numIonCold));
      // logger.flush();
      // END DEBUG

      //movepTimer.start();
      // move ions
      cudaStream_t movepStreams[4];
      for(int streamIdx = 0; streamIdx < 4; streamIdx++)
      {
         cudaStreamCreate(&movepStreams[streamIdx]);
      }
      cudaThreadSynchronize();
#ifdef DEBUG_TRACE
      std::cout << "MoveHi" << std::endl;
#endif
      movep(d_ionHotLoc, d_ionHotVel, simState.numIonHot, 
         RATO, dev_ex, dev_ey, movepStreams[0]);
#ifdef DEBUG_TRACE
      std::cout << "MoveCi" << std::endl;
#endif
      movep(d_ionColdLoc, d_ionColdVel, simState.numIonCold, 
         RATO, dev_ex, dev_ey, movepStreams[1]);

      // move electrons
#ifdef DEBUG_TRACE
      std::cout << "MoveHe" << std::endl;
#endif
      movep(d_eleHotLoc, d_eleHotVel, simState.numEleHot, 
         (float) -1.0, dev_ex, dev_ey, movepStreams[2]);
#ifdef DEBUG_TRACE
      std::cout << "MoveCe" << std::endl;
#endif
      movep(d_eleColdLoc, d_eleColdVel, simState.numEleCold, 
         (float) -1.0, dev_ex, dev_ey, movepStreams[3]);
      for(int streamIdx = 0; streamIdx < 4; streamIdx++)
      {
         cudaStreamDestroy(movepStreams[streamIdx]);
      }

      //cudaThreadSynchronize();
      //movepTimer.stop();

      // DEBUG
      //cudaThreadSynchronize();
      //logger.pushLogItem(
      //   new LogParticlesAscii(ind, d_eleHotLoc, d_eleHotVel,
      //   d_eleColdLoc, d_eleColdVel,
      //   d_ionHotLoc, d_ionHotVel,
      //   d_ionColdLoc, d_ionColdVel,
      //   simState.numEleHot, simState.numEleCold,
      //   simState.numIonHot, simState.numIonCold));
      //logger.flush();
      // END DEBUG

      iterationTimer.stop();

      if (lfd >= LF) 
      {
         cudaThreadSynchronize();
         logger.logInfo(ind, simState.simTime, 
            simState.numEleHot + simState.numEleCold,
            simState.numIonHot + simState.numIonCold);
         logger.logForPerformance(ind, simState.simTime, 
            simState.numEleHot, simState.numEleCold, 
            simState.numIonHot, simState.numIonCold, 
            (unsigned int) iterationTimer.intervalInMilliS(),
            0, 0, 0, 0, 0,
            options.getRestartPoint() > 0 ? true : false);
            //(unsigned int) injectTimer.intervalInMilliS(),
            //(unsigned int) densTimer.intervalInMilliS(),
            //(unsigned int) potent2Timer.intervalInMilliS(),
            //(unsigned int) fieldTimer.intervalInMilliS(),
            //(unsigned int) movepTimer.intervalInMilliS());
         lfdint = lfdint + 1;
         if (lfdint >= lfint) 
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
            lfdint = 0;
         }
         lfd=0 ;
         ind=ind+1;
      }
   }

   stopTime = time(0);
   unsigned int timeMin = (unsigned int)(stopTime - startTime) / 60;
   unsigned int timeSec = (unsigned int)(stopTime - startTime) % 60;

   std::string runStatisticsFn = outputDir + "/run_statistics.txt";
   FILE *f = fopen(runStatisticsFn.c_str(), "w");
   fprintf(f, "nit %u reached at %u min %u sec\n", nit, timeMin, timeSec);
   fclose(f);

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
   try
   {
      executePic(argc, argv);
   }
   catch(CudaRuntimeError e)
   {
      std::cout << e.what() << std::endl;
      ParticleAllocator::getRef().cleanup();
      DevMemReuse::getRef().cleanup();
      cudaDeviceReset();
      throw;
   }

   ParticleAllocator::getRef().cleanup();
   DevMemReuse::getRef().cleanup();
   cudaDeviceReset();

   return 0;
}
