#include <cuda.h>
#include <float.h>

#include "d_global_variables.h"

//*****************************************************************************
//  Function: injectWriteBlock
//  Purpose: Write a block's worth of particles back out to global memory
//  Parameters:
//  -------------------
//  partArray - The global particle array to write to
//  offset - The offset in the particle array to start at
//  posX[] - The x positions of the particles
//  posY[] - The y positions of the particles
//  velX[] - The x velocities of the particles
//  velY[] - The y velocities of the particles
//  velZ[] - the z velocities of the particles
//*****************************************************************************
__device__
void injectWriteBlock(float2 d_loc[], float3 d_vel[], 
                      int offset, 
                      volatile float s_posX[], volatile float s_posY[], 
                      volatile float s_velX[], volatile float s_velY[], volatile float s_velZ[]
)
{
   int particleStart = blockIdx.x * blockDim.x + offset + threadIdx.x;
   float2 loc;
   float3 vel;

   loc.x = s_posX[threadIdx.x];
   loc.y = s_posY[threadIdx.x];
   d_loc[particleStart] = loc;

   vel.x = s_velX[threadIdx.x];
   vel.y = s_velY[threadIdx.x];
   vel.z = s_velZ[threadIdx.x];
   d_vel[particleStart] = vel;
}

//*****************************************************************************
//  Function: inject
//  Purpose: Inject new particles into the top and bottom cells. Each particle
//           type has NIJ*NX1 new particles created
//  Caveats: This functino expects a block size of 512. To change this behavior
//           set the BLOCKSIZE variable to the desired block size
//  Parameters:
//  -------------------
//  eleHot[] - Two dimensional array represented as one dimension. Each row
//             is a separate particle. Particle values can be indexed as
//             follows:
//                [x*5 + 0] = Position x
//                [x*5 + 1] = Position y
//                [x*5 + 2] = Velocity x
//                [x*5 + 3] = Velocity y
//                [x*5 + 4] = Velocity z
//  eleCold[*5 + 5] - Two dimensional array represented as one dimension. Each row
//             is a separate particle. Particle values can be indexed as
//             follows:
//                [x*5 + 0] = Position x
//                [x*5 + 1] = Position y
//                [x*5 + 2] = Velocity x
//                [x*5 + 3] = Velocity y
//                [x*5 + 4] = Velocity z
//  ionHot[*5 + 5] - Two dimensional array represented as one dimension. Each row
//             is a separate particle. Particle values can be indexed as
//             follows:
//                [x*5 + 0] = Position x
//                [x*5 + 1] = Position y
//                [x*5 + 2] = Velocity x
//                [x*5 + 3] = Velocity y
//                [x*5 + 4] = Velocity z
//  ionCold[*5 + 5] - Two dimensional array represented as one dimension. Each row
//             is a separate particle. Particle values can be indexed as
//             follows:
//                [x*5 + 0] = Position x
//                [x*5 + 1] = Position y
//                [x*5 + 2] = Velocity x
//                [x*5 + 3] = Velocity y
//                [x*5 + 4] = Velocity z
//  DX - The delta between x positions in the grid
//  DY - The delta between y positions in the grid
//  numElectronsHot - The number of hot electrons
//  numElectronsCold - The number of cold electrons
//  numIonsHot - The number of hot ions
//  numIonsCold - The number of cold ions
//  randPool - An array of random numbers ranging from 0 to 1
//  randPoolSize - The number of elements in randPool
//  NX1 - The width of the grid
//  NY1 - The height of the grid
//  NIJ - Avg num particles per cell
//*****************************************************************************
__global__
void inject(float2 eleHotLoc[], float3 eleHotVel[], 
            float2 eleColdLoc[], float3 eleColdVel[],
            float2 ionHotLoc[], float3 ionHotVel[], 
            float2 ionColdLoc[], float3 ionColdVel[],
            const int botXStart, const int injectWidth,
            const float DX, const float DY,
            const int numElectronsHot, const int numElectronsCold, 
            const int numIonsHot, const int numIonsCold,
            float randPool[], const int randPoolSize,
            const unsigned int NX1, const unsigned int NY1,
            const unsigned int NIJ)
{
   const int RANDS_PER_THREAD = 24;
   int randOffset = blockIdx.x * blockDim.x * RANDS_PER_THREAD +
      threadIdx.x * RANDS_PER_THREAD;
   // An shared memory array for new particles
   extern __shared__ float sharedBeg[]; 
   volatile float *posX = sharedBeg;
   volatile float *posY = posX + blockDim.x;
   volatile float *velX = posY + blockDim.x;
   volatile float *velY = velX + blockDim.x;
   volatile float *velZ = velY + blockDim.x;
   // Check and make sure this thread has work, if it doesn't,
   // return here.
   bool hasWork = (blockIdx.x*blockDim.x+threadIdx.x < NIJ*NX1) ? true : false;
   const float velmass = static_cast<float>(1./D_RATO);
   float vpar;
   float tpar; 
   float stpar; // sin of tpar
   float ctpar; // cos of tpar
   //--------------------------------------------------------
   //                    electrons
   //--------------------------------------------------------
   //                     hot e
   //--------------------------------------------------------
   if(hasWork)
   {
      posX[threadIdx.x] = (float)(DX*NX1*randPool[randOffset]);
      posY[threadIdx.x] = (float)(DY*(NY1-1)+DY*randPool[randOffset+1]);
      vpar=(float)((1.414f*rsqrtf(D_SIGMA3))*
         sqrtf(-logf(1.0f-randPool[randOffset+2] + FLT_MIN)));
      tpar = (float)(D_TPI*randPool[randOffset+3] - D_PI);
      // TODO - Figure out why vpar causes the "Double is not supported" warning
      velX[threadIdx.x] = (float)vpar*__sinf((float)tpar);
      vpar=(float)((1.414f*rsqrtf(D_SIGMA3))*
         sqrtf(-logf(1.0f-randPool[randOffset+4] + FLT_MIN)));
      tpar=(float)(D_TPI*randPool[randOffset+5] - D_PI);
      __sincosf(tpar, &stpar, &ctpar);
      velY[threadIdx.x] = vpar*stpar;
      velZ[threadIdx.x] = vpar*ctpar;
      posY[threadIdx.x] = posY[threadIdx.x]+D_DELT*velY[threadIdx.x];

      injectWriteBlock(eleHotLoc, eleHotVel, numElectronsHot, 
         posX, posY, velX, velY, velZ);
   }
   __syncthreads();

   //---------------------------------------------------------
   //                    cold e            
   //---------------------------------------------------------
   if(hasWork)
   {
      posX[threadIdx.x] = (float)(DX*injectWidth*randPool[randOffset+6]+botXStart);
      posY[threadIdx.x] = (float)(DY*randPool[randOffset+7]);
      vpar = (float)((1.414f*rsqrtf(D_SIGMA))*
         sqrtf(-logf(1-randPool[randOffset+8] + FLT_MIN)));
      tpar = (float)(D_TPI*randPool[randOffset+9] - D_PI);
      velX[threadIdx.x] = (float)(vpar*__sinf(tpar));
      vpar = (float)((1.414f/sqrtf(D_SIGMA))*
         sqrtf(-logf(1-randPool[randOffset+10] + FLT_MIN)));
      tpar = (float)(D_TPI*randPool[randOffset+11] - D_PI);
      __sincosf(tpar, &stpar, &ctpar);
      velY[threadIdx.x] = (float)(vpar*stpar);
      velY[threadIdx.x] = static_cast<float>(velY[threadIdx.x]+(1.1f*rsqrtf(D_SIGMA)));
      velZ[threadIdx.x] = vpar*ctpar;
      posY[threadIdx.x] = posY[threadIdx.x]+D_DELT*velY[threadIdx.x];

      posY[threadIdx.x] = max(posY[threadIdx.x], 0.0f);
      injectWriteBlock(eleColdLoc, eleColdVel, numElectronsCold, 
         posX, posY, velX, velY, velZ);
   }
   __syncthreads();

   //---------------------------------------------------------
   // hot ions
   //---------------------------------------------------------
   if(hasWork)
   {
      posX[threadIdx.x]= (float)(DX*NX1*randPool[randOffset+12]);
      posY[threadIdx.x]= (float)(DY*(NY1-1)+DY*randPool[randOffset+13]);
      vpar = (float)((1.414f*rsqrtf(velmass*D_SIGMA2))*
         sqrtf(-logf(1.0f-randPool[randOffset+14] + FLT_MIN)));
      tpar = (float)(D_TPI*randPool[randOffset+15] - D_PI);
      velX[threadIdx.x] = (float)vpar*__sinf((float)tpar);
      vpar = (float)((1.414f*rsqrtf(velmass*D_SIGMA2))*
         sqrtf(-logf(1.0f-randPool[randOffset+16] + FLT_MIN)));
      tpar = (float)(D_TPI*randPool[randOffset+17] - D_PI);
      __sincosf(tpar, &stpar, &ctpar);
      velY[threadIdx.x] = vpar*stpar;
      velY[threadIdx.x] = (float)(velY[threadIdx.x]-(1.5f*rsqrtf(D_SIGMA2*velmass)));
      velZ[threadIdx.x] = vpar*ctpar;
      posY[threadIdx.x] = posY[threadIdx.x]+D_DELT*velY[threadIdx.x];

      injectWriteBlock(ionHotLoc, ionHotVel, numIonsHot, 
         posX, posY, velX, velY, velZ);
   }
   __syncthreads();

   //-------------------------------------------------------
   //            cold ions          
   //-------------------------------------------------------
   if(hasWork)
   {
      posX[threadIdx.x] = (float)(DX*injectWidth*randPool[randOffset+6]+botXStart);
      posY[threadIdx.x] = (float)(DY*randPool[randOffset+19]);
      vpar = (float)((1.414f*rsqrtf(D_SIGMA1*velmass))*
         sqrtf(-logf(1.0f-randPool[randOffset+20] + FLT_MIN)));
      tpar = (float)(D_TPI*randPool[randOffset+21] - D_PI);
      velX[threadIdx.x] = (float)vpar*__sinf((float)tpar);
      vpar = (float)((1.414f*rsqrtf(D_SIGMA1*velmass))*
         sqrtf(-logf(1.0f-randPool[randOffset+22] + FLT_MIN)));
      tpar = (float)(D_TPI*randPool[randOffset+23] - D_PI);
      __sincosf(tpar, &stpar, &ctpar);
      velY[threadIdx.x] = vpar*stpar;
      velZ[threadIdx.x] = vpar*ctpar;
      posY[threadIdx.x] = posY[threadIdx.x]+D_DELT*velY[threadIdx.x];

      posY[threadIdx.x] = max(posY[threadIdx.x], 0.0f);
      injectWriteBlock(ionColdLoc, ionColdVel, numIonsCold, 
         posX, posY, velX, velY, velZ);
   }
}
