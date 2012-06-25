/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

#include "mersenne_twister.h"

#include <stdio.h>

#include "dev_mem.h"
#include "dci.h"
#include "global_variables.h"
#include "pic_utils.h"

__device__ static mt_struct_stripped ds_MT[MT_RNG_COUNT];
static mt_struct_stripped h_MT[MT_RNG_COUNT];



static mt_struct MT[MT_RNG_COUNT];
static uint32_t state[MT_NN];



//extern "C" void initMTRef(const char *fname){
//    
//    FILE *fd = fopen(fname, "rb");
//    if(!fd){
//        printf("initMTRef(): failed to open %s\n", fname);
//        printf("TEST FAILED\n");
//        exit(0);
//    }
//
//    for (int i = 0; i < MT_RNG_COUNT; i++){
//        //Inline structure size for compatibility,
//        //since pointer types are 8-byte on 64-bit systems (unused *state variable)
//        if( !fread(MT + i, 16 /* sizeof(mt_struct) */ * sizeof(int), 1, fd) ){
//            printf("initMTRef(): failed to load %s\n", fname);
//            printf("TEST FAILED\n");
//            exit(0);
//        }
//    }
//
//    fclose(fd);
//}

extern "C" void RandomRef(
    float *h_Random,
    int NPerRng,
    unsigned int seed
){
    int iRng, iOut;

    for(iRng = 0; iRng < MT_RNG_COUNT; iRng++){
        MT[iRng].state = state;
        sgenrand_mt(seed, &MT[iRng]);

        for(iOut = 0; iOut < NPerRng; iOut++)
           h_Random[iRng * NPerRng + iOut] = ((float)genrand_mt(&MT[iRng]) + 1.0f) / 4294967296.0f;
    }
}

//Load twister configurations
void loadMTGPU(const char *fname){
    FILE *fd = fopen(fname, "rb");
    if(!fd){
        printf("initMTGPU(): failed to open %s\n", fname);
        printf("TEST FAILED\n");
        exit(0);
    }
    if( !fread(h_MT, sizeof(h_MT), 1, fd) ){
        printf("initMTGPU(): failed to load %s\n", fname);
        printf("TEST FAILED\n");
        exit(0);
    }
    fclose(fd);
}

__device__ int dev_iState[MT_RNG_COUNT];
__device__ unsigned int dev_mt[MT_RNG_COUNT * MT_NN];

//Initialize/seed twister for current GPU context
void seedMTGPU(unsigned int seed[]){
    int i;
    //Need to be thread-safe
    mt_struct_stripped *MT = (mt_struct_stripped *)malloc(MT_RNG_COUNT * sizeof(mt_struct_stripped));

    for(i = 0; i < MT_RNG_COUNT; i++){
        MT[i]      = h_MT[i];
        MT[i].seed = seed[i];
    }
    cudaMemcpyToSymbol(ds_MT, MT, sizeof(h_MT));
    checkForCudaError("seedMTGPU cudaMemcpyToSymbol");

    free(MT);
}

__global__
void initializeGpuTwister()
{
    const int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    const int THREAD_N = blockDim.x * gridDim.x;
    int iState;
    unsigned int mt[MT_NN];

    for(int iRng = threadX; iRng < MT_RNG_COUNT; iRng += THREAD_N)
    {
        //Load bit-vector Mersenne Twister parameters
        mt_struct_stripped config = ds_MT[iRng];

        //Initialize current state
        mt[0] = config.seed;
        for(iState = 1; iState < MT_NN; iState++)
        {
            mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;
            // Keeping a copy out of the back references is faster than writing directly to global memory
            dev_mt[MT_RNG_COUNT * iState + iRng] = mt[iState];
        }

        dev_iState[iRng] = 0;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Write MT_RNG_COUNT vertical lanes of NPerRng random numbers to *d_Random.
// For coalesced global writes MT_RNG_COUNT should be a multiple of warp size.
// Initial states for each generator are the same, since the states are
// initialized from the global seed. In order to improve distribution properties
// on small NPerRng supply dedicated (local) seed to each twister.
// The local seeds, in their turn, can be extracted from global seed
// by means of any simple random number generator, like LCG.
////////////////////////////////////////////////////////////////////////////////
__global__ void RandomGPU(
    float *d_Random,
    int NPerRng )
{
    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int THREAD_N = blockDim.x * gridDim.x;

    int iState, iState1, iStateM, iOut;
    unsigned int mti, mti1, mtiM, x;
    unsigned int mt[MT_NN];

    for(int iRng = tid; iRng < MT_RNG_COUNT; iRng += THREAD_N){
        //Load bit-vector Mersenne Twister parameters
        mt_struct_stripped config = ds_MT[iRng];

        // Load state from global memory
        for(iState = 1; iState < MT_NN; iState++)
        {
            mt[iState] = dev_mt[MT_RNG_COUNT * iState + iRng];
        }

        iState = dev_iState[iRng];
        mti1 = mt[0];
        for(iOut = 0; iOut < NPerRng; iOut++){
            iState1 = iState + 1;
            iStateM = iState + MT_MM;
            if(iState1 >= MT_NN) iState1 -= MT_NN;
            if(iStateM >= MT_NN) iStateM -= MT_NN;
            mti  = mti1;
            mti1 = mt[iState1];
            mtiM = mt[iStateM];

            x    = (mti & MT_UMASK) | (mti1 & MT_LMASK);
            x    =  mtiM ^ (x >> 1) ^ ((x & 1) ? config.matrix_a : 0);
            mt[iState] = x;
            iState = iState1;

            //Tempering transformation
            x ^= (x >> MT_SHIFT0);
            x ^= (x << MT_SHIFTB) & config.mask_b;
            x ^= (x << MT_SHIFTC) & config.mask_c;
            x ^= (x >> MT_SHIFT1);

            //Convert to (0, 1] float and write to global memory
            d_Random[iRng + iOut * MT_RNG_COUNT] = ((float)x + 1.0f) / 4294967296.0f;
        }

        // Write state back out to global memory
        dev_mt[MT_RNG_COUNT * iState + iRng] = mt[iState];
        dev_iState[iRng] = iState;
    }
}



////////////////////////////////////////////////////////////////////////////////
// Transform each of MT_RNG_COUNT lanes of NPerRng uniformly distributed 
// random samples, produced by RandomGPU(), to normally distributed lanes
// using Cartesian form of Box-Muller transformation.
// NPerRng must be even.
////////////////////////////////////////////////////////////////////////////////
__device__ void BoxMuller(float& u1, float& u2){
    float   r = sqrtf(-2.0f * logf(u1));
    float phi = 2 * D_PI * u2;
    u1 = r * __cosf(phi);
    u2 = r * __sinf(phi);
}

__global__ void BoxMullerGPU(float *d_Random, int NPerRng){
    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int THREAD_N = blockDim.x * gridDim.x;

    for(int iRng = tid; iRng < MT_RNG_COUNT; iRng += THREAD_N)
        for(int iOut = 0; iOut < NPerRng; iOut += 2)
            BoxMuller(
                d_Random[iRng + (iOut + 0) * MT_RNG_COUNT],
                d_Random[iRng + (iOut + 1) * MT_RNG_COUNT]
            );
}

extern "C"
void saveMersenneState(const char *fileName)
{
   FILE *file = fopen(fileName, "wb");
   DevMem<mt_struct_stripped> dev_MT(ds_MT, MT_RNG_COUNT);
   DevMem<uint32_t> dev_state(state, MT_NN);
   std::vector<mt_struct_stripped> h_mt(MT_RNG_COUNT);
   std::vector<uint32_t> h_state(MT_NN);
   dev_MT.copyArrayToHost(&h_mt[0]);
   dev_state.copyArrayToHost(&h_state[0]);
   fclose(file);
}