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
 
#ifndef MERSENNE_TWISTER_H
#define MERSENNE_TWISTER_H



#define      DCMT_SEED 4172
#define  MT_RNG_PERIOD 607


typedef struct{
    unsigned int matrix_a;
    unsigned int mask_b;
    unsigned int mask_c;
    unsigned int seed;
} mt_struct_stripped;


#define   MT_RNG_COUNT 4096
#define          MT_MM 9
#define          MT_NN 19
#define       MT_WMASK 0xFFFFFFFFU
#define       MT_UMASK 0xFFFFFFFEU
#define       MT_LMASK 0x1U
#define      MT_SHIFT0 12
#define      MT_SHIFTB 7
#define      MT_SHIFTC 15
#define      MT_SHIFT1 18

///////////////////////////////////////////////////////////////////////////////
// Reference MT front-end and Box-Muller transform
///////////////////////////////////////////////////////////////////////////////
extern "C" void initMTRef(const char *fname);
extern "C" void RandomRef(float *h_Random, int NPerRng, unsigned int seed);
extern "C" void BoxMullerRef(float *h_Random, int NPerRng);
void loadMTGPU(const char *fname);
void seedMTGPU(unsigned int seed[]);
__global__ void initializeGpuTwister();
__global__ void RandomGPU(float *d_Random, int NPerRng);
extern "C" void saveMersenneState(const char *fileName);

#endif
