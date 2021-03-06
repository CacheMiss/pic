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
#ifndef GLOBAL_VARIABLES_H
#define GLOBAL_VARIABLES_H

#include <math.h>
#include <string>

#include "d_global_variables.h"

// define system dimensions
//const int NM_PRTS = D_NM_PRTS;     // max number of particles ions or electrons 
extern int X_GRD;           // number of x grid points + 1
extern int Y_GRD;          // number of y grid points + 1

// define system constants
const float PI = static_cast<float>(D_PI);
const float TPI = static_cast<float>(D_TPI);
const unsigned int ISEED = static_cast<unsigned int>(D_ISEED);

extern float B0;
extern double P0;
extern bool UNIFORM_P0;
const float SCALE = static_cast<float>(D_SCALE);
const float RATO = static_cast<float>(D_RATO);
const float DELT = static_cast<float>(D_DELT);
const float BZM = static_cast<float>(D_BZM);
extern float SIGMA_CE; // sigma for cold electrons
extern float SIGMA_CI; // sigma for cold ions
extern float SIGMA_HI; // sigma for hot ions
extern float SIGMA_HE; // sigma for hot electrons
// The perpendicular sigma for hot electrons
// If this is 0 the horizontal sigma is used
extern float SIGMA_HE_PERP;
// The perpendicular sigma for hot ions
// If this is 0 the horizontal sigma is used
extern float SIGMA_HI_PERP;
extern float SIGMA_CE_SECONDARY; // sigma for secondary cold electrons
// The percentage (0-1) of the injected particles which are secondary
extern double PERCENT_SECONDARY;

const float TSTART = static_cast<float>(D_TSTART);
const int LF = D_LF; // number of iterations between info file outputs
const unsigned int LOG_IDX_WIDTH = D_LOG_IDX_WIDTH;
const float DX = static_cast<float>(D_DX);
const float DX2 = static_cast<float>(D_DX2);
const float DY = static_cast<float>(D_DY);
const float TOTA = static_cast<float>(D_TOTA);

// derived parameters
extern int NX;
extern int NX1;
extern int NX12;

extern int NY;
extern int NY1;
extern int NY12;

// avg number of particles per cell?
const int NIJ = D_NIJ; // avg number of particle per cell?

extern float OOB_PARTICLE;

const unsigned int SORT_INTERVAL = 140;

// CUDA Globals
const int MAX_THREADS_PER_BLOCK = D_MAX_THREADS_PER_BLOCK;
const int SQUARE_BLOCK_MAX_THREADS_PER_DIM = 
  static_cast<int>(sqrt(static_cast<double>(MAX_THREADS_PER_BLOCK)));
const int WARPSIZE = 32;

extern std::string outputPath;
extern std::string errorLogName;

#endif
