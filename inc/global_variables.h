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
//const float P0 = static_cast<float>(D_P0);
const float SCALE = static_cast<float>(D_SCALE);
const float RATO = static_cast<float>(D_RATO);
const float DELT = static_cast<float>(D_DELT);
//const float BXM = static_cast<float>(D_BXM);
//const float BYM = static_cast<float>(D_BYM);
//const float BZM = static_cast<float>(D_BZM);
extern float SIGMA_CE; // sigma for cold electrons
extern float SIGMA_CI; // sigma for cold ions
extern float SIGMA_HI; // sigma for hot ions
extern float SIGMA_HE; // sigma for hot electrons

const float TSTART = static_cast<float>(D_TSTART);
//float TMAX = static_cast<float>(D_TMAX);
const int LF = D_LF; // number of iterations between info file outputs
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
