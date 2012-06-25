#include "d_global_variables.h"

int X_GRD = D_X_GRD;           // number of x grid points + 1
int Y_GRD = D_Y_GRD;          // number of y grid points + 1

int NX = D_NX;
int NX1 = D_NX1;
int NX12 = D_NX12;

int NY = D_NY;
int NY1 = D_NY1;
int NY12 = D_NY12;

float SIGMA = static_cast<float>(D_SIGMA); // sigma for cold electrons
float SIGMA1 = static_cast<float>(D_SIGMA1); // sigma for cold ions
float SIGMA2 = static_cast<float>(D_SIGMA2); // sigma for hot ions
float SIGMA3 = static_cast<float>(D_SIGMA3); // sigma for hot electrons
