#include "d_global_variables.h"

#include <boost/filesystem.hpp>
#include <string>

float B0 = 10;

int X_GRD = D_X_GRD;           // number of x grid points + 1
int Y_GRD = D_Y_GRD;          // number of y grid points + 1

int NX = D_NX;
int NX1 = D_NX1;
int NX12 = D_NX12;

int NY = D_NY;
int NY1 = D_NY1;
int NY12 = D_NY12;

float SIGMA_CE = static_cast<float>(D_SIGMA_CE); // sigma for cold electrons
float SIGMA_CI = static_cast<float>(D_SIGMA_CI); // sigma for cold ions
float SIGMA_HI = static_cast<float>(D_SIGMA_HI); // sigma for hot ions
float SIGMA_HE = static_cast<float>(D_SIGMA_HE); // sigma for hot electrons

std::string outputPath = "run_output";
std::string errorLogName;