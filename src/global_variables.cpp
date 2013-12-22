#include "d_global_variables.h"

#include <boost/filesystem.hpp>
#include <string>

float B0 = 10;
double P0 = -15;
bool UNIFORM_P0 = false;

int X_GRD = D_X_GRD;           // number of x grid points + 1
int Y_GRD = D_Y_GRD;          // number of y grid points + 1

int NX = D_NX;
int NX1 = D_NX1;
int NX12 = D_NX12;

int NY = D_NY;
int NY1 = D_NY1;
int NY12 = D_NY12;

float SIGMA_CE;
float SIGMA_CI;
float SIGMA_HI;
float SIGMA_HE;
float SIGMA_CE_SECONDARY;
double PERCENT_SECONDARY;
float SIGMA_HE_PERP;
float SIGMA_HI_PERP;

float OOB_PARTICLE;


std::string outputPath = "run_output";
std::string errorLogName;

