#include "d_global_variables.h"

#include <boost/filesystem.hpp>
#include <string>

float B0 = 10;
double P0 = -15;
bool UNIFORM_P0 = false;

int X_GRD = 258;           // number of x grid points + 1
int Y_GRD = 514;          // number of y grid points + 1

int NX = X_GRD - 1;
int NX1 = NX - 1;
int NX12 = NX1 / 2;

int NY = Y_GRD - 1;
int NY1 = NY - 1;
int NY12 = NY1 / 2;

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

