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

