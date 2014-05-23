////////////////////////////////////////////////////////////////////////////////
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
// 
//   http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
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

