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
#include <cassert>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_functions.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include "error_check.h"

void checkCuda(cudaError_t error)
{
   if(error != cudaSuccess)
   {
      std::stringstream s;
      s << "ERROR: Cuda call failed! ";
      s << cudaGetErrorString(error);
      s << std::endl;
      std::cerr << s.str() << std::endl;
      throw CudaRuntimeError(s.str());
   }
}
