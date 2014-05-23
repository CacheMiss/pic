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
#ifndef DENS_H
#define DENS_H

#include "dev_stream.h"
#include "typedefs.h"

void dens(DevMemF &dev_rho,
          DevMemF &dev_rhoe,
          DevMemF &dev_rhoi,
          DevMem<float2>& d_eleHotLoc, DevMem<float3>& d_eleHotVel,
          DevMem<float2>& d_eleColdLoc, DevMem<float3>& d_eleColdVel,
          DevMem<float2>& d_ionHotLoc, DevMem<float3>& d_ionHotVel,
          DevMem<float2>& d_ionColdLoc, DevMem<float3>& d_ionColdVel,
          unsigned int& numHotElectrons, unsigned int& numColdElectrons,
          unsigned int& numHotIons,      unsigned int& numColdIons,
          bool sortEleHot, bool sortEleCold,
          bool sortIonHot, bool sortIonCold,
          DevStream &stream1);

#endif
