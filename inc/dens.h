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
          DevStream &stream1,
          DevStream &stream2);

#endif
