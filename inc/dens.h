#ifndef DENS_H
#define DENS_H

#include "typedefs.h"

void dens(DevMemF &dev_rho,
          DevMemF &dev_rhoe,
          DevMemF &dev_rhoi,
          const DevMem<float2> &d_eleHotLoc, const DevMem<float2> &d_eleColdLoc,
          const DevMem<float2> &d_ionHotLoc, const DevMem<float2> &d_ionColdLoc,
          unsigned int numHotElectrons, unsigned int numColdElectrons,
          unsigned int numHotIons, unsigned int numColdIons);

#endif
