#ifndef MOVEP_H
#define MOVEP_H

#include "typedefs.h"

void movep(DevMem<float2> &partLoc, DevMem<float3> &partVel,
           unsigned int &numParticles, float mass,
           const DevMemF &ex, const DevMemF &ey,
           cudaStream_t &stream);

#endif

