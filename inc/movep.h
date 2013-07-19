#ifndef MOVEP_H
#define MOVEP_H

#include "typedefs.h"
#include "pitched_ptr.h"

void movep(DevMem<float2> &partLoc, DevMem<float3> &partVel,
           unsigned int &numParticles, float mass,
           const PitchedPtr<float> &ex, const PitchedPtr<float> &ey,
           cudaStream_t &stream);

#endif

