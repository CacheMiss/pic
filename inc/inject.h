#ifndef INJECT_H
#define INJECT_H

#include "dev_stream.h"

void inject(DevMem<float2>& eleHotLoc, DevMem<float3>& eleHotVel, 
            DevMem<float2>& eleColdLoc, DevMem<float3>& eleColdVel,
            DevMem<float2>& ionHotLoc, DevMem<float3>& ionHotVel, 
            DevMem<float2>& ionColdLoc, DevMem<float3>& ionColdVel,
            const float DX, const float DY,
            unsigned int &numElectronsHot, unsigned int &numElectronsCold, 
            unsigned int &numIonsHot, unsigned int &numIonsCold,
            const DevMem<float>& randPool,
				int neededParticles,
            const unsigned int NX1, const unsigned int NY1,
            const unsigned int NIJ,
            const float SIGMA_HE, const float SIGMA_HI,
            const float SIGMA_CE, const float SIGMA_CI,
				const unsigned int injectWidth,
				const unsigned int injectStartX,
				DevStream &stream);

#endif
