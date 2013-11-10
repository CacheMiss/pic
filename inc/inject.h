#ifndef INJECT_H
#define INJECT_H

__global__
void inject(float2 eleHotLoc[], float3 eleHotVel[], 
            float2 eleColdLoc[], float3 eleColdVel[],
            float2 ionHotLoc[], float3 ionHotVel[], 
            float2 ionColdLoc[], float3 ionColdVel[],
            const int botXStart, const int injectWidth,
            const float DX, const float DY,
            const int numElectronsHot, const int numElectronsCold, 
            const int numIonsHot, const int numIonsCold,
            float randPool[], const int randPoolSize,
            const unsigned int NX1, const unsigned int NY1,
            const unsigned int NIJ,
            const float SIGMA_HE, const float SIGMA_HI,
            const float SIGMA_CE, const float SIGMA_CI);

#endif
