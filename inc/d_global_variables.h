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
#ifndef D_GLOBAL_VARIABLES
#define D_GLOBAL_VARIABLES

// SPECIAL NOTES: Due to the use of __fdividef, the following size
// restraints apply to certian variables:
// 1. The result of D_X_GRD * D_DX and D_Y_GRD * D_DY must not be >= 2^126
// 2. D_NM_PRTS < 2^126

// define system constants
#define D_PI (3.14159265358979f)
#define D_TPI (6.283185307179586476925287f)
#define D_ISEED 300085189 // random number seed

// The electric potential at the top of the grid
#define D_SCALE (10.0f)
#define D_RATO  (1.0f/400.0f)
#define D_DELT  (0.05f)  // She simulation step size
//#define D_BXM (0.0f)
//#define D_BYM (3.0f)
#define D_BZM (0.0f)

#define D_TSTART 0.0f
#define D_LF 20                // number of iterations before 
                               // incrementing the log number
#define D_LOG_IDX_WIDTH 6
#define D_DX (1.0f)
#define D_DX2 (1.0/2.)
#define D_DY (1.0f)
#define D_TOTA D_DX * D_DY

// avg number of particles per cell?
#define D_NIJ 18 // avg number of particle per cell?

// CUDA Globals
#define D_MAX_THREADS_PER_BLOCK 512

//#define NO_THRUST

#ifndef NO_THRUST
//#define USE_THRUST_SORT
#endif

#endif
