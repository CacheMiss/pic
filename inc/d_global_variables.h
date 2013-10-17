#ifndef D_GLOBAL_VARIABLES
#define D_GLOBAL_VARIABLES

// SPECIAL NOTES: Due to the use of __fdividef, the following size
// restraints apply to certian variables:
// 1. The result of D_X_GRD * D_DX and D_Y_GRD * D_DY must not be >= 2^126
// 2. D_NM_PRTS < 2^126

// define system dimensions
//#define D_NM_PRTS 5638400     // max number of particles ions or electrons 
//#define D_NM_PRTS 3638400     // max number of particles ions or electrons 
#define D_X_GRD 514           // number of x grid points + 1
#define D_Y_GRD 4098          // number of y grid points + 1

// define system constants
#define D_PI (3.14159265358979f)
#define D_TPI (6.283185307179586476925287f)
#define D_ISEED 300085189 // random number seed

// The electric potential at the top of the grid
#define D_P0 (-10.0f) 
#define D_SCALE (10.0f)
#define D_RATO  (1.0f/400.0f)
//#define D_DELT  (D_TPI/64.0f)  // She simulation step size
#define D_DELT  (0.05f)  // She simulation step size
#define D_BXM (0.0f)
#define D_BYM (3.0f)
#define D_BZM (0.0f)
#define D_SIGMA_CE (10.0f)        // sigma for cold electrons
#define D_SIGMA_CI (10.0f)       // sigma for cold ions
#define D_SIGMA_HI (0.3f)        // sigma for hot ions
#define D_SIGMA_HE (1.0f)        // sigma for hot electrons

#define D_TSTART 0.0f
//#define D_TMAX 100.0
//#define D_TMAX 100000.0
#define D_LF 20                // number of iterations before 
                               // incrementing the log number
//#define D_LFINT 5            // number of info intervals between output files
#define D_DX (1.0f)
#define D_DX2 (1.0/2.)
#define D_DY (1.0f)
#define D_TOTA D_DX * D_DY

// derived parameters
#define D_NX (D_X_GRD-1)
#define D_NX1 (D_NX - 1)
#define D_NX12 (D_NX1 / 2)

#define D_NY (D_Y_GRD-1)
#define D_NY1 (D_NY - 1)
#define D_NY12 (D_NY1 / 2)

// avg number of particles per cell?
#define D_NIJ 18 // avg number of particle per cell?

#define FPP 5 // Floats Per Particle

// CUDA Globals
#define D_MAX_THREADS_PER_BLOCK 512

//#define NO_THRUST

#ifndef NO_THRUST
//#define USE_THRUST_SORT
#endif

#endif
