#ifndef PIC_UTILS_H
#define PIC_UTILS_H

#include <assert.h>
#include <driver_functions.h>
#include <string>

#include "array2d.h"
#include "global_variables.h"

//void checkForCudaError();
void checkForCudaError(const char *errorMsg);
void checkForCudaErrorSync(const char *errorMsg);
void checkForCudaError(const char *errorMsg, cudaError_t error);
void errExit(const char *errorString);
void outinfo(const std::string &fname,int idx_nm,float time,int need,int niid);
void createOutputDir(const char *);
bool fileExists(const std::string &fileName);
void resizeDim3(dim3 &rhs, int x, int y=1, int z=1);
void loadPrevSimState(unsigned int loadIndex, const std::string &loadDir,
                      DevMem<float2> &dev_eleHotLoc, DevMem<float3> &dev_eleHotVel, 
                      DevMem<float2> &dev_eleColdLoc, DevMem<float3> &dev_eleColdVel,
                      DevMem<float2> &dev_ionHotLoc, DevMem<float3> &dev_ionHotVel, 
                      DevMem<float2> &dev_ionColdLoc, DevMem<float3> &dev_ionColdVel,
                      unsigned int &numEleHot, unsigned int &numEleCold,
                      unsigned int &numIonHot, unsigned int &numIonCold);

inline unsigned int calcNumBlocks(const unsigned int numThreads, 
                           const unsigned int numElements)
{
   unsigned int ret = (numElements + numThreads - 1) / numThreads;
   assert(ret > 0);
   return ret;
}

//*****************************************************************************
// Name: out2dr
// Purpose: Output two dimensional scalar grid files 
// Parameters:
// ---------------------
// fname  - Filename to output
// idx_nm - Index to append to fname
// numRows    - Number of rows
// numColumns - Number of columns
// arry   - The array to print
//*****************************************************************************
template<class ArrayType>
void out2dr(const std::string &fname,int idx_nm,int numRows,int numColumns,
            const ArrayType &arry, bool printColumnOrder=false)
{
   char name[400];
   FILE *fp;
   int i,j;
   sprintf(name,"%s/%s_%04d", outputDir.c_str(), fname.c_str(), idx_nm);
   if((fp=fopen(name,"wt"))==NULL) {
      printf("Cannot open '%s' file for writing\n",name);
      exit(1);
   }
   if(!printColumnOrder)
   {
      for (i=0;i<numRows;i++) 
      {
         for (j=0;j<numColumns;j++) {
            if ((j)%6==0) fprintf(fp,"\n");
            fprintf(fp,"%12.3e ",arry(i, j));
         }
         fprintf(fp,"\n");
      }
   }
   else
   {
      for (j=0;j<numColumns;j++) 
      {
         for (i=0;i<numRows;i++) {
            if ((i)%6==0) fprintf(fp,"\n");
            fprintf(fp,"%12.3e ",arry(i, j));
         }
         fprintf(fp,"\n");
      }
   }
   fprintf(fp,"\n");
   fclose(fp);
}

//*****************************************************************************
// Name: out2drBin
// Purpose: Output two dimensional scalar grid files in a binary format
// Parameters:
// ---------------------
// fname  - Filename to output
// idx_nm - Index to append to fname
// numRows    - Number of rows
// numColumns - Number of columns
// arry   - The array to print
//*****************************************************************************
template<class ArrayType>
void out2drBin(const std::string &fname,int idx_nm,int numRows,int numColumns,
            const ArrayType &arry, bool printColumnOrder=false)
{
   char name[400];
   FILE *fp;
   int i,j;
   sprintf(name,"%s/%s_%04d", outputDir.c_str(), fname.c_str(), idx_nm);
   if((fp=fopen(name,"wb"))==NULL) {
      printf("Cannot open '%s' file for writing\n",name);
      exit(1);
   }
   fwrite(&numRows, sizeof(numRows), 1, fp);
   fwrite(&numColumns, sizeof(numColumns), 1, fp);
   int columnOrder = (printColumnOrder == false) ? 0:1;
   fwrite(&columnOrder, sizeof(columnOrder), 1, fp);
   if(!printColumnOrder)
   {
      for (i=0;i<numRows;i++) 
      {
         for (j=0;j<numColumns;j++) 
         {
            fwrite(&arry(i,j), sizeof(float), 1, fp);
         }
      }
   }
   else
   {
      for (j=0;j<numColumns;j++) 
      {
         for (i=0;i<numRows;i++) 
         {
            fwrite(&arry(i,j), sizeof(float), 1, fp);         
         }
      }
   }
   fclose(fp);
}

//******************************************************************************
// Name: outprt
// Purpose: Output particle (ion or ele) data 
//          arry[Particle Number][element number] where
//          element number = 0 -- x position
//                         = 1 -- y position
//                         = 2 -- x velocity
//                         = 3 -- y velocity
//                         = 4 -- z velocity
// Parameters:
// ----------------------
// fname   - The filename to write to
// idx_nm  - The numer to append to the filename
// hot     - The array of hot particles
// cold    - The array of cold particles
// numHot  - The number of hot particles
// numCold - The number of cold particles
//******************************************************************************
template<class ArrayType>
void outprt(const std::string &fname, int idx_nm, const ArrayType &hot,
            const ArrayType &cold, int numHot, int numCold)
{
   char name[100];
   FILE *fp;
   int i,j;
   sprintf(name,"%s/%s_%04d", outputDir.c_str(), fname.c_str(), idx_nm);
   if((fp=fopen(name,"wt"))==NULL) {
      printf("Cannot open '%s' file for writing\n",name);
      exit(1);
   }
   for (i=0;i<numHot;i++) {
      for (j=0;j<5;j++) {
         fprintf(fp,"%12.3e ",hot(i, j));
      }
      fprintf(fp,"%12.3e ", 1.0);
      fprintf(fp,"\n");
   }
   for (i=0;i<numCold;i++) {
      for (j=0;j<5;j++) {
         fprintf(fp,"%12.3e ",cold(i, j));
      }
      fprintf(fp,"%12.3e ", 0.0);
      fprintf(fp,"\n");
   }
   fprintf(fp,"\n");
   fclose(fp);
}

//******************************************************************************
// Name: outprtBin
// Purpose: Output in binary particle (ion or ele) data 
//          arry[Particle Number][element number] where
//          element number = 0 -- x position
//                         = 1 -- y position
//                         = 2 -- x velocity
//                         = 3 -- y velocity
//                         = 4 -- z velocity
// Parameters:
// ----------------------
// fname   - The filename to write to
// idx_nm  - The numer to append to the filename
// hot     - The array of hot particles
// cold    - The array of cold particles
// numHot  - The number of hot particles
// numCold - The number of cold particles
//******************************************************************************
template<class ArrayType>
void outprtBin(const std::string &fname, int idx_nm, const ArrayType &hot,
            const ArrayType &cold, int numHot, int numCold)
{
   char name[100];
   FILE *fp;
   int i,j;
   float one = 1;
   float zero = 0;
   sprintf(name,"%s/%s_%04d", outputDir.c_str(), fname.c_str(), idx_nm);
   if((fp=fopen(name,"wb"))==NULL) {
      printf("Cannot open '%s' file for writing\n",name);
      exit(1);
   }
   int totalPart = numHot + numCold;
   fwrite(&totalPart, sizeof(totalPart), 1, fp);
   for (i=0;i<numHot;i++) {
      for (j=0;j<5;j++) {
         fwrite(&hot(i,j), sizeof(float), 1, fp);
      }
      fwrite(&one, sizeof(one), 1, fp);
   }
   for (i=0;i<numCold;i++) {
      for (j=0;j<5;j++) {
         fwrite(&cold(i,j), sizeof(float), 1, fp);
      }
      fwrite(&zero, sizeof(zero), 1, fp);
   }
   fclose(fp);
}

#endif