#ifndef ARRAY2D_H
#define ARRAY2D_H

#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <vector>

#include "dev_mem.h"

template<class Type, class StorageType=std::vector<Type> >
class Array2d
{
   private:
   StorageType array;
   unsigned int numRows;
   unsigned int numColumns;

   public:
   Array2d()
     :array()
   {}
   Array2d(int rows, int columns)
     :numRows(rows), numColumns(columns)
   {
      array.resize(numRows * numColumns);
   }

   Array2d(const Array2d<Type, StorageType> &rhs)
   {
      Array2d(rhs, rhs.numRows, rhs.numColumns);
   }

   Array2d(const Array2d<Type, StorageType> &rhs, 
           unsigned int newRows, unsigned int newColumns)
   {
      assert(!rhs.array.empty());
      numRows = newRows;
      numColumns = newColumns;
      array.resize(numRows * numColumns);
      memcpy(&array[0], &rhs.array[0], numRows * numColumns * sizeof(Type));
   }

   ~Array2d()
   {
   }

   void resize(const int rows, const int columns)
   {
      assert(rows != 0);
      assert(columns != 0);
      //printf("DEBUG: Resizing to %d rows and %d columns\n", rows, columns);

      if(array != NULL)
      {
         delete [] array;
      }
      numRows = rows;
      numColumns = columns;
      array.resize(numRows * numColumns);
   }

   int size() const
   {
      return numRows * numColumns;
   }

   StorageType & getStorageContainer()
   {
      return array;
   }

   const Type* getData() const
   {
      return &array[0];
   }

   int getRows() const
   {
      return numRows;
   }

   int getColumns() const
   {
      return numColumns;
   }

   inline Type& operator()(const unsigned int row, 
                           const unsigned int column)
   {
      assert(numRows > 0);
      assert(numColumns > 0);
      assert(row < numRows);
      assert(column < numColumns);
      assert(row*numColumns+column < numRows*numColumns);

      return array[row * numColumns + column];
   }

   inline const Type& operator()(const unsigned int row, 
                                 const unsigned int column) const
   {
      assert(numRows > 0);
      assert(numColumns > 0);
      assert(row < numRows);
      assert(column < numColumns);

      return array[row * numColumns + column];
   }
   void copy (Type rhs[], const int rows, const int columns)
   {
      if(array != NULL)
      {
         array = new Type[rows*columns];
      }

      numRows = rows;
      numColumns = columns;
      memcpy(array, rhs, rows*columns*sizeof(Type));
   }

   Array2d<Type>& operator=(const Array2d<Type, StorageType> &rhs)
   {
      if(&rhs == this)
      {
         return *this;
      }
      numRows = rhs.numRows;
      numColumns = rhs.numColumns;
      array = rhs.array;
      return *this;
   }

   Array2d<Type>& operator=(const StorageType &rhs)
   {
      assert(rhs.size() == array.size());
      array = rhs;
      return *this;
   }

   Array2d<Type>& operator=(const DevMem<Type> &rhs)
   {
      cudaError_t error;
      assert(rhs.size() == array.size());
      error = cudaMemcpy(&array[0], rhs.getPtr(), rhs.size() * sizeof(Type),
         cudaMemcpyDeviceToHost);

      if(error != cudaSuccess)
      {
         fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
         assert(error == cudaSuccess);
         exit(1);
      } 
      return *this;
   }

   template<class Alloc>
   Array2d<Type>& operator=(const PitchedPtr<Type, Alloc> &rhs)
   {
      checkCuda(cudaMemcpy2D(&array[0], rhs.getWidthBytes(), 
                             reinterpret_cast<void*>(rhs.getPtr().ptr), rhs.getPitch(), 
                             rhs.getWidthBytes(), rhs.getY(), 
                             cudaMemcpyDeviceToHost));
      return *this;
   }

   void loadRows(const DevMem<Type> &rhs, const unsigned int numRows)
   {
      cudaError_t error;

      if(numRows == 0)
      {
         return;
      }
      // Check if DevMem is large enough
      assert(rhs.size() / this->numColumns >= numRows);
      // Check if Array2d is large enough
      assert(this->numRows >= numRows);
      error = cudaMemcpy(&array[0], rhs.getPtr(), 
         numRows * this->numColumns * sizeof(Type),
         cudaMemcpyDeviceToHost);

      if(error != cudaSuccess)
      {
         fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
         assert(error == cudaSuccess);
         exit(1);
      } 
   }
};

#endif
