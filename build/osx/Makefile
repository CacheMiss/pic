###############################################################################
#
# Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
###############################################################################
#
# CUda UTility library build script
#
###############################################################################

###############################################################################
# SOURCE VARS
THIRDPARTY := ../../third_party
BOOST := $(THIRDPARTY)/boost_1_43_0
BOOSTLIB := $(BOOST)/lib/osx32
THRUST := $(THIRDPARTY)/thrust_1.3.0
RADIX := $(THIRDPARTY)/LsbRadixSort_1.0.215
RADIX_INCLUDE := -I$(RADIX) \
                 -I$(RADIX)/KernelCommon \
                 -I$(RADIX)/LsbRadixSort \
                 -I$(RADIX)/LsbRadixSort/kernel

EXECUTABLE := pic
CCFILES := genmtrand.cpp \
           logging_thread.cpp \
           logging_types.cpp \
           pic_utils.cpp \
           precisiontimer.cpp \
           global_variables.cpp \
           simulation_state.cpp \
           device_stats.cpp \
           commandline_options.cpp
CUFILES_sm_11 := dens.cu \
           field.cu \
           inject.cu \
           mersenne_twister.cu \
           movep.cu \
           pic2d_turbo.cu \
           potent2.cu

SRCDIR := ../../src/
INCLUDES += -I ../../inc -I$(THRUST) -I$(BOOST) $(RADIX_INCLUDE)
USERLIB := -L$(BOOSTLIB) \
  $(BOOSTLIB)/libboost_date_time.a $(BOOSTLIB)/libboost_thread.a \
  $(BOOSTLIB)/libboost_filesystem.a $(BOOSTLIB)/libboost_system.a \
  $(BOOSTLIB)/libboost_program_options.a

ROOTDIR := /Developer/GPU\ Computing/shared
BINDIR := .

USECUFFT := 1
OMIT_CUTIL_LIB := 1

include common.mk

