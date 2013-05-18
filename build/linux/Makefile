ROOT=../..
CUDADIR=/opt/asn/apps/cuda_5.0
#CUDADIR=/usr/local/cuda
SRC=$(ROOT)/src
INC=$(ROOT)/inc
BINDIR=$(ROOT)/bin
EXE=$(BINDIR)/pic
OBJDIR=.obj
OBJ= \
   $(OBJDIR)/asc-gpulib.o \
   $(OBJDIR)/commandline_options.o \
   $(OBJDIR)/device_stats.o \
   $(OBJDIR)/error_check.o \
   $(OBJDIR)/genmtrand.o \
   $(OBJDIR)/global_variables.o \
   $(OBJDIR)/load_sim_state.o \
   $(OBJDIR)/logging_thread.o \
   $(OBJDIR)/logging_types.o \
   $(OBJDIR)/particle_grid_bin.o \
   $(OBJDIR)/pic_utils.o \
   $(OBJDIR)/precisiontimer.o \
   $(OBJDIR)/simulation_state.o \
   $(OBJDIR)/dev_mem_reuse.o \
   $(OBJDIR)/cuda_allocator.o \
   $(OBJDIR)/particle_allocator.o
CUOBJ= \
   $(OBJDIR)/dens.o \
   $(OBJDIR)/field.o \
   $(OBJDIR)/inject.o \
   $(OBJDIR)/movep.o \
   $(OBJDIR)/pic2d_turbo.o \
   $(OBJDIR)/potent2.o \
   $(OBJDIR)/binned_memory_kernels.o

INCLUDE_PATHS=-I$(INC) -I$(CUDADIR)/include -I~/boost/include
LIB_PATHS=-L~/boost/lib
THIRD_PARTY_LIBS=-lboost_thread -lboost_filesystem -lboost_program_options -lcufft -lcurand
PRE_PROCESSOR=-DASC -D_DEBUG
COMMON_FLAGS=-g $(PRE_PROCESSOR)
CUDA_FLAGS=-arch=sm_20 -m64 $(COMMON_FLAGS)
CC_FLAGS=-m64 $(COMMON_FLAGS)

all: $(EXE)

$(BINDIR):
	mkdir -p $(BINDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(EXE): $(OBJ) $(CUOBJ) | $(BINDIR)
	nvcc -o $(EXE) $(CUDA_FLAGS) $(OBJ) $(CUOBJ) $(LIB_PATHS) $(THIRD_PARTY_LIBS)

$(OBJDIR)/%.o: $(SRC)/%.cu | $(OBJDIR)
	nvcc -c -o $@ $< $(INCLUDE_PATHS) $(CUDA_FLAGS)

$(OBJDIR)/%.o: $(SRC)/../%.cpp | $(OBJDIR)
	g++ -c -o $@ $< $(CC_FLAGS) $(INCLUDE_PATHS)

$(OBJDIR)/%.o: $(SRC)/%.cpp | $(OBJDIR)
	g++ -c -o $@ $< $(CC_FLAGS) $(INCLUDE_PATHS)

tidy::
	rm -rf $(OBJDIR)

clean:: tidy
	rm -f $(EXE)
