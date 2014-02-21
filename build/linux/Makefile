ROOT=../..
CUDADIR=/opt/apps/cuda/5.0
#BOOSTDIR=/opt/apps/intel13/boost/1.51.0
BOOSTDIR=/home1/02498/sewells/boost
#CUDADIR=/usr/local/cuda
TBB=/home1/02498/sewells/tbb
SRC=$(ROOT)/src
INC=$(ROOT)/inc
BINDIR=$(ROOT)/bin
EXE=$(BINDIR)/pic
OBJDIR=.obj
OBJ= \
   $(OBJDIR)/commandline_options.o \
   $(OBJDIR)/cuda_allocator.o \
   $(OBJDIR)/device_stats.o \
   $(OBJDIR)/dev_mem_reuse.o \
   $(OBJDIR)/error_check.o \
   $(OBJDIR)/genmtrand.o \
   $(OBJDIR)/global_variables.o \
   $(OBJDIR)/logging_thread.o \
   $(OBJDIR)/logging_types.o \
   $(OBJDIR)/particle_allocator.o \
   $(OBJDIR)/phi_avg.o \
   $(OBJDIR)/pic2d_turbo.o \
   $(OBJDIR)/pic_utils.o \
   $(OBJDIR)/pitched_ptr.o \
   $(OBJDIR)/precisiontimer.o \
   $(OBJDIR)/simulation_state.o \

CUOBJ= \
   $(OBJDIR)/dens.o \
   $(OBJDIR)/field.o \
   $(OBJDIR)/inject.o \
   $(OBJDIR)/movep.o \
   $(OBJDIR)/potent2.o \

INCLUDE_PATHS=-I$(INC) -I$(CUDADIR)/include -I$(BOOSTDIR)/include -I$(TBB)/include
RPATH=-Wl,-rpath,$(CUDADIR)/lib64,-rpath,$(BOOSTDIR)/lib,-rpath,$(TBB)/lib
LIB_PATHS=-L$(CUDADIR)/lib64 -L$(BOOSTDIR)/lib -L$(TBB)/lib
THIRD_PARTY_LIBS=-lboost_thread -lboost_system -lboost_filesystem -lboost_program_options -lboost_chrono -lboost_timer -lcufft -lcurand -lcudart -ltbb
#DEBUG_FLAGS=-g -D_DEBUG
#CUDA_DEBUG_FLAGS=-G
OPTIMIZATION=-O2 -DNDEBUG
PRE_PROCESSOR=
COMMON_FLAGS=$(PRE_PROCESSOR) $(DEBUG_FLAGS) $(OPTIMIZATION)
SUPPORTED_VIRTUAL_ARCH=-arch=compute_30 -code=compute_30,sm_30
CUDA_FLAGS=$(SUPPORTED_VIRTUAL_ARCH) -m64 $(COMMON_FLAGS)
CC_FLAGS=-m64 $(COMMON_FLAGS)

all: $(EXE)

$(BINDIR):
	mkdir -p $(BINDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(EXE): $(OBJ) $(CUOBJ) | $(BINDIR)
	g++ -o $(EXE) $(OBJ) $(CUOBJ) $(LIB_PATHS) $(THIRD_PARTY_LIBS) $(RPATH)

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
