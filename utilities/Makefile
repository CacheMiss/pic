CXX=g++
TARGETS=testPartFile
OBJ=obj
SRC=src
INCLUDE=-I$(HOME)/boost/include
LIB=-L$(HOME)/boost/lib -Wl,-rpath,$(HOME)/boost/lib -lboost_filesystem -lboost_system
CC_FLAGS=$(INCLUDE)

.PHONY : all clean tidy

all : $(TARGETS)

testPartFile : $(OBJ)/testPartFile.o
	$(CXX) $(CC_FLAGS) $(LIB) -o $@ $^

$(OBJ)/%.o : $(SRC)/%.cpp | $(OBJ)
	$(CXX) $(CC_FLAGS) -c -o $@ $<

clean: tidy
	rm -f $(TARGETS)

tidy:
	rm -rf $(OBJ)

$(OBJ):
	mkdir $(OBJ)
