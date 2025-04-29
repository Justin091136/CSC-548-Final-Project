UNAME := $(shell uname)

CXX = g++
MPICXX = mpic++
CXXFLAGS = -std=c++11 -O3
LDFLAGS =

ifeq ($(UNAME), Darwin)
    # macOS
    CXX = /opt/homebrew/opt/llvm/bin/clang++
    export OMPI_CXX = /opt/homebrew/opt/llvm/bin/clang++
    CXXFLAGS += -fopenmp -I/opt/homebrew/opt/llvm/include -L/opt/homebrew/opt/llvm/lib
    LDFLAGS += -lomp
else
    # Linux
    CXXFLAGS += -fopenmp
endif

BIN_DIR = ../bin
SRC_DIR = .

TARGETS = \
    $(BIN_DIR)/kmeans \
    $(BIN_DIR)/kmeans-openmp \
    $(BIN_DIR)/kmeans-mpi \
    $(BIN_DIR)/kmeans-hybrid

all: $(BIN_DIR) $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/kmeans: $(SRC_DIR)/kmeans.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/kmeans-openmp: $(SRC_DIR)/kmeans-openmp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/kmeans-mpi: $(SRC_DIR)/kmeans-mpi.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/kmeans-hybrid: $(SRC_DIR)/kmeans-hybrid.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

.PHONY: all clean

clean:
	rm -f $(BIN_DIR)/kmeans*
