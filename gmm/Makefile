UNAME := $(shell uname)

CXX = g++
MPICXX = mpic++
CXXFLAGS = -std=c++11 -O2
LDFLAGS =

ifeq ($(UNAME), Darwin)
    # macOS
    CXX = /opt/homebrew/opt/llvm/bin/clang++
    CXXFLAGS += -fopenmp -I/opt/homebrew/opt/llvm/include -L/opt/homebrew/opt/llvm/lib -lomp
else
    CXXFLAGS += -fopenmp
endif

BIN_DIR = ../bin
SRC_DIR = .

TARGETS =  $(BIN_DIR)/gmm \
    $(BIN_DIR)/gmm-openmp \
    $(BIN_DIR)/gmm-mpi \
    $(BIN_DIR)/gmm-hybrid

all: $(BIN_DIR) $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/gmm: $(SRC_DIR)/gmm.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/gmm-openmp: $(SRC_DIR)/gmm-openmp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/gmm-mpi: $(SRC_DIR)/gmm-mpi.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/gmm-hybrid: $(SRC_DIR)/gmm-hybrid.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

.PHONY: all clean

clean:
	rm -f $(BIN_DIR)/gmm*
