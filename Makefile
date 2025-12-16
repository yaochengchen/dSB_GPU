# ========================
# Compilers
# ========================
CXX   := g++
NVCC  := nvcc

# ========================
# Flags
# ========================
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -pedantic -DUSE_CDSB=1
NVCCFLAGS := -O2 -std=c++17 \
             --use_fast_math \
             --expt-relaxed-constexpr

# 如果你需要指定架构（推荐）
NVCCFLAGS += -gencode arch=compute_90,code=sm_90
NVCCFLAGS += -gencode arch=compute_90,code=compute_90
# 或者（较通用）
# NVCCFLAGS += -arch=sm_90

# ========================
# Paths
# ========================
EIGEN_INC ?= /usr/include/eigen3
CUDA_HOME ?= /usr/local/cuda

INCLUDES := \
  -I. \
  -Isrc \
  -I$(EIGEN_INC) \
  -I$(CUDA_HOME)/include

LDFLAGS := -L$(CUDA_HOME)/lib64 -lcudart

# ========================
# Sources
# ========================
CPP_SRCS := \
  cdsb_fasthare_qplib.cpp \
  fasthare_api.cpp \
  src/fasthare.cpp \
  src/graph.cpp

CU_SRCS := \
  cdsb_fastshare.cu

OBJS := \
  $(CPP_SRCS:.cpp=.o) \
  $(CU_SRCS:.cu=.o)

BIN := app

# ========================
# Rules
# ========================
.PHONY: all clean

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(BIN)
