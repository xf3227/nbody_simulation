CC = gcc
CFLAGS = -Wall -Werror -O1 -mavx2 -mfma -fopenmp

NVCC = /usr/local/cuda-11.7/bin/nvcc
NVCCFLAGS = -arch=sm_86 -O1 -Xcompiler -fPIC

LDFLAGS = -Xcompiler -fopenmp

BIN_DIR = bin
SRC_DIR = src
OBJ_DIR = build
LIB_DIR = lib

SRCS_C = $(wildcard $(SRC_DIR)/*.c)
OBJS_C = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS_C))

SRCS_CU = $(wildcard $(SRC_DIR)/*.cu)
OBJS_CU = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS_CU))

.PHONY: all clean

all: $(BIN_DIR)/nbody $(BIN_DIR)/test_performance $(LIB_DIR)/nbody.so

$(BIN_DIR)/test_performance: $(OBJS_C) $(OBJS_CU)
	$(NVCC) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/nbody: $(OBJS_C) $(OBJS_CU)
	$(NVCC) $^ -o $@ $(LDFLAGS)

$(LIB_DIR)/nbody.so: $(OBJS_C) $(OBJS_CU)
	$(NVCC) $^ -o $@ $(LDFLAGS) -shared

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ_DIR)/*.o $(LIB_DIR)/*.so
