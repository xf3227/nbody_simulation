CC = gcc
CFLAGS = -std=c99 -Wall -Werror -O1 -mavx2 -mfma -fopenmp


LDFLAGS = -fopenmp -lm

BIN_DIR = bin
SRC_DIR = src
OBJ_DIR = build
LIB_DIR = lib

SRCS_C = $(wildcard $(SRC_DIR)/*.c)
OBJS_C = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS_C))


.PHONY: all clean

all: $(BIN_DIR)/nbody

$(BIN_DIR)/nbody: $(OBJS_C)
	$(CC) $^ -o $@ $(LDFLAGS)

#$(LIB_DIR)/nbody.so: $(OBJS_C)
#	$(CC) $^ -o $@ $(LDFLAGS) -shared

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@


clean:
	rm -f $(OBJ_DIR)/*.o $(LIB_DIR)/*.so
