SRC_DIR = ./src/
OBJ_DIR = $(SRC_DIR)/object

all: install

install:
	+$(MAKE) -C $(OBJ_DIR)

clean:
	+$(MAKE) clean -C $(OBJ_DIR)


