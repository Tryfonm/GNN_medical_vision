.PHONY: train tensorboard

EPOCHS ?= 100
LR ?= 0.001
PATIENCE ?= 5
BATCH_SIZE ?= 64

all: help

train:
	. venv/bin/activate && \
	python3 src/prototype/conv_nn__cifar.py \
	--epochs=$(EPOCHS) \
	--lr=$(LR) \
	--batch_size=$(BATCH_SIZE) \
	--patience=$(PATIENCE)


tensorboard:
	. venv/bin/activate && tensorboard --logdir=runs/

nvtop:
	nvtop

clean:
	rm -rf models/* && rm -rf runs/* || true

help:
	@echo "Makefile Usage:"
	@echo "  make train            : Train the model (Default: EPOCHS=100, LR=0.001, PATIENCE=3)"
	@echo "  make tensorboard      : Start TensorBoard for visualization"
	@echo "  make nvtop            : Run nvtop for GPU monitoring"
	@echo "  make help             : Display Makefile usage information"
