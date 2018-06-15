export SHELL := /bin/bash
export NICE := nice -n 19
export PYTHON := python2 -u

export CUDA_VISIBLE_DEVICES := -1 #no GPU
# export CUDA_VISIBLE_DEVICES := GPU-715edb49

all: \
	validation_images_evaluated \

UCF101_limited.dat: UCF101.dat
	$(NICE) $(PYTHON) UCF101_limited.py

UCF101.dat: network_trained
	CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} $(NICE) $(PYTHON) CNN_extract_features_testset.py

validation_images_evaluated: testset_evaluated
	set -o pipefail; \
	CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} $(NICE) $(PYTHON) CNN_validate_images.py 2>&1 | tee $@_fail && mv $@_fail $@

testset_evaluated: network_trained
	set -o pipefail; \
	CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} $(NICE) $(PYTHON) CNN_evaluate_testset.py 2>&1 | tee $@_fail && mv $@_fail $@

network_trained: data/files_extracted
	set -o pipefail; \
	CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} $(NICE) $(PYTHON) CNN_train_UCF101.py 2>&1 | tee $@_fail && mv $@_fail $@

data/files_extracted: data/files_moved
	set -o pipefail; \
	cd $(dir $@) && $(NICE) $(PYTHON) 2_extract_files.py 2>&1 | tee $(notdir $@)

data/files_moved: data/UCF101.rar
	set -o pipefail; \
	cd $(dir $@) && $(NICE) $(PYTHON) 1_move_files.py 2>&1 | tee $(notdir $@)

data/UCF101.rar:
	cd $(dir $@) && \
	mkdir -p train test sequences checkpoints && \
	wget -N http://crcv.ucf.edu/data/UCF101/UCF101.rar && \
	unrar e UCF101.rar; \
