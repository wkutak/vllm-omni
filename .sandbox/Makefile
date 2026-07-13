IMAGE_NAME := vllm-omni-nim
IMAGE_NAME_CUDA129 := $(IMAGE_NAME)-cuda129
IMAGE_VERSION := $(shell git rev-parse --short HEAD)
PROJECT_ROOT := $(shell dirname "$(PWD)")

# GPU selection. Forwarded to docker --gpus. Defaults to `all` to preserve
# existing behavior. Set to a comma-separated list of GPU indices to restrict:
#   GPUS=0       make brr   # single GPU 0
#   GPUS=0,1     make brr   # GPUs 0 and 1
#   GPUS=all     make brr   # explicit (default)
GPUS ?= all

ifeq ($(GPUS),all)
GPU_FLAG := --gpus all
else
GPU_FLAG := --gpus '"device=$(GPUS)"'
endif


SCRATCH_SPACE ?= /home/scratch.wkutak_other/
DATA_SPACE ?= /home/scratch.wkutak_other_1/
MOUNTS ?= -v $(SCRATCH_SPACE):$(SCRATCH_SPACE) -v $(DATA_SPACE):$(DATA_SPACE)

# Host port published for the serving container.
HOST_PORT ?= 8000

MODEL_DIR ?= $(DATA_SPACE)/dev/cosmos3/quantization/data
NANO_BF16_MODEL ?= $(MODEL_DIR)/cosmos3-nano/bf16
NANO_FP8_MODEL ?= $(MODEL_DIR)/cosmos3-nano/fp8

SUPER_BF16_MODEL ?= $(MODEL_DIR)/cosmos3-super/bf16
SUPER_FP8_MODEL ?= $(MODEL_DIR)/cosmos3-super/fp8

SUPER_T2I_BF16_MODEL ?= $(MODEL_DIR)/super-t2i/bf16
SUPER_T2I_FP8_MODEL ?= $(MODEL_DIR)/super-t2i/fp8

SUPER_I2V_BF16_MODEL ?= $(MODEL_DIR)/super-i2v/bf16
SUPER_I2V_FP8_MODEL ?= $(MODEL_DIR)/super-i2v/fp8

VLLM_OMNI_DIR ?= $(SCRATCH_SPACE)/dev/vllm-omni-cosmos-genai-nim/

build: 
	DOCKER_BUILDKIT=1 docker build \
		-t $(IMAGE_NAME):$(IMAGE_VERSION) \
		-f ./docker/Dockerfile.cuda.local \
		$(PROJECT_ROOT)
	@echo Image $(IMAGE_NAME) built.

build-cuda129: 
	DOCKER_BUILDKIT=1 docker build \
		-t $(IMAGE_NAME_CUDA129):$(IMAGE_VERSION) \
		-f ./docker/Dockerfile.cuda129.local \
		$(PROJECT_ROOT)
	@echo Image $(IMAGE_NAME) built.


upload:
	docker tag $(IMAGE_NAME):$(IMAGE_VERSION) nvcr.io/nvidian/vllm_omni-cosmos3:v0.25.0_git_vllm_$(IMAGE_VERSION) && \
	docker push nvcr.io/nvidian/vllm_omni-cosmos3:v0.25.0_git_vllm_$(IMAGE_VERSION)

upload-cuda129:
	docker tag $(IMAGE_NAME_CUDA129):$(IMAGE_VERSION) nvcr.io/nvidian/vllm_omni-cosmos3:v0.25.0_cu129_git_vllm_$(IMAGE_VERSION) && \
	docker push nvcr.io/nvidian/vllm_omni-cosmos3:v0.25.0_cu129_git_vllm_$(IMAGE_VERSION)

run:
	docker run -it --rm \
		--name $(IMAGE_NAME)-serve \
		$(GPU_FLAG) \
		--ipc host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-p $(HOST_PORT):8000 \
		-e HF_TOKEN \
		-e HUGGING_FACE_HUB_TOKEN="$(HF_TOKEN)" \
		-u 0:0 \
		-e NGC_API_KEY=$(NGC_API_KEY) \
		$(MOUNTS) \
		$(IMAGE_NAME):$(IMAGE_VERSION) \
		bash

infer-nano:
	docker run -it --rm \
		--name $(IMAGE_NAME)-serve \
		$(GPU_FLAG) \
		--ipc host \
		-p $(HOST_PORT):8000 \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-e HF_TOKEN \
		-e HUGGING_FACE_HUB_TOKEN="$(HF_TOKEN)" \
		-u 0:0 \
		-e NGC_API_KEY=$(NGC_API_KEY) \
		$(MOUNTS) \
		$(IMAGE_NAME):$(IMAGE_VERSION) \
		bash -c 'cd $(VLLM_OMNI_DIR) && \
		python .sandbox/overlay/cosmos3_infer.py \
			--model $(NANO_FP8_MODEL) \
			--height 720 --width 1280 --num-frames 189 --num-inference-steps 50 \
			--no-system-prompt --negative-prompt "Subjects are poorly placed within the frame, appearing at awkward positions that violate basic compositional rules." \
			--guidance-scale 6.0 --flow-shift 10 \
			--output .sandbox/overlay/infer/super_fp8_$(shell date +%Y%m%d_%H%M%S).mp4'

infer-super:
	docker run -it --rm \
		--name $(IMAGE_NAME)-serve \
		$(GPU_FLAG) \
		--ipc host \
		-p $(HOST_PORT):8000 \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-e HF_TOKEN \
		-e HUGGING_FACE_HUB_TOKEN="$(HF_TOKEN)" \
		-u 0:0 \
		-e NGC_API_KEY=$(NGC_API_KEY) \
		$(MOUNTS) \
		$(IMAGE_NAME):$(IMAGE_VERSION) \
		bash -c 'cd $(VLLM_OMNI_DIR) && \
		python .sandbox/overlay/cosmos3_infer.py \
			--model $(SUPER_FP8_MODEL) \
			--height 720 --width 1280 --num-frames 189 --num-inference-steps 50 \
			--no-system-prompt --negative-prompt "Subjects are poorly placed within the frame, appearing at awkward positions that violate basic compositional rules." \
			--guidance-scale 6.0 --flow-shift 10 \
			--output .sandbox/overlay/infer/super_fp8_$(shell date +%Y%m%d_%H%M%S).mp4'

infer-i2v:
	docker run -it --rm \
		--name $(IMAGE_NAME)-serve \
		$(GPU_FLAG) \
		--ipc host \
		-p $(HOST_PORT):8000 \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-e HF_TOKEN \
		-e HUGGING_FACE_HUB_TOKEN="$(HF_TOKEN)" \
		-u 0:0 \
		-e NGC_API_KEY=$(NGC_API_KEY) \
		$(MOUNTS) \
		$(IMAGE_NAME):$(IMAGE_VERSION) \
		bash -c 'cd $(VLLM_OMNI_DIR) && \
		python .sandbox/overlay/cosmos3_infer_i2v.py \
			--model $(SUPER_I2V_FP8_MODEL) \
			--output .sandbox/overlay/infer/super_i2v_fp8_$(shell date +%Y%m%d_%H%M%S).mp4'

infer-t2i:
	docker run -it --rm \
		--name $(IMAGE_NAME)-serve \
		$(GPU_FLAG) \
		--ipc host \
		-p $(HOST_PORT):8000 \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-e HF_TOKEN \
		-e HUGGING_FACE_HUB_TOKEN="$(HF_TOKEN)" \
		-u 0:0 \
		-e NGC_API_KEY=$(NGC_API_KEY) \
		$(MOUNTS) \
		$(IMAGE_NAME):$(IMAGE_VERSION) \
		bash -c 'cd $(VLLM_OMNI_DIR) && \
		python .sandbox/overlay/cosmos3_infer_t2i.py \
			--model $(SUPER_T2I_FP8_MODEL) \
			--prompt "A photorealistic red fox sitting in a snowy forest at dawn." \
			--output .sandbox/overlay/infer/super_t2i_fp8_$(shell date +%Y%m%d_%H%M%S).png'
