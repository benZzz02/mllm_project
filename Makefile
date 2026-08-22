PYTHON ?= python3
LLAMAFACTORY ?= llamafactory-cli
METADATA_DIR ?= /path/to/DGM4/metadata
IMAGE_ROOT ?= /path/to/datasets
DATA_DIR ?= data/generated

.PHONY: data test sft infer-base-pool infer-sft-pool mine-base-prefs mine-sft-prefs base-dpo dpo ppo grpo simpo

data:
	$(PYTHON) scripts/convert_dgm4_to_sharegpt.py --metadata-dir $(METADATA_DIR) --image-root $(IMAGE_ROOT) --output-dir $(DATA_DIR)

test:
	$(PYTHON) -m unittest discover -s tests -v

sft:
	$(LLAMAFACTORY) train configs/sft_lora.yaml

infer-base-pool:
	$(PYTHON) scripts/infer_qwen3vl_with_scores.py --dataset $(DATA_DIR)/dgm4_preference_pool.jsonl --output predictions/base_preference_pool.jsonl

infer-sft-pool:
	$(PYTHON) scripts/infer_qwen3vl_with_scores.py --dataset $(DATA_DIR)/dgm4_preference_pool.jsonl --adapter outputs/sft_lora --output predictions/sft_preference_pool.jsonl

mine-base-prefs:
	$(PYTHON) scripts/build_preference_pairs.py --pool $(DATA_DIR)/dgm4_preference_pool.jsonl --predictions predictions/base_preference_pool.jsonl --output $(DATA_DIR)/dgm4_base_badcase_preference.jsonl

mine-sft-prefs:
	$(PYTHON) scripts/build_preference_pairs.py --pool $(DATA_DIR)/dgm4_preference_pool.jsonl --predictions predictions/sft_preference_pool.jsonl --output $(DATA_DIR)/dgm4_badcase_preference.jsonl

base-dpo:
	$(LLAMAFACTORY) train configs/base_dpo_lora.yaml

dpo:
	bash scripts/run_dpo.sh

simpo:
	$(LLAMAFACTORY) train configs/simpo_lora.yaml

ppo:
	bash scripts/run_ppo.sh

grpo:
	bash scripts/run_grpo.sh
