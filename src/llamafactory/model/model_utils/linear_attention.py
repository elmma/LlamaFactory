# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import TYPE_CHECKING, Any

import torch

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def load_linear_attention_model(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
    init_kwargs: dict[str, Any],
) -> "PreTrainedModel":
    r"""Load pretrained model with linear attention layers replaced.

    Two-stage loading for distilled linear attention models:
    1. Load base model architecture
    2. Replace attention layers with GLA (Gated Linear Attention)
    3. Load distilled weights from checkpoint

    Args:
        config: Model configuration from base model
        model_args: Model arguments including linear attention settings
        init_kwargs: Initialization arguments for model loading

    Returns:
        Model with linear attention layers and loaded distilled weights
    """
    from transformers import AutoModelForImageTextToText

    # Import from VLM-Research project
    import sys

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.models.qwen3vl_linear import replace_attention_layers

    # 1. Load base model
    init_kwargs["config"] = config
    init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
    init_kwargs["torch_dtype"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(**init_kwargs)

    # 2. Get layer indices from checkpoint config or args
    checkpoint_path = model_args.linear_attention_checkpoint
    layer_indices = None
    attention_type = model_args.linear_attention_type

    # Try to read from checkpoint config.json
    if checkpoint_path:
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            import json

            with open(config_path, "r") as f:
                ckpt_config = json.load(f)
            layer_indices = ckpt_config.get("linear_attention_layers")
            attention_type = ckpt_config.get("linear_attention_type", attention_type)

    # Override with explicit args if provided
    if model_args.linear_attention_layers:
        layer_indices = [int(x) for x in model_args.linear_attention_layers.split(",")]

    # 3. Replace attention layers
    logger.info_rank0(f"Replacing attention layers: {layer_indices}, type: {attention_type}")
    replace_attention_layers(model, layer_indices=layer_indices, attention_type=attention_type)

    # 4. Load distilled weights if checkpoint provided
    if checkpoint_path:
        weights_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(weights_path):
            from safetensors.torch import load_file

            weights = load_file(weights_path)
        else:
            weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
            if os.path.exists(weights_path):
                weights = torch.load(weights_path, map_location="cpu")
            else:
                logger.warning_rank0(f"No weights found at {checkpoint_path}, using initialized weights")
                return model

        missing, unexpected = model.load_state_dict(weights, strict=False)
        if missing:
            logger.warning_rank0(f"Missing keys when loading weights: {len(missing)}")
        if unexpected:
            logger.warning_rank0(f"Unexpected keys when loading weights: {len(unexpected)}")

        logger.info_rank0(f"Loaded linear attention model from {checkpoint_path}")

    return model
