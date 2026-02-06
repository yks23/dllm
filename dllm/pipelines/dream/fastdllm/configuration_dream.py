# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
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
"""DreamFastdLLM model configuration."""

from ..models.configuration_dream import DreamConfig


class DreamFastdLLMConfig(DreamConfig):
    """
    Thin wrapper over :class:`~dllm.pipelines.dream.models.configuration_dream.DreamConfig` that only
    changes the `model_type` used for auto-registration.
    """

    model_type = "Dream_fastdllm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
