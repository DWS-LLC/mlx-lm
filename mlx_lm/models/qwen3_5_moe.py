# Copyright © 2026 Apple Inc.

import re
from dataclasses import dataclass

import mlx.core as mx

from .base import BaseModelArgs
from .qwen3_5 import Model as Qwen3_5Model


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(Qwen3_5Model):

    def sanitize(self, weights):
        is_raw_checkpoint = any(not k.startswith("language_model.") for k in weights)
        new_weights = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            new_weights[key] = value

        # Fuse MoE expert weights (gate_up_proj -> gate/up, down_proj -> down)
        # for both the backbone and any MTP head layers.
        for key in list(new_weights):
            if key.endswith(".experts.gate_up_proj"):
                prefix = key[: -len(".experts.gate_up_proj")]
                gate_up = new_weights.pop(key)
                mid = gate_up.shape[-2] // 2
                new_weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[
                    ..., :mid, :
                ]
                new_weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[
                    ..., mid:, :
                ]
                new_weights[f"{prefix}.switch_mlp.down_proj.weight"] = new_weights.pop(
                    f"{prefix}.experts.down_proj"
                )

        # Per-expert MoE layout (Qwen3.5): experts.<i>.{gate,up,down}_proj.weight.
        # Stack each projection across experts into switch_mlp.<proj>.weight.
        per_expert = {}
        for key in list(new_weights):
            m = re.fullmatch(
                r"(.+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", key
            )
            if m:
                prefix, expert, proj = m.group(1), int(m.group(2)), m.group(3)
                per_expert.setdefault((prefix, proj), {})[expert] = new_weights.pop(key)
        for (prefix, proj), experts in per_expert.items():
            new_weights[f"{prefix}.switch_mlp.{proj}.weight"] = mx.stack(
                [experts[i] for i in sorted(experts)], axis=0
            )

        return self.language_model.sanitize(
            new_weights, is_raw_checkpoint=is_raw_checkpoint
        )
