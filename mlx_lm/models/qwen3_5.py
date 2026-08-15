# Copyright © 2026 Apple Inc.

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_map

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
)
from .cache import KVCache, TrimmableArraysCache
from .gated_delta import gated_delta_update, gated_delta_update_per_t
from .pipeline import PipelineMixin
from .qwen3_next import Qwen3NextAttention as Attention
from .qwen3_next import Qwen3NextMLP as MLP
from .qwen3_next import Qwen3NextRMSNormGated as RMSNormGated
from .qwen3_next import Qwen3NextSparseMoeBlock as SparseMoeBlock


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = ""
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    full_attention_interval: int = 4
    # Number of layers in the native Multi-Token Prediction head (0 = none).
    mtp_num_hidden_layers: int = 0

    # MoE fields (optional, for Qwen3_5MoeForConditionalGeneration)
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True

    # Rope parameters
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        }
    )

    # Derived from rope_parameters (set in __post_init__)
    partial_rotary_factor: float = 0.25
    rope_theta: float = 100000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.rope_parameters:
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")

            self.partial_rotary_factor = self.rope_parameters.get(
                "partial_rotary_factor", 0.25
            )
            self.rope_theta = self.rope_parameters.get("rope_theta", 100000.0)
            self.rope_scaling = self.rope_parameters


class GatedDeltaNet(nn.Module):
    def __init__(self, config: TextModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.sharding_group = None

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        capture = getattr(cache, "capture_states", False)
        if capture and S > 1 and mask is None:
            out, state, state_per_t = gated_delta_update_per_t(
                q,
                k,
                v,
                a,
                b,
                self.A_log,
                self.dt_bias,
                state,
                mask=None,
                use_kernel=not self.training,
            )
        else:
            out, state = gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                self.A_log,
                self.dt_bias,
                state,
                mask,
                use_kernel=not self.training,
            )

        if cache is not None:
            cache[1] = state
            if capture and S > 1 and mask is None:
                cache.store_history(
                    mx.contiguous(conv_input), state_per_t, self.conv_kernel_size - 1
                )
            elif capture:
                cache.append_history(state, cache[0])
            cache.advance(S)

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if args.num_experts > 0:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class MTPDecoderLayer(nn.Module):
    """A full-attention-only decoder layer for the MTP head (no GatedDeltaNet)."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        if args.num_experts > 0:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class MTPModule(nn.Module):
    """Native Multi-Token Prediction head (Qwen3.5/3.6/3.8 speculative decoding).

    Predicts token t+2 from the backbone's pre-final-norm hidden state at t and
    the sampled token t+1, reusing the backbone's shared lm_head. Only built when
    the checkpoint actually contains ``mtp.*`` weights (see ``TextModel.sanitize``).
    """

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.layers = [MTPDecoderLayer(args) for _ in range(args.mtp_num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        embed_tokens: nn.Embedding,
        cache: Optional[Any] = None,
        spec_step_idx: int = 0,
        inputs_embeds: Optional[mx.array] = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = embed_tokens(next_token_ids)
        e = self.pre_fc_norm_embedding(inputs_embeds)
        h = self.pre_fc_norm_hidden(hidden_states)
        fused = self.fc(mx.concatenate([e, h], axis=-1))

        if cache is None:
            cache = [None] * len(self.layers)
        # MTP layers are step-specific heads: select one per speculative step
        # and advance only its cache.
        layer_idx = spec_step_idx % len(self.layers)
        mask = create_attention_mask(fused, cache[layer_idx])
        fused = self.layers[layer_idx](fused, mask, cache[layer_idx])
        # Return (normed, pre-norm): the pre-norm hidden feeds the next
        # iterative draft step.
        return self.norm(fused), fused


class Qwen3_5TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def pipeline(self, group):
        super().pipeline(group)
        self.ssm_idx = None
        self.fa_idx = None
        for e, l in enumerate(self.pipeline_layers):
            if self.ssm_idx is None and l.is_linear:
                self.ssm_idx = e
            elif self.fa_idx is None and not l.is_linear:
                self.fa_idx = e
            if self.ssm_idx is not None and self.fa_idx is not None:
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        fa_mask = None
        ssm_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        if self.ssm_idx is not None:
            ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            hidden_states = mx.distributed.recv_like(hidden_states, (pipeline_rank + 1))

        for layer, c in zip(self.pipeline_layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            hidden_states = mx.distributed.send(
                hidden_states, (pipeline_rank - 1) % pipeline_size
            )
            if cache[-1] is not None:
                if hasattr(cache[-1], "keys"):
                    cache[-1].keys = mx.depends(cache[-1].keys, hidden_states)
                else:
                    cache[-1][0] = mx.depends(cache[-1][0], hidden_states)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            hidden_states = mx.distributed.all_gather(hidden_states)[
                : hidden_states.shape[0]
            ]

        return hidden_states


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3_5TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        # Pipeline loading discovers local shard files from the model parameter
        # tree before it downloads weights. Config-declared MTP layers must be
        # present at that point or their mtp.* shards are omitted.
        if args.mtp_num_hidden_layers > 0:
            self.mtp = MTPModule(args)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        return_hidden: bool = False,
    ) -> mx.array:
        hidden = self.model(inputs, cache, input_embeddings=input_embeddings)
        normed = self.model.norm(hidden)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(normed)
        else:
            out = self.lm_head(normed)
        if return_hidden:
            return out, hidden
        return out

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        return [
            TrimmableArraysCache(size=2) if l.is_linear else KVCache()
            for l in self.layers
        ]

    def make_mtp_cache(self):
        """Return a fresh list of KVCache entries for the MTP layer(s)."""
        if hasattr(self, "mtp"):
            return [KVCache() for _ in self.mtp.layers]
        return []

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        spec_step_idx=0,
        inputs_embeds=None,
    ):
        """Run the MTP head and apply the shared lm_head.

        Args:
            hidden_states: backbone pre-final-norm hidden state (B, N, H).
            next_token_ids: next token ids, shape (B, N). Ignored when
                ``inputs_embeds`` is provided.
            mtp_cache: KVCache entries for the MTP transformer layer(s).
            spec_step_idx: the speculative-step index selecting which of the
                step-specific MTP layers to run.
            inputs_embeds: pre-computed token embeddings (B, N, H), used for
                embedding-prefill generation instead of ``next_token_ids``.

        Returns:
            (logits, fused) where logits is (B, N, vocab_size) and fused is
            the MTP layer's pre-norm hidden (B, N, H) to feed the next step.
        """
        normed, fused = self.mtp(
            hidden_states,
            next_token_ids,
            self.model.embed_tokens,
            mtp_cache,
            spec_step_idx=spec_step_idx,
            inputs_embeds=inputs_embeds,
        )
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed), fused
        return self.lm_head(normed), fused

    def sanitize(self, weights, is_raw_backbone=False, is_raw_mtp=False):
        has_mtp_weights = any("mtp." in k for k in weights)
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
        )
        # Norms need a +1 shift only in raw HF checkpoints (detected before key
        # rewriting by ``Model.sanitize``). Track the backbone and MTP head
        # provenance separately: a converted backbone combined with raw
        # standalone mtp.* weights must shift only the MTP norms, never
        # double-shift the already-shifted backbone norms.
        should_shift_backbone = is_raw_backbone and (
            has_mtp_weights or has_unsanitized_conv1d
        )
        should_shift_mtp = is_raw_mtp and has_mtp_weights

        # Config-declared MTP layers are constructed in __init__ so pipeline
        # shard discovery includes their parameters before weights download.
        # Retain that head for an empty lazy-load weight map; after an actual
        # non-empty checkpoint lacks mtp.* tensors, remove it and load a plain
        # trunk instead. When the config omits the count, infer it from the
        # mtp.layers.<i> weight keys.
        if has_mtp_weights:
            if self.args.mtp_num_hidden_layers <= 0:
                indices = set()
                for k in weights:
                    m = re.search(r"mtp\.layers\.(\d+)\.", k)
                    if m:
                        indices.add(int(m.group(1)))
                if indices:
                    self.args.mtp_num_hidden_layers = max(indices) + 1
            if self.args.mtp_num_hidden_layers > 0 and not hasattr(self, "mtp"):
                self.mtp = MTPModule(self.args)
            elif self.args.mtp_num_hidden_layers <= 0:
                weights = {k: v for k, v in weights.items() if "mtp." not in k}
        else:
            weights = {k: v for k, v in weights.items() if "mtp." not in k}
            if weights and hasattr(self, "mtp"):
                delattr(self, "mtp")

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        )
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if v.ndim == 1 and any(k.endswith(sfx) for sfx in norm_keys):
                if "mtp." in k:
                    if should_shift_mtp:
                        weights[k] = v + 1.0
                elif should_shift_backbone:
                    weights[k] = v + 1.0
        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # Native Qwen MTP keeps the fusion projection in full precision:
            # its draft logits are highly sensitive to the normalized hidden /
            # next-token embedding combination.
            if path.endswith("mtp.fc"):
                return False
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith("A_log"):
                return False
            return True

        return predicate


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        return_hidden: bool = False,
    ):
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            return_hidden=return_hidden,
        )

    @property
    def model(self):
        return self.language_model.model

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        spec_step_idx=0,
        inputs_embeds=None,
    ):
        return self.language_model.mtp_forward(
            hidden_states,
            next_token_ids,
            mtp_cache,
            spec_step_idx=spec_step_idx,
            inputs_embeds=inputs_embeds,
        )

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    def sanitize(self, weights):
        def _is_vision(k):
            return k.startswith("vision_tower") or k.startswith("model.visual")

        # Track the backbone and MTP provenance separately so a converted
        # language_model.* backbone combined with raw standalone mtp.* weights
        # only shifts the MTP norms. Every supported raw backbone layout enters
        # as model.* (including text-only model.layers.*); vision keys are not
        # language weights and must not flip a converted checkpoint to "raw".
        is_raw_backbone = any(
            k.startswith("model.") and not _is_vision(k) for k in weights
        )
        is_raw_mtp = any(k.startswith("mtp.") for k in weights)
        sanitized = {}
        for key, value in weights.items():
            if _is_vision(key):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            sanitized[key] = value
        return self.language_model.sanitize(
            sanitized, is_raw_backbone=is_raw_backbone, is_raw_mtp=is_raw_mtp
        )

    def shard(self, group=None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()

        # A sharding factory for the convolution in gated delta net
        def conv_sharding(key_dim):
            return lambda p, w: (0, [key_dim, 2 * key_dim])

        def repeat_kv_layer_inplace(layer, h):
            # No repeat needed cause we have more heads than nodes
            if N <= h:
                return

            # Repeat function to apply to the layer weights
            def _repeat(p):
                s = p.shape
                p = p.reshape(h, s[0] // h, *s[1:])
                p = mx.repeat(p, N // h, axis=0)
                p = p.reshape(-1, *s[1:])
                return p

            layer.update(tree_map(_repeat, layer.parameters()))

        def shard_decoder_layer_inplace(layer):
            # MTPDecoderLayer is full-attention-only; backbone DecoderLayer
            # carries is_linear for its GatedDeltaNet layers.
            if getattr(layer, "is_linear", False):
                kd = layer.linear_attn.key_dim
                layer.linear_attn.sharding_group = group
                shard_inplace(layer.linear_attn.conv1d, conv_sharding(kd), group=group)
                layer.linear_attn.conv1d.groups //= N
                shard_inplace(
                    layer.linear_attn.in_proj_qkv,
                    "all-to-sharded",
                    segments=[kd, 2 * kd],
                    group=group,
                )
                shard_inplace(
                    layer.linear_attn.in_proj_z, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_b, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_a, "all-to-sharded", group=group
                )
                layer.linear_attn.dt_bias = mx.contiguous(
                    mx.split(layer.linear_attn.dt_bias, N)[rank]
                )
                layer.linear_attn.A_log = mx.contiguous(
                    mx.split(layer.linear_attn.A_log, N)[rank]
                )
                shard_inplace(layer.linear_attn.out_proj, "sharded-to-all", group=group)
                layer.linear_attn.num_k_heads //= N
                layer.linear_attn.num_v_heads //= N
                layer.linear_attn.key_dim //= N
                layer.linear_attn.value_dim //= N
                layer.linear_attn.conv_dim //= N
            else:
                layer.self_attn.o_proj = shard_linear(
                    layer.self_attn.o_proj, "sharded-to-all", group=group
                )
                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.k_proj, layer.self_attn.num_key_value_heads
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.v_proj, layer.self_attn.num_key_value_heads
                )
                layer.self_attn.k_proj = shard_linear(
                    layer.self_attn.k_proj, "all-to-sharded", group=group
                )
                layer.self_attn.v_proj = shard_linear(
                    layer.self_attn.v_proj, "all-to-sharded", group=group
                )
                layer.self_attn.num_attention_heads //= N
                layer.self_attn.num_key_value_heads = max(
                    1, layer.self_attn.num_key_value_heads // N
                )

            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.shared_expert.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )

        for layer in self.layers:
            shard_decoder_layer_inplace(layer)
        if hasattr(self.language_model, "mtp"):
            for layer in self.language_model.mtp.layers:
                shard_decoder_layer_inplace(layer)

    @property
    def layers(self):
        return self.language_model.model.pipeline_layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
