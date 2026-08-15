# Copyright © 2026 Apple Inc.

import contextlib
import unittest
from unittest.mock import patch

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm import utils
from mlx_lm.generate import mtp_generate_step, stream_generate
from mlx_lm.models.cache import KVCache, TrimmableArraysCache, make_prompt_cache
from mlx_lm.models.qwen3_5 import Model, ModelArgs, MTPModule
from mlx_lm.tokenizer_utils import TokenizerWrapper


def _text_config(mtp_num_hidden_layers=1):
    return {
        "model_type": "qwen3_5",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32,
        "linear_num_value_heads": 1,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2,
        "tie_word_embeddings": False,
        "max_position_embeddings": 64,
        "mtp_num_hidden_layers": mtp_num_hidden_layers,
    }


def _make_model(mtp_num_hidden_layers=1):
    args = ModelArgs.from_dict(
        {
            "model_type": "qwen3_5",
            "text_config": _text_config(mtp_num_hidden_layers),
        }
    )
    return Model(args)


def _mtp_weights():
    return {
        "mtp.fc.weight": mx.zeros((8, 16)),
        "mtp.pre_fc_norm_hidden.weight": mx.zeros((8,)),
        "mtp.pre_fc_norm_embedding.weight": mx.zeros((8,)),
        "mtp.norm.weight": mx.zeros((8,)),
        "mtp.layers.0.input_layernorm.weight": mx.zeros((8,)),
        "mtp.layers.0.post_attention_layernorm.weight": mx.zeros((8,)),
        "mtp.layers.0.self_attn.q_proj.weight": mx.zeros((8, 8)),
        "mtp.layers.0.self_attn.k_proj.weight": mx.zeros((8, 8)),
        "mtp.layers.0.self_attn.v_proj.weight": mx.zeros((8, 8)),
        "mtp.layers.0.self_attn.o_proj.weight": mx.zeros((8, 8)),
        "mtp.layers.0.self_attn.q_norm.weight": mx.zeros((8,)),
        "mtp.layers.0.self_attn.k_norm.weight": mx.zeros((8,)),
        "mtp.layers.0.mlp.gate_proj.weight": mx.zeros((16, 8)),
        "mtp.layers.0.mlp.up_proj.weight": mx.zeros((16, 8)),
        "mtp.layers.0.mlp.down_proj.weight": mx.zeros((8, 16)),
    }


class TestMTP(unittest.TestCase):

    def test_sanitize_attaches_mtp_only_with_weights(self):
        model = _make_model(mtp_num_hidden_layers=1)
        model.sanitize({"language_model.model.embed_tokens.weight": mx.zeros((32, 8))})
        self.assertFalse(hasattr(model.language_model, "mtp"))

        model = _make_model(mtp_num_hidden_layers=1)
        model.sanitize(_mtp_weights())
        self.assertTrue(hasattr(model.language_model, "mtp"))

    def test_config_declared_mtp_parameters_survive_empty_sanitize(self):
        model = _make_model(mtp_num_hidden_layers=2)
        # pipeline_load performs this config-only sanitize before collecting
        # tree_flatten(model.parameters()) to choose safetensor shards.
        model.sanitize({})

        self.assertTrue(hasattr(model.language_model, "mtp"))
        keys = {key for key, _ in tree_flatten(model.parameters())}
        self.assertTrue(any(key.startswith("language_model.mtp.") for key in keys))
        self.assertTrue(
            any(key.startswith("language_model.mtp.layers.1.") for key in keys)
        )

    def test_pipeline_shards_include_configless_mtp_weights(self):
        # The first pipeline model load sees config only. A zero MTP count
        # therefore contributes no mtp.* parameter keys to tree_flatten.
        model = _make_model(mtp_num_hidden_layers=0)
        model.sanitize({})
        self.assertFalse(hasattr(model.language_model, "mtp"))

        weight_index = {
            key: "backbone.safetensors" for key, _ in tree_flatten(model.parameters())
        }
        weight_index["language_model.mtp.layers.0.self_attn.q_proj.weight"] = (
            "mtp-only.safetensors"
        )

        local_files = utils._pipeline_local_files(model, weight_index)
        self.assertIn("backbone.safetensors", local_files)
        self.assertIn("mtp-only.safetensors", local_files)

    def test_pipeline_shards_allow_absent_configured_mtp_weights(self):
        model = _make_model(mtp_num_hidden_layers=1)
        model.sanitize({})
        self.assertTrue(hasattr(model.language_model, "mtp"))

        # The converted index has every backbone tensor but no MTP tensor.
        # Pipeline selection must defer removal to the second load rather than
        # treating the config-only MTP parameters as an invalid checkpoint.
        weight_index = {
            key: "backbone.safetensors"
            for key, _ in tree_flatten(model.parameters())
            if not key.startswith("language_model.mtp.")
        }
        local_files = utils._pipeline_local_files(model, weight_index)
        self.assertEqual(local_files, {"backbone.safetensors"})

        with self.assertRaises(ValueError):
            utils._pipeline_local_files(model, {})

    def test_pipeline_shards_reject_partial_configured_mtp_weights(self):
        model = _make_model(mtp_num_hidden_layers=1)
        model.sanitize({})
        weight_index = {
            key: "backbone.safetensors"
            for key, _ in tree_flatten(model.parameters())
            if not key.startswith("language_model.mtp.")
        }
        # One MTP tensor means this is a declared but incomplete head, not the
        # valid backbone-only fallback.
        weight_index["language_model.mtp.fc.weight"] = "mtp.safetensors"
        with self.assertRaises(ValueError):
            utils._pipeline_local_files(model, weight_index)

    def test_quant_predicate_preserves_mtp_fusion_projection(self):
        dense_predicate = _make_model(mtp_num_hidden_layers=1).quant_predicate
        self.assertFalse(dense_predicate("language_model.mtp.fc", None))
        self.assertTrue(
            dense_predicate("language_model.model.layers.0.mlp.up_proj", None)
        )

        from mlx_lm.models import qwen3_5_moe

        text_config = _text_config(mtp_num_hidden_layers=1)
        text_config.update(
            {
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 16,
                "shared_expert_intermediate_size": 8,
            }
        )
        args = qwen3_5_moe.ModelArgs.from_dict(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {"model_type": "qwen3_5_moe", **text_config},
            }
        )
        moe_predicate = qwen3_5_moe.Model(args).quant_predicate
        self.assertFalse(moe_predicate("language_model.mtp.fc", None))

    def test_sanitize_no_double_shift_on_converted(self):
        base = mx.arange(8, dtype=mx.float32)
        hf_norm_key = "model.language_model.layers.0.input_layernorm.weight"
        mlx_norm_key = "language_model.model.layers.0.input_layernorm.weight"

        model = _make_model(mtp_num_hidden_layers=1)
        raw = model.sanitize(
            {
                hf_norm_key: base,
                "mtp.pre_fc_norm_hidden.weight": base,
                "mtp.pre_fc_norm_embedding.weight": base,
                "mtp.norm.weight": base,
            }
        )
        self.assertTrue(mx.array_equal(raw[mlx_norm_key], base + 1.0))
        self.assertTrue(
            mx.array_equal(
                raw["language_model.mtp.pre_fc_norm_hidden.weight"], base + 1.0
            )
        )

        model2 = _make_model(mtp_num_hidden_layers=1)
        loaded = model2.sanitize(raw)
        self.assertTrue(mx.array_equal(loaded[mlx_norm_key], base + 1.0))
        self.assertTrue(
            mx.array_equal(
                loaded["language_model.mtp.pre_fc_norm_hidden.weight"], base + 1.0
            )
        )

    def test_sanitize_mixed_checkpoint_shifts_only_raw_mtp_norms(self):
        base = mx.arange(8, dtype=mx.float32)
        converted_backbone_key = "language_model.model.layers.0.input_layernorm.weight"
        model = _make_model(mtp_num_hidden_layers=1)
        sanitized = model.sanitize(
            {
                # Converted MLX weights are already shifted.
                converted_backbone_key: base + 1.0,
                # Standalone MTP weights retain raw HF RMSNorm convention.
                "mtp.pre_fc_norm_hidden.weight": base,
            }
        )

        self.assertTrue(mx.array_equal(sanitized[converted_backbone_key], base + 1.0))
        self.assertTrue(
            mx.array_equal(
                sanitized["language_model.mtp.pre_fc_norm_hidden.weight"], base + 1.0
            )
        )

    def test_make_mtp_cache_and_extract(self):
        model = _make_model(mtp_num_hidden_layers=1)
        model.language_model.mtp = MTPModule(model.language_model.args)
        cache = model.make_mtp_cache()
        self.assertEqual(len(cache), 1)
        self.assertTrue(all(isinstance(c, KVCache) for c in cache))

        arr = TrimmableArraysCache(size=2)
        arr[0] = mx.zeros((4, 3, 8))
        arr[1] = mx.zeros((4, 2, 4, 4))
        self.assertIsInstance(arr.extract(0), TrimmableArraysCache)

    def test_mtp_generate_step_matches_greedy(self):
        prev_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            model = _make_model(mtp_num_hidden_layers=1)
            model.language_model.mtp = MTPModule(model.language_model.args)
            model.eval()
            mx.eval(model.parameters())

            prompt = mx.array([1, 2, 3])
            max_tokens = 8

            ref = []
            y = prompt
            c = make_prompt_cache(model)
            while y.size > 1:
                model(y[:-1][None], cache=c)
                y = y[-1:]
            cur = y.item()
            for _ in range(max_tokens):
                logits = model(mx.array([[cur]]), cache=c)
                mx.eval(logits)
                cur = mx.argmax(logits[0, -1], axis=-1).item()
                ref.append(cur)

            prompt_cache = make_prompt_cache(model)
            got = [
                tok
                for tok, _lp, _from_draft in mtp_generate_step(
                    prompt, model, prompt_cache=prompt_cache, max_tokens=max_tokens
                )
            ]
            self.assertEqual(got, ref)

            for entry in prompt_cache:
                if isinstance(entry, TrimmableArraysCache):
                    self.assertFalse(entry.capture_states)
        finally:
            mx.set_default_device(prev_device)

    def test_mtp_multi_layer_reconciliation_is_recursive(self):
        prev_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            model = _make_model(mtp_num_hidden_layers=2)
            model.language_model.mtp = MTPModule(model.language_model.args)
            model.eval()
            mx.eval(model.parameters())

            calls = []
            mtp_forward = model.mtp_forward

            def record_mtp_forward(hidden_states, next_token_ids, mtp_cache, **kwargs):
                logits, fused = mtp_forward(
                    hidden_states, next_token_ids, mtp_cache, **kwargs
                )
                calls.append((kwargs.get("spec_step_idx", 0), hidden_states, fused))
                return logits, fused

            # Force target and MTP argmaxes to agree. The first cycle accepts a
            # draft and therefore exercises reconciliation before generation
            # reaches the final direct-backbone token.
            force_token = mx.array([[1e6] + [0.0] * 31])
            with patch.object(model, "mtp_forward", side_effect=record_mtp_forward):
                generated = list(
                    mtp_generate_step(
                        mx.array([1, 2, 3]),
                        model,
                        max_tokens=3,
                        num_draft_tokens=1,
                        logits_processors=[
                            lambda _tokens, logits: logits + force_token
                        ],
                    )
                )

            self.assertEqual([token for token, _lp, _draft in generated], [0, 0, 0])
            self.assertTrue(any(from_draft for _token, _lp, from_draft in generated))

            # The only one-token MTP calls are the reconciliation chain. Its
            # layer-1 input must be layer 0's fused output, not a fresh
            # backbone hidden-state slice.
            reconcile_calls = [call for call in calls if call[1].shape[1] == 1]
            self.assertEqual(
                [step for step, _hidden, _fused in reconcile_calls], [0, 1]
            )
            _, _layer0_input, layer0_fused = reconcile_calls[0]
            _, layer1_input, _layer1_fused = reconcile_calls[1]
            mx.eval(layer0_fused, layer1_input)
            self.assertTrue(mx.allclose(layer1_input, layer0_fused))
        finally:
            mx.set_default_device(prev_device)

    def test_mtp_cache_prefill(self):
        model = _make_model(mtp_num_hidden_layers=1)
        model.language_model.mtp = MTPModule(model.language_model.args)
        model.eval()
        mx.eval(model.parameters())

        mtp_cache = model.make_mtp_cache()
        hidden = mx.random.normal((1, 3, 8))
        tokens = mx.array([[2, 3, 4]])
        _, _ = model.mtp_forward(hidden, tokens, mtp_cache)
        mx.eval([c.state for c in mtp_cache])

        # A prefilled MTP cache must hold one KV entry per prompt position.
        self.assertEqual(mtp_cache[0].offset, 3)

    def test_moe_expert_stacking_for_mtp(self):
        from mlx_lm.models import qwen3_5_moe

        text_config = _text_config(mtp_num_hidden_layers=1)
        text_config["num_experts"] = 2
        text_config["num_experts_per_tok"] = 1
        text_config["moe_intermediate_size"] = 16
        text_config["shared_expert_intermediate_size"] = 8
        args = qwen3_5_moe.ModelArgs.from_dict(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {"model_type": "qwen3_5_moe", **text_config},
            }
        )
        model = qwen3_5_moe.Model(args)

        sanitized = model.sanitize(
            {
                "mtp.layers.0.mlp.experts.gate_up_proj": mx.zeros((2, 32, 8)),
                "mtp.layers.0.mlp.experts.down_proj": mx.zeros((2, 8, 16)),
            }
        )
        # The MTP expert weights must be fused into switch_mlp, not left as
        # unmatched experts.* tensors.
        self.assertIn(
            "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight", sanitized
        )
        self.assertIn(
            "language_model.mtp.layers.0.mlp.switch_mlp.up_proj.weight", sanitized
        )
        self.assertIn(
            "language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight", sanitized
        )
        self.assertFalse(any("experts." in k for k in sanitized))

        # Per-expert layout (Qwen3.5): experts.<i>.{gate,up,down}_proj.weight
        # must be stacked across experts into switch_mlp.<proj>.weight.
        sanitized2 = model.sanitize(
            {
                "mtp.layers.0.mlp.experts.0.gate_proj.weight": mx.zeros((16, 8)),
                "mtp.layers.0.mlp.experts.0.up_proj.weight": mx.zeros((16, 8)),
                "mtp.layers.0.mlp.experts.0.down_proj.weight": mx.zeros((8, 16)),
                "mtp.layers.0.mlp.experts.1.gate_proj.weight": mx.zeros((16, 8)),
                "mtp.layers.0.mlp.experts.1.up_proj.weight": mx.zeros((16, 8)),
                "mtp.layers.0.mlp.experts.1.down_proj.weight": mx.zeros((8, 16)),
            }
        )
        self.assertEqual(
            sanitized2[
                "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight"
            ].shape,
            (2, 16, 8),
        )
        self.assertEqual(
            sanitized2[
                "language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight"
            ].shape,
            (2, 8, 16),
        )
        self.assertFalse(any("experts." in k for k in sanitized2))

    def test_raw_checkpoint_detection_ignores_vision(self):
        model = _make_model(mtp_num_hidden_layers=1)
        base = mx.arange(8, dtype=mx.float32)
        # A converted checkpoint: all language_model.* weights plus leftover
        # vision tensors. The vision keys must not mark it "raw" and force a
        # second +1 norm shift.
        sanitized = model.sanitize(
            {
                "language_model.model.layers.0.input_layernorm.weight": base,
                "language_model.mtp.pre_fc_norm_hidden.weight": base,
                "language_model.mtp.pre_fc_norm_embedding.weight": base,
                "language_model.mtp.norm.weight": base,
                "language_model.mtp.layers.0.input_layernorm.weight": base,
                "language_model.mtp.layers.0.post_attention_layernorm.weight": base,
                "vision_tower.encoder.weight": mx.zeros((8, 8)),
                "model.visual.proj.weight": mx.zeros((8, 8)),
            }
        )
        self.assertTrue(
            mx.array_equal(
                sanitized["language_model.model.layers.0.input_layernorm.weight"],
                base,
            )
        )

    def test_top_level_raw_dense_backbone_shifts_norms(self):
        base = mx.arange(8, dtype=mx.float32)
        model = _make_model(mtp_num_hidden_layers=0)
        sanitized = model.sanitize(
            {
                "model.layers.0.input_layernorm.weight": base,
                "model.norm.weight": base,
                "model.layers.0.linear_attn.conv1d.weight": mx.zeros((8, 4, 3)),
            }
        )

        self.assertTrue(
            mx.array_equal(
                sanitized["language_model.model.layers.0.input_layernorm.weight"],
                base + 1.0,
            )
        )
        self.assertTrue(
            mx.array_equal(sanitized["language_model.model.norm.weight"], base + 1.0)
        )

    def test_top_level_raw_moe_backbone_shifts_norms(self):
        from mlx_lm.models import qwen3_5_moe

        text_config = _text_config(mtp_num_hidden_layers=0)
        text_config.update(
            {
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 16,
                "shared_expert_intermediate_size": 8,
            }
        )
        args = qwen3_5_moe.ModelArgs.from_dict(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {"model_type": "qwen3_5_moe", **text_config},
            }
        )
        model = qwen3_5_moe.Model(args)
        base = mx.arange(8, dtype=mx.float32)
        sanitized = model.sanitize(
            {
                "model.layers.0.input_layernorm.weight": base,
                "model.norm.weight": base,
                "model.layers.0.linear_attn.conv1d.weight": mx.zeros((8, 4, 3)),
            }
        )

        self.assertTrue(
            mx.array_equal(
                sanitized["language_model.model.layers.0.input_layernorm.weight"],
                base + 1.0,
            )
        )
        self.assertTrue(
            mx.array_equal(sanitized["language_model.model.norm.weight"], base + 1.0)
        )

    def test_moe_expert_ids_must_be_complete(self):
        from mlx_lm.models import qwen3_5_moe

        text_config = _text_config(mtp_num_hidden_layers=1)
        text_config["num_experts"] = 2
        text_config["num_experts_per_tok"] = 1
        text_config["moe_intermediate_size"] = 16
        text_config["shared_expert_intermediate_size"] = 8
        args = qwen3_5_moe.ModelArgs.from_dict(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {"model_type": "qwen3_5_moe", **text_config},
            }
        )
        model = qwen3_5_moe.Model(args)

        # Experts 0 and 2 (gap at 1) must be rejected, not silently reindexed.
        with self.assertRaises(ValueError):
            model.sanitize(
                {
                    "mtp.layers.0.mlp.experts.0.gate_proj.weight": mx.zeros((16, 8)),
                    "mtp.layers.0.mlp.experts.0.up_proj.weight": mx.zeros((16, 8)),
                    "mtp.layers.0.mlp.experts.0.down_proj.weight": mx.zeros((8, 16)),
                    "mtp.layers.0.mlp.experts.2.gate_proj.weight": mx.zeros((16, 8)),
                    "mtp.layers.0.mlp.experts.2.up_proj.weight": mx.zeros((16, 8)),
                    "mtp.layers.0.mlp.experts.2.down_proj.weight": mx.zeros((8, 16)),
                }
            )

        # A projection missing an expert must also be rejected.
        with self.assertRaises(ValueError):
            model.sanitize(
                {
                    "mtp.layers.0.mlp.experts.0.gate_proj.weight": mx.zeros((16, 8)),
                    "mtp.layers.0.mlp.experts.0.up_proj.weight": mx.zeros((16, 8)),
                    "mtp.layers.0.mlp.experts.0.down_proj.weight": mx.zeros((8, 16)),
                    "mtp.layers.0.mlp.experts.1.gate_proj.weight": mx.zeros((16, 8)),
                    "mtp.layers.0.mlp.experts.1.up_proj.weight": mx.zeros((16, 8)),
                    "mtp.layers.0.mlp.experts.1.down_proj.weight": mx.zeros((8, 16)),
                    "mtp.layers.0.mlp.experts.2.down_proj.weight": mx.zeros((8, 16)),
                }
            )

    def test_mtp_generate_step_processor_history(self):
        prev_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            model = _make_model(mtp_num_hidden_layers=1)
            model.language_model.mtp = MTPModule(model.language_model.args)
            model.eval()
            mx.eval(model.parameters())

            seen = []

            def recorder(tokens, logits):
                # Repetition penalties call len(tokens); logit bias indexes
                # logits[:, indices]. Both require a real token history and a
                # batched [1, vocab] tensor at every position.
                self.assertIsNotNone(tokens)
                self.assertEqual(logits.ndim, 2)
                seen.append(len(tokens))
                return logits

            prompt = mx.array([1, 2, 3])
            list(
                mtp_generate_step(
                    prompt, model, max_tokens=8, logits_processors=[recorder]
                )
            )

            # Every processor call got a non-empty history, and the history
            # grows as tokens are generated (never a constant prompt-only stub).
            self.assertTrue(seen)
            self.assertTrue(all(n >= len(prompt) for n in seen))
            self.assertGreater(max(seen), len(prompt))
        finally:
            mx.set_default_device(prev_device)

    def test_mtp_processor_exception_restores_prompt_cache(self):
        prev_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            model = _make_model(mtp_num_hidden_layers=1)
            model.language_model.mtp = MTPModule(model.language_model.args)
            model.eval()
            mx.eval(model.parameters())

            prompt = mx.array([1, 2, 3])
            prompt_cache = make_prompt_cache(model)
            calls = 0

            def fail_during_verify(_tokens, logits):
                nonlocal calls
                calls += 1
                # bootstrap tok0, MTP seed d1, then the first verify logit
                if calls == 3:
                    raise RuntimeError("processor failure")
                return logits

            with self.assertRaisesRegex(RuntimeError, "processor failure"):
                list(
                    mtp_generate_step(
                        prompt,
                        model,
                        prompt_cache=prompt_cache,
                        max_tokens=3,
                        num_draft_tokens=1,
                        logits_processors=[fail_during_verify],
                    )
                )

            for entry in prompt_cache:
                if isinstance(entry, KVCache):
                    self.assertEqual(entry.offset, len(prompt))
        finally:
            mx.set_default_device(prev_device)

    def test_mtp_quantization_exception_restores_prompt_cache(self):
        prev_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            model = _make_model(mtp_num_hidden_layers=1)
            model.language_model.mtp = MTPModule(model.language_model.args)
            model.eval()
            mx.eval(model.parameters())

            prompt = mx.array([1, 2, 3])
            prompt_cache = make_prompt_cache(model)
            calls = 0

            def fail_during_verify_quantization(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                # Prefill and bootstrap quantization run first; the third call
                # follows the cache-mutating verification forward.
                if calls == 3:
                    raise RuntimeError("quantization failure")

            with patch(
                "mlx_lm.generate.maybe_quantize_kv_cache",
                side_effect=fail_during_verify_quantization,
            ):
                with self.assertRaisesRegex(RuntimeError, "quantization failure"):
                    list(
                        mtp_generate_step(
                            prompt,
                            model,
                            prompt_cache=prompt_cache,
                            max_tokens=3,
                            num_draft_tokens=1,
                        )
                    )

            for entry in prompt_cache:
                if isinstance(entry, KVCache):
                    self.assertEqual(entry.offset, len(prompt))
        finally:
            mx.set_default_device(prev_device)

    def test_mtp_final_token_quantization_failure_restores_prompt_cache(self):
        prev_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            model = _make_model(mtp_num_hidden_layers=1)
            model.eval()
            mx.eval(model.parameters())

            prompt = mx.array([1])
            prompt_cache = make_prompt_cache(model)
            calls = 0

            def fail_during_final_quantization(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                # Bootstrap quantization runs first; max_tokens=1 then takes
                # the final-token backbone write path.
                if calls == 2:
                    raise RuntimeError("final quantization failure")

            with patch(
                "mlx_lm.generate.maybe_quantize_kv_cache",
                side_effect=fail_during_final_quantization,
            ):
                with self.assertRaisesRegex(RuntimeError, "final quantization failure"):
                    list(
                        mtp_generate_step(
                            prompt, model, prompt_cache=prompt_cache, max_tokens=1
                        )
                    )

            for entry in prompt_cache:
                if isinstance(entry, KVCache):
                    self.assertEqual(entry.offset, len(prompt))
        finally:
            mx.set_default_device(prev_device)

    def test_mtp_rejects_non_positive_num_draft_tokens(self):
        model = _make_model(mtp_num_hidden_layers=1)
        model.language_model.mtp = MTPModule(model.language_model.args)
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                list(
                    mtp_generate_step(mx.array([1, 2, 3]), model, num_draft_tokens=bad)
                )

    def test_mtp_rejects_non_trimmable_prompt_cache(self):
        from mlx_lm.models.cache import ArraysCache

        model = _make_model(mtp_num_hidden_layers=1)
        model.language_model.mtp = MTPModule(model.language_model.args)
        # A plain ArraysCache is not rollback-capable; the MTP preflight must
        # reject it instead of silently leaving rejected drafts cached.
        prompt_cache = [ArraysCache(size=2), KVCache()]
        with self.assertRaises(ValueError):
            list(
                mtp_generate_step(mx.array([1, 2, 3]), model, prompt_cache=prompt_cache)
            )

    def test_mtp_rejects_populated_prompt_cache(self):
        prev_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            model = _make_model(mtp_num_hidden_layers=1)
            model.language_model.mtp = MTPModule(model.language_model.args)
            model.eval()
            mx.eval(model.parameters())

            prompt_cache = make_prompt_cache(model)
            model(mx.array([[1, 2, 3]]), cache=prompt_cache)
            mx.eval([entry.state for entry in prompt_cache])

            with self.assertRaisesRegex(ValueError, "populated prompt_cache"):
                list(mtp_generate_step(mx.array([4]), model, prompt_cache=prompt_cache))
        finally:
            mx.set_default_device(prev_device)

    def test_mtp_final_token_commits_to_prompt_cache(self):
        prev_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            model = _make_model(mtp_num_hidden_layers=1)
            model.language_model.mtp = MTPModule(model.language_model.args)
            model.eval()
            mx.eval(model.parameters())

            prompt = mx.array([1, 2, 3])

            # Full greedy reference over the whole sequence.
            ref = []
            y = prompt
            c = make_prompt_cache(model)
            while y.size > 1:
                model(y[:-1][None], cache=c)
                y = y[-1:]
            cur = y.item()
            for _ in range(3):
                logits = model(mx.array([[cur]]), cache=c)
                mx.eval(logits)
                cur = mx.argmax(logits[0, -1], axis=-1).item()
                ref.append(cur)

            # max_tokens=1 forces the num_draft <= 0 branch on the first cycle.
            prompt_cache = make_prompt_cache(model)
            got1 = [
                tok
                for tok, _lp, _fd in mtp_generate_step(
                    prompt, model, prompt_cache=prompt_cache, max_tokens=1
                )
            ]
            self.assertEqual(got1, ref[:1])

            # The emitted token must be committed to the cache: exact offset,
            # not merely <= prompt + output.
            for entry in prompt_cache:
                if isinstance(entry, KVCache):
                    self.assertEqual(entry.offset, len(prompt) + 1)

            # MTP rejects the populated cache because it cannot reconstruct its
            # corresponding head state. A direct backbone forward still proves
            # the emitted token was committed: the next reference token must
            # predict the one after it from the reused cache.
            logits = model(mx.array([[ref[1]]]), cache=prompt_cache)
            mx.eval(logits)
            self.assertEqual(mx.argmax(logits[0, -1], axis=-1).item(), ref[2])
        finally:
            mx.set_default_device(prev_device)

    def test_stream_mtp_finalizes_cache_before_terminal_response(self):
        class FakeTokenizer:
            eos_token_id = 0
            chat_template = None

            def get_vocab(self):
                return {}

            def encode(self, _text, add_special_tokens=False):
                return [1]

            def decode(self, tokens):
                return "".join(map(str, tokens))

        prev_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            model = _make_model(mtp_num_hidden_layers=1)
            model.language_model.mtp = MTPModule(model.language_model.args)
            model.eval()
            mx.eval(model.parameters())

            prompt = mx.array([1, 2, 3])
            prompt_cache = make_prompt_cache(model)
            eos_bias = mx.array([[1e6] + [0.0] * 31])

            with patch(
                "mlx_lm.generate.wired_limit", return_value=contextlib.nullcontext()
            ):
                responses = list(
                    stream_generate(
                        model,
                        TokenizerWrapper(FakeTokenizer()),
                        prompt,
                        prompt_cache=prompt_cache,
                        mtp=True,
                        max_tokens=4,
                        logits_processors=[lambda _tokens, logits: logits + eos_bias],
                    )
                )

            self.assertEqual(len(responses), 1)
            self.assertEqual(responses[0].finish_reason, "stop")
            for entry in prompt_cache:
                if isinstance(entry, KVCache):
                    self.assertEqual(entry.offset, len(prompt) + 1)
        finally:
            mx.set_default_device(prev_device)

    def test_mtp_layer_selection(self):
        model = _make_model(mtp_num_hidden_layers=2)
        model.language_model.mtp = MTPModule(model.language_model.args)
        model.eval()
        mx.eval(model.parameters())

        mtp_cache = model.make_mtp_cache()
        self.assertEqual(len(mtp_cache), 2)
        hidden = mx.random.normal((1, 1, 8))
        # spec_step_idx=1 selects layer 1 and must advance only its cache.
        _, _ = model.mtp_forward(hidden, mx.array([[1]]), mtp_cache, spec_step_idx=1)
        self.assertEqual(mtp_cache[0].offset, 0)
        self.assertEqual(mtp_cache[1].offset, 1)

    def test_shard_includes_dense_mtp_decoder_layer(self):
        class Group:
            def size(self):
                return 1

            def rank(self):
                return 0

        model = _make_model(mtp_num_hidden_layers=1)
        mtp_layer = model.language_model.mtp.layers[0]
        sharded_linear = []

        def record_linear(linear, *_args, **_kwargs):
            sharded_linear.append(linear)
            return linear

        with patch(
            "mlx_lm.models.qwen3_5.shard_linear", side_effect=record_linear
        ), patch("mlx_lm.models.qwen3_5.shard_inplace"):
            model.shard(Group())

        sharded_linear_ids = {id(linear) for linear in sharded_linear}
        for linear in (
            mtp_layer.self_attn.q_proj,
            mtp_layer.self_attn.k_proj,
            mtp_layer.self_attn.v_proj,
            mtp_layer.self_attn.o_proj,
            mtp_layer.mlp.gate_proj,
            mtp_layer.mlp.up_proj,
            mtp_layer.mlp.down_proj,
        ):
            self.assertIn(id(linear), sharded_linear_ids)

    def test_shard_includes_moe_mtp_decoder_layer(self):
        from mlx_lm.models import qwen3_5_moe

        class Group:
            def size(self):
                return 1

            def rank(self):
                return 0

        text_config = _text_config(mtp_num_hidden_layers=1)
        text_config.update(
            {
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 16,
                "shared_expert_intermediate_size": 8,
            }
        )
        args = qwen3_5_moe.ModelArgs.from_dict(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {"model_type": "qwen3_5_moe", **text_config},
            }
        )
        model = qwen3_5_moe.Model(args)
        mtp_layer = model.language_model.mtp.layers[0]
        sharded_inplace = []

        def record_inplace(module, *_args, **_kwargs):
            sharded_inplace.append(module)

        with patch(
            "mlx_lm.models.qwen3_5.shard_linear",
            side_effect=lambda linear, *_a, **_k: linear,
        ), patch("mlx_lm.models.qwen3_5.shard_inplace", side_effect=record_inplace):
            model.shard(Group())

        sharded_inplace_ids = {id(module) for module in sharded_inplace}
        for module in (
            mtp_layer.mlp.shared_expert.gate_proj,
            mtp_layer.mlp.shared_expert.up_proj,
            mtp_layer.mlp.shared_expert.down_proj,
            mtp_layer.mlp.switch_mlp.gate_proj,
            mtp_layer.mlp.switch_mlp.up_proj,
            mtp_layer.mlp.switch_mlp.down_proj,
        ):
            self.assertIn(id(module), sharded_inplace_ids)

    def test_mtp_count_inferred_when_omitted(self):
        model = _make_model(mtp_num_hidden_layers=0)
        model.sanitize(_mtp_weights())
        self.assertTrue(hasattr(model.language_model, "mtp"))
        self.assertEqual(len(model.language_model.mtp.layers), 1)

    def test_mtp_rejects_empty_prompt(self):
        model = _make_model(mtp_num_hidden_layers=1)
        model.language_model.mtp = MTPModule(model.language_model.args)
        with self.assertRaises(ValueError):
            list(mtp_generate_step(mx.array([]), model))

    def test_mtp_embedding_only_prompt_skips_initial_processor(self):
        prev_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            model = _make_model(mtp_num_hidden_layers=1)
            model.eval()
            mx.eval(model.parameters())

            prompt = mx.array([], mx.uint32)
            embeddings = model.model.embed_tokens(mx.array([1], mx.uint32))
            logits = model(prompt[None], input_embeddings=embeddings[None])
            mx.eval(logits)
            expected = mx.argmax(logits[0, -1], axis=-1).item()
            forced_token = (expected + 1) % model.language_model.args.vocab_size
            force_bias = mx.array(
                [[1e6 if i == forced_token else 0.0 for i in range(32)]]
            )
            histories = []

            def force_token(tokens, processor_logits):
                histories.append(len(tokens))
                return processor_logits + force_bias

            generator = mtp_generate_step(
                prompt,
                model,
                max_tokens=1,
                input_embeddings=embeddings,
                logits_processors=[force_token],
            )
            try:
                token, _logprobs, _from_draft = next(generator)
            finally:
                generator.close()

            self.assertEqual(token, expected)
            self.assertEqual(histories, [1])
        finally:
            mx.set_default_device(prev_device)

    def test_mtp_recursive_prefill_fills_all_layers(self):
        prev_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            model = _make_model(mtp_num_hidden_layers=2)
            model.language_model.mtp = MTPModule(model.language_model.args)
            model.eval()
            mx.eval(model.parameters())

            mtp_cache = model.make_mtp_cache()
            self.assertEqual(len(mtp_cache), 2)
            hidden = mx.random.normal((1, 3, 8))
            tokens = mx.array([[2, 3, 4]])
            # Replicate the recursive prefill: layer 0 consumes the backbone
            # hidden, layer 1 consumes layer 0's fused output.
            for j in range(2):
                _, hidden = model.mtp_forward(
                    hidden, tokens, mtp_cache, spec_step_idx=j
                )
                mx.eval(hidden)
            mx.eval([c.state for c in mtp_cache])
            # Every layer's causal cache must be warmed with the prompt.
            self.assertEqual(mtp_cache[0].offset, 3)
            self.assertEqual(mtp_cache[1].offset, 3)
        finally:
            mx.set_default_device(prev_device)


if __name__ == "__main__":
    unittest.main()
