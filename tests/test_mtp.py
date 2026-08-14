# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.generate import mtp_generate_step
from mlx_lm.models.cache import KVCache, TrimmableArraysCache, make_prompt_cache
from mlx_lm.models.qwen3_5 import Model, ModelArgs, MTPModule


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


if __name__ == "__main__":
    unittest.main()
