"""Focused tests for Gemma4 tensor-name mapping and QAT dequantization."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from conversion.gemma import Gemma4Model
from conversion.base import ModelBase, gguf


class TestGemma4Conversion(unittest.TestCase):
    def setUp(self):
        self.model = object.__new__(Gemma4Model)
        self.model.tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.GEMMA4, 1)

    def _qat_model(self, tensors, quant_config):
        model = object.__new__(Gemma4Model)
        model.ftype = gguf.LlamaFileType.MOSTLY_F16
        model.hparams = {"quantization_config": quant_config}
        model.model_tensors = {name: (lambda tensor=tensor: tensor) for name, tensor in tensors.items()}
        return model

    def test_standard_names_use_arch_tensor_map(self):
        for source, expected in (
            ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
            ("model.layers.0.self_attn.o_proj.weight", "blk.0.attn_output.weight"),
            ("model.layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"),
            ("model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"),
            ("model.layers.0.router.proj.weight", "blk.0.ffn_gate_inp.weight"),
        ):
            with self.subTest(source=source):
                self.assertEqual(self.model.map_tensor_name(source), expected)

    def test_shared_name_mapping_preserves_other_architectures(self):
        for arch in (gguf.MODEL_ARCH.LLAMA, gguf.MODEL_ARCH.GEMMA2, gguf.MODEL_ARCH.QWEN2):
            model = object.__new__(ModelBase)
            model.model_arch = arch
            model.tensor_map = gguf.get_tensor_name_map(arch, 1)
            with self.subTest(arch=arch.name):
                self.assertEqual(
                    model.map_tensor_name("model.layers.0.self_attn.q_proj.weight"),
                    "blk.0.attn_q.weight",
                )
                self.assertEqual(
                    model.map_tensor_name("model.layers.0.mlp.gate_proj.weight"),
                    "blk.0.ffn_gate.weight",
                )

    def test_switch_glu_expert_names_use_canonical_names(self):
        for projection, tensor_type in (
            ("gate_proj", gguf.MODEL_TENSOR.FFN_GATE_EXP),
            ("up_proj", gguf.MODEL_TENSOR.FFN_UP_EXP),
            ("down_proj", gguf.MODEL_TENSOR.FFN_DOWN_EXP),
        ):
            with self.subTest(projection=projection):
                source = f"model.layers.0.experts.switch_glu.{projection}.weight"
                self.assertEqual(
                    self.model.map_tensor_name(source),
                    self.model.format_tensor_name(tensor_type, 0),
                )

    def test_quant_biases_are_not_mapped_as_model_biases(self):
        with self.assertRaisesRegex(ValueError, "Can not map tensor"):
            self.model.map_tensor_name("model.layers.0.self_attn.q_proj.biases")

    def test_weight_scale_is_not_mistaken_for_weight(self):
        with self.assertRaisesRegex(ValueError, "Can not map tensor"):
            self.model.map_tensor_name("model.layers.0.mlp.gate_proj.weight_scale")

    def test_nvfp4_u32_weights_dequantize_and_consume_scales(self):
        weight_name = "model.layers.0.self_attn.q_proj.weight"
        scale_name = "model.layers.0.self_attn.q_proj.scales"
        tensors = {
            weight_name: torch.full((1, 8), 0x11111111, dtype=torch.int32),
            scale_name: torch.full((1, 4), 0x38, dtype=torch.uint8),
        }
        model = self._qat_model(tensors, {"mode": "nvfp4", "bits": 4, "group_size": 16})

        model._dequantize_nvfp4_checkpoint()

        self.assertNotIn(scale_name, model.model_tensors)
        dequantized = model.model_tensors[weight_name]()
        self.assertEqual(tuple(dequantized.shape), (1, 64))
        torch.testing.assert_close(dequantized, torch.full((1, 64), 0.5))

    def test_per_layer_int8_override_dequantizes_unsigned_packed_bytes_with_bias(self):
        weight_name = "model.layers.0.mlp.gate_proj.weight"
        scale_name = "model.layers.0.mlp.gate_proj.scales"
        bias_name = "model.layers.0.mlp.gate_proj.biases"
        words = np.full((1, 16), 0xFF0001FE, dtype=np.uint32).view(np.int32).copy()
        tensors = {
            weight_name: torch.from_numpy(words),
            scale_name: torch.full((1, 1), 0.5, dtype=torch.bfloat16),
            bias_name: torch.full((1, 1), -64.0, dtype=torch.bfloat16),
        }
        config = {
            "mode": "nvfp4",
            "bits": 4,
            "group_size": 16,
            "language_model.model.layers.0.mlp.gate_proj": {"bits": 8, "group_size": 64},
        }
        model = self._qat_model(tensors, config)

        model._dequantize_nvfp4_checkpoint()

        self.assertNotIn(scale_name, model.model_tensors)
        self.assertNotIn(bias_name, model.model_tensors)
        expected = torch.tensor([63.0, -63.5, -64.0, 63.5] * 16).reshape(1, 64)
        torch.testing.assert_close(model.model_tensors[weight_name](), expected)

    def test_expert_scales_after_filtering_are_consumed(self):
        weight_name = "model.layers.0.experts.switch_glu.gate_proj.weight"
        scale_name = "model.layers.0.experts.switch_glu.gate_proj.scales.weight"
        tensors = {
            weight_name: torch.full((2, 1, 8), 0x11111111, dtype=torch.int32),
            scale_name: torch.full((2, 1, 4), 0x38, dtype=torch.uint8),
        }
        model = self._qat_model(tensors, {"mode": "nvfp4", "bits": 4, "group_size": 16})

        model._dequantize_nvfp4_checkpoint()

        self.assertNotIn(scale_name, model.model_tensors)
        self.assertEqual(tuple(model.model_tensors[weight_name]().shape), (2, 1, 64))


if __name__ == "__main__":
    unittest.main()
