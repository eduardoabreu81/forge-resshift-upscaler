from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class RuntimePrecisionTests(unittest.TestCase):
    def test_cpu_never_selects_cuda_autocast(self):
        from forge_resshift.runtime import ResShiftRuntime
        runtime = ResShiftRuntime(Path("unused"), torch.device("cpu"))
        self.assertFalse(runtime.use_bf16())

    def test_older_cuda_gpu_uses_fp32_without_manual_configuration(self):
        from forge_resshift.runtime import ResShiftRuntime
        runtime = ResShiftRuntime(Path("unused"), torch.device("cuda"))
        with patch("torch.cuda.is_bf16_supported", return_value=False):
            self.assertFalse(runtime.use_bf16())

    def test_forge_precision_override_is_respected(self):
        from forge_resshift.runtime import ResShiftRuntime
        from types import SimpleNamespace
        memory = SimpleNamespace(should_use_bf16=lambda device: False)
        runtime = ResShiftRuntime(Path("unused"), torch.device("cuda"), memory=memory)
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            self.assertFalse(runtime.use_bf16())


class RuntimeTileTests(unittest.TestCase):
    def test_low_memory_selects_small_tiles_and_releases_weights(self):
        self.check_tiles(1024 ** 3, False, [256])

    def test_tile_oom_retries_with_small_tiles_and_releases_weights(self):
        self.check_tiles(7 * 1024 ** 3, True, [512, 256])

    def check_tiles(self, free_memory, fail_large_tile, expected_tiles):
        from forge_resshift.runtime import ResShiftRuntime

        class DeviceProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.moves = []

            def to(self, device=None, **kwargs):
                self.moves.append(str(device))
                return self

        student, vqgan = DeviceProbe(), DeviceProbe()
        bundle = SimpleNamespace(student=student, vqgan=vqgan, cfg=None,
                                 diffusion=None, alignment=256, conditioning="latent")
        releases = []
        reservations = []
        memory = SimpleNamespace(
            should_use_bf16=lambda device: False,
            module_size=lambda model: 4,
            free_memory=lambda size, device: reservations.append(size),
            get_free_memory=lambda device: free_memory,
            soft_empty_cache=lambda: releases.append(True),
        )
        runtime = ResShiftRuntime(Path("unused"), torch.device("cuda"), memory=memory)
        tiles = []

        def infer(cfg, diffusion, autoencoder, model, pixels, device, scale, alignment, **kwargs):
            tiles.append(kwargs["chop"])
            if fail_large_tile and kwargs["chop"] == 512:
                raise RuntimeError("CUDA out of memory")
            return torch.zeros(1, 3, pixels.shape[2] * scale, pixels.shape[3] * scale)

        with patch.object(runtime, "load", return_value=bundle), \
                patch("forge_resshift.runtime.inference.upscale", side_effect=infer):
            output = runtime.enhance(Image.new("RGB", (24, 16)), 2)
        self.assertEqual(output.size, (48, 32))
        self.assertEqual(tiles, expected_tiles)
        self.assertEqual(reservations, [8 + 2 * 1024 ** 3])
        self.assertEqual(student.moves[-1], "cpu")
        self.assertEqual(vqgan.moves[-1], "cpu")
        self.assertEqual(len(releases), 2 if fail_large_tile else 1)


if __name__ == "__main__":
    unittest.main()
