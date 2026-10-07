from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class InferenceBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from forge_resshift import inference
        cls.inference = inference

    def test_both_scales_use_the_actual_vae_and_swin_alignment(self):
        for tag in ("x2", "x4"):
            cfg = self.inference.load_configs("unused.pth", self.inference.CONFIGS[tag])
            self.assertEqual(self.inference.swin_align(cfg), 256)
            self.assertEqual(cfg.diffusion.params.sf, int(tag[1:]))

    def test_tiny_image_padding_and_cropping_keep_all_pixels(self):
        class Autoencoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def encode(self, value):
                return F.avg_pool2d(value, 4)

            def decode(self, value, force_not_quantize):
                return F.interpolate(value, scale_factor=4, mode="nearest")

        class Student(torch.nn.Module):
            noise_channels = 1

            def forward(self, x, timestep, lq, eps):
                return lq

        cfg = self.inference.load_configs("unused.pth")
        diff = types.SimpleNamespace(kappa=2, num_timesteps=15, _scale_input=lambda x, t: x)
        source = torch.full((1, 3, 12, 20), 0.25)
        result = self.inference._sample_tile(
            cfg, diff, Autoencoder(), Student(), source, 256, torch.device("cpu"),
            False, torch.Generator().manual_seed(0),
        )
        self.assertEqual(result.shape, (1, 3, 12, 20))
        torch.testing.assert_close(result, source, rtol=0, atol=0)

    def test_overlap_blending_preserves_values_at_every_border(self):
        source = torch.arange(3 * 17 * 19, dtype=torch.float32).reshape(1, 3, 17, 19)
        tiles = self.inference.ImageSpliterTh(source, pch_size=12, stride=8, extra_bs=2)
        for tile, positions in tiles:
            tiles.update(tile, positions)
        output = tiles.gather()
        self.assertEqual(output.shape, source.shape)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, source)

    def test_pixel_conditioning_does_not_accidentally_use_vae_latents(self):
        pixels = torch.full((1, 3, 16, 16), 0.75)
        latent = torch.full((1, 3, 4, 4), -0.25)
        result = self.inference.make_cond_lq(pixels, latent, "pixel")
        torch.testing.assert_close(result, torch.full((1, 3, 4, 4), 0.75))
        self.assertIs(self.inference.make_cond_lq(pixels, latent, "latent"), latent)

    def test_infinite_neural_output_is_not_clamped_into_a_valid_image(self):
        class InvalidAutoencoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def encode(self, value):
                return F.avg_pool2d(value, 4)

            def decode(self, value, force_not_quantize):
                return torch.full((1, 3, 256, 256), float("inf"))

        class Student(torch.nn.Module):
            noise_channels = 1

            def forward(self, x, timestep, lq, eps):
                return lq

        cfg = self.inference.load_configs("unused.pth")
        diff = types.SimpleNamespace(kappa=2, num_timesteps=15, _scale_input=lambda x, t: x)
        with self.assertRaises(FloatingPointError):
            self.inference._sample_tile(
                cfg, diff, InvalidAutoencoder(), Student(), torch.zeros(1, 3, 256, 256),
                256, torch.device("cpu"), False, torch.Generator().manual_seed(0),
            )

    def test_non_cuda_backend_does_not_require_a_device_specific_generator(self):
        cfg = self.inference.load_configs("unused.pth")
        source = torch.full((1, 3, 12, 20), 0.25)
        with patch.object(self.inference, "_sample_tile",
                          side_effect=lambda cfg, diff, ae, model, pixels, *args, **kwargs: pixels):
            result = self.inference.upscale(
                cfg, None, None, None, source, torch.device("mps"), 2, 256,
                overlap=64, chop=256, tile_batch=1, amp=False, seed=123,
            )
        torch.testing.assert_close(result, torch.full((1, 3, 24, 40), 0.25))


if __name__ == "__main__":
    unittest.main()
