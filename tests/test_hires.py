import importlib.util
import logging
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class HostScript:
    def before_process(self, p, *args):
        pass

    def postprocess_image_after_composite(self, p, pp, *args):
        pass

    def postprocess(self, p, processed, *args):
        pass


class HostProcessing(types.SimpleNamespace):
    def calculate_target_resolution(self):
        if self.hr_resize_x or self.hr_resize_y:
            width = self.hr_resize_x or self.hr_resize_y * self.width / self.height
            height = self.hr_resize_y or self.hr_resize_x * self.height / self.width
        else:
            width, height = self.width * self.hr_scale, self.height * self.hr_scale
        self.hr_upscale_to_x = round(width / 8) * 8
        self.hr_upscale_to_y = round(height / 8) * 8
        self.extra_generation_params["Hires upscale"] = self.hr_scale


class HiresIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.state = types.SimpleNamespace(interrupted=False)
        self.shared = types.SimpleNamespace(sd_upscalers=[], state=self.state)
        host = types.ModuleType("modules")
        host.shared = self.shared
        host.script_callbacks = types.SimpleNamespace(on_before_ui=lambda callback: None)
        host.scripts = types.SimpleNamespace(Script=HostScript, AlwaysVisible=object())
        host.upscaler = types.SimpleNamespace(
            Upscaler=type("Upscaler", (), {"__init__": lambda self: None}),
            UpscalerData=lambda *args, **kwargs: None,
        )
        backend_logging = types.ModuleType("backend.logging")
        backend_logging.setup_logger = lambda logger: logger.setLevel(logging.INFO)
        self.host_patch = patch.dict(sys.modules, {
            "modules": host, "backend.logging": backend_logging,
        })
        self.host_patch.start()
        self.addCleanup(self.host_patch.stop)
        spec = importlib.util.spec_from_file_location(
            "resshift_hires_test", ROOT / "scripts" / "resshift_upscaler.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.script = getattr(self.module, "Script", HostScript)()
        self.calls = []

        def reconstruct(image, factor):
            self.calls.append((image.size, image.mode, image.tobytes(), factor))
            return image.resize((image.width * factor, image.height * factor), Image.Resampling.NEAREST)

        self.inference_patch = patch("forge_resshift.upscaler.enhance", side_effect=reconstruct)
        self.inference_patch.start()
        self.addCleanup(self.inference_patch.stop)

    def processing(self, **overrides):
        params = dict(
            enable_hr=True, hr_upscaler="Distilled ResShift", width=832, height=1216,
            hr_scale=2, hr_resize_x=0, hr_resize_y=0, hr_second_pass_steps=8,
            denoising_strength=0.6, hr_cfg=1, hr_sampler_name="Euler", hr_scheduler="Simple",
            batch_size=1, n_iter=1, iteration=0,
            extra_generation_params={"Hires steps": 8, "Denoising strength": 0.6},
        )
        params.update(overrides)
        return HostProcessing(**params)

    def run_image(self, p, image=None, index=0):
        source = image if image is not None else Image.new("RGB", (p.width, p.height), (20, 80, 140))
        pp = types.SimpleNamespace(image=source, index=index)
        self.script.postprocess_image_after_composite(p, pp)
        return pp.image

    def test_disables_only_the_native_hires_pass_and_keeps_base_resolution(self):
        p = self.processing()
        self.script.before_process(p)
        self.assertFalse(p.enable_hr)
        self.assertEqual((p.width, p.height), (832, 1216))
        self.assertEqual((p.hr_scale, p.hr_resize_x, p.hr_resize_y), (2, 0, 0))
        self.assertEqual(self.run_image(p).size, (1664, 2432))
        self.assertEqual(len(self.calls), 1)

    def test_preserves_explicit_targets_before_disabling_hires(self):
        for resize, expected in [((1000, 1400), (1000, 1400)), ((1248, 0), (1248, 1824))]:
            with self.subTest(resize=resize):
                self.calls.clear()
                p = self.processing(hr_resize_x=resize[0], hr_resize_y=resize[1])
                self.script.before_process(p)
                self.assertEqual((p.width, p.height), (832, 1216))
                self.assertEqual(self.run_image(p).size, expected)
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(self.calls[0][3], 2)

    def test_requested_scale_selects_one_native_student_and_one_optional_resize(self):
        resize = Image.Image.resize
        for scale, size, factor, methods in [
            (2, (1664, 2432), 2, [Image.Resampling.NEAREST]),
            (1.5, (1248, 1824), 2, [Image.Resampling.NEAREST, Image.Resampling.LANCZOS]),
            (4, (3328, 4864), 4, [Image.Resampling.NEAREST]),
        ]:
            with self.subTest(scale=scale):
                self.calls.clear()
                p = self.processing(hr_scale=scale)
                self.script.before_process(p)
                used = []

                def track(image, target, resample=None, **kwargs):
                    used.append(resample)
                    return resize(image, target, resample=resample, **kwargs)

                with patch.object(Image.Image, "resize", track):
                    output = self.run_image(p)
                self.assertEqual(output.size, size)
                self.assertEqual([call[3] for call in self.calls], [factor])
                self.assertEqual(used, methods)

    def test_receives_unmodified_base_rgb_pixels(self):
        p = self.processing(width=24, height=16)
        source = Image.new("RGB", (24, 16))
        source.putdata([(index % 256, index * 7 % 256, index * 13 % 256) for index in range(384)])
        self.script.before_process(p)
        self.run_image(p, source)
        self.assertEqual(self.calls, [((24, 16), "RGB", source.tobytes(), 2)])

    def test_each_image_runs_once_across_batches_and_repeated_callbacks(self):
        for batch_size, count in [(1, 1), (3, 1), (2, 3)]:
            with self.subTest(batch_size=batch_size, count=count):
                self.calls.clear()
                p = self.processing(width=24, height=16, batch_size=batch_size, n_iter=count)
                self.script.before_process(p)
                for batch in range(count):
                    p.iteration = batch
                    for item in range(batch_size):
                        pp = types.SimpleNamespace(image=Image.new("RGB", (24, 16)), index=batch * batch_size + item)
                        self.script.postprocess_image_after_composite(p, pp)
                        self.script.postprocess_image_after_composite(p, pp)
                        self.assertEqual(pp.image.size, (48, 32))
                self.assertEqual(len(self.calls), batch_size * count)

    def test_hires_diffusion_settings_do_not_affect_output_or_neural_arguments(self):
        outputs = []
        for settings in [dict(), dict(denoising_strength=0.95, hr_second_pass_steps=40,
                                     hr_cfg=12, hr_sampler_name="DPM++ 2M", hr_scheduler="Karras")]:
            p = self.processing(width=24, height=16, **settings)
            self.script.before_process(p)
            self.assertFalse(p.enable_hr)
            outputs.append(self.run_image(p).tobytes())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(self.calls[0], self.calls[1])

    def test_metadata_describes_only_applied_upscaling_and_final_size(self):
        p = self.processing()
        self.script.before_process(p)
        self.run_image(p)
        self.assertEqual(p.extra_generation_params.get("ResShift"), "Distilled")
        self.assertEqual(p.extra_generation_params.get("ResShift scale"), 2)
        self.assertEqual(p.extra_generation_params.get("Size"), "1664x2432")
        self.assertNotIn("Denoising strength", p.extra_generation_params)
        self.assertFalse(any(key.startswith("Hires ") for key in p.extra_generation_params))
        self.assertNotIn("ResShift resize", p.extra_generation_params)

    def test_other_upscalers_and_disabled_hires_are_untouched(self):
        for options in [dict(hr_upscaler="Lanczos"), dict(hr_upscaler="Latent"), dict(enable_hr=False)]:
            with self.subTest(options=options):
                p = self.processing(**options)
                before = dict(vars(p))
                before["extra_generation_params"] = dict(p.extra_generation_params)
                source = Image.new("RGB", (832, 1216))
                self.script.before_process(p)
                self.assertIs(self.run_image(p, source), source)
                self.assertEqual(vars(p), before)
        self.assertEqual(self.calls, [])

    def test_interrupted_generation_does_not_run_or_claim_upscaling(self):
        p = self.processing()
        self.script.before_process(p)
        self.state.interrupted = True
        source = Image.new("RGB", (832, 1216))
        self.assertIs(self.run_image(p, source), source)
        self.assertEqual(self.calls, [])
        self.assertNotIn("ResShift", p.extra_generation_params)
        self.assertNotIn("Size", p.extra_generation_params)

    def test_cancellation_during_reconstruction_keeps_output_unmarked(self):
        p = self.processing()
        self.script.before_process(p)
        source = Image.new("RGB", (832, 1216))
        with patch("forge_resshift.upscaler.enhance", side_effect=InterruptedError("cancelled")):
            self.assertIs(self.run_image(p, source), source)
        self.assertNotIn("ResShift", p.extra_generation_params)
        self.assertNotIn("Size", p.extra_generation_params)

    def test_restores_request_state_without_changing_returned_metadata(self):
        p = self.processing()
        original = dict(p.extra_generation_params)
        self.script.before_process(p)
        self.run_image(p)
        processed = types.SimpleNamespace(extra_generation_params=p.extra_generation_params,
                                         width=832, height=1216, denoising_strength=0.6)
        self.script.postprocess(p, processed)
        self.assertTrue(p.enable_hr)
        self.assertEqual(p.extra_generation_params, original)
        self.assertEqual(processed.extra_generation_params.get("ResShift"), "Distilled")
        self.assertEqual((processed.width, processed.height), (1664, 2432))
        self.assertIsNone(processed.denoising_strength)


if __name__ == "__main__":
    unittest.main()
