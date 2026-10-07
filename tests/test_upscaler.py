import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class UpscalerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.callbacks = []
        cls.shared = types.SimpleNamespace(sd_upscalers=[])
        host = types.ModuleType("modules")
        host.shared = cls.shared
        host.script_callbacks = types.SimpleNamespace(on_before_ui=cls.callbacks.append)
        host.scripts = types.SimpleNamespace(Script=type("Script", (), {}), AlwaysVisible=object())
        host.upscaler = types.SimpleNamespace(
            Upscaler=type("Upscaler", (), {"__init__": lambda self: None}),
            UpscalerData=lambda name, path, upscaler, scale: types.SimpleNamespace(
                name=name, data_path=path, scaler=upscaler, scale=scale
            ),
        )
        cls.host_patch = patch.dict(sys.modules, {"modules": host})
        cls.host_patch.start()
        try:
            from forge_resshift.upscaler import UpscalerDistilledResShift
            cls.upscaler_type = UpscalerDistilledResShift
        except ImportError as error:
            cls.host_patch.stop()
            raise AssertionError("The native Hires upscaler is not implemented yet") from error

    @classmethod
    def tearDownClass(cls):
        cls.host_patch.stop()

    def test_registration_survives_forge_rebuilding_the_upscaler_list(self):
        self.callbacks.clear()
        spec = importlib.util.spec_from_file_location(
            "resshift_script_test", ROOT / "scripts" / "resshift_upscaler.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        existing = types.SimpleNamespace(name="Lanczos")
        self.shared.sd_upscalers = [existing]
        for callback in self.callbacks:
            callback()
            callback()
        self.assertEqual([item.name for item in self.shared.sd_upscalers],
                         ["Lanczos", "Distilled ResShift"])
        self.assertIs(self.shared.sd_upscalers[0], existing)

    def test_arbitrary_scale_uses_one_neural_pass_and_exact_dimensions(self):
        upscaler = self.upscaler_type()
        source = Image.new("RGB", (20, 12), "red")
        calls = []

        def enhance(image, scale):
            calls.append((image.mode, image.size, scale))
            return image.resize((image.width * scale, image.height * scale))

        with patch("forge_resshift.upscaler.enhance", side_effect=enhance):
            output = upscaler.upscale(source, 1.5)
        self.assertEqual(output.size, (30, 18))
        self.assertEqual(calls, [("RGB", (20, 12), 2)])

    def test_large_scale_does_not_repeat_generative_reconstruction(self):
        upscaler = self.upscaler_type()
        source = Image.new("RGB", (12, 8))
        with patch("forge_resshift.upscaler.enhance", return_value=source.resize((48, 32))) as run:
            output = upscaler.upscale(source, 6)
        self.assertEqual(output.size, (72, 48))
        run.assert_called_once()
        self.assertEqual(run.call_args.args[1], 4)

    def test_rgba_and_grayscale_are_normalized_to_rgb(self):
        for mode in ("RGBA", "L"):
            with self.subTest(mode=mode):
                upscaler = self.upscaler_type()
                source = Image.new(mode, (12, 8))
                with patch("forge_resshift.upscaler.enhance",
                           side_effect=lambda image, scale: image.resize((24, 16))) as run:
                    output = upscaler.upscale(source, 2)
                self.assertEqual(run.call_args.args[0].mode, "RGB")
                self.assertEqual(output.mode, "RGB")

    def test_neural_failure_is_not_replaced_by_interpolation(self):
        upscaler = self.upscaler_type()
        with patch("forge_resshift.upscaler.enhance", side_effect=RuntimeError("inference failed")):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                upscaler.upscale(Image.new("RGB", (12, 8)), 2)

    def test_invalid_scale_is_rejected_before_inference(self):
        upscaler = self.upscaler_type()
        for scale in (0, -1, math.nan, math.inf):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    upscaler.upscale(Image.new("RGB", (12, 8)), scale)


if __name__ == "__main__":
    unittest.main()
