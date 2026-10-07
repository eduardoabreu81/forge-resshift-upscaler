from copy import copy
import logging
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from PIL import Image
from modules import script_callbacks, scripts, shared


extension_root = str(Path(__file__).resolve().parents[1])
if extension_root not in sys.path:
    sys.path.insert(0, extension_root)

from forge_resshift.upscaler import UpscalerDistilledResShift, register_upscaler


# Forge rebuilds sd_upscalers after loading extension scripts.
script_callbacks.on_before_ui(register_upscaler)


class Script(scripts.Script):
    create_group = False

    def __init__(self):
        super().__init__()
        from backend.logging import setup_logger

        self.logger = logging.getLogger("forge_resshift.hires")
        setup_logger(self.logger)
        setup_logger(logging.getLogger("forge_resshift.runtime"))

    def title(self):
        return UpscalerDistilledResShift.name

    def show(self, is_img2img):
        return False if is_img2img else scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def before_process(self, p, *args):
        if not getattr(p, "enable_hr", False) or p.hr_upscaler != UpscalerDistilledResShift.name:
            return

        # Resolve native Hires sizing on a copy without changing base generation.
        sizing = copy(p)
        sizing.extra_generation_params = dict(p.extra_generation_params)
        sizing.calculate_target_resolution()
        target = (int(sizing.hr_upscale_to_x), int(sizing.hr_upscale_to_y))
        scale = (max(target[0] / p.width, target[1] / p.height)
                 if p.hr_resize_x or p.hr_resize_y else p.hr_scale)
        if not math.isfinite(scale) or scale <= 0 or min(target) <= 0:
            raise ValueError("Distilled ResShift requires a positive output size and scale")

        p._resshift_upscale = SimpleNamespace(
            target=target, scale=scale, completed=set(),
            original_params=dict(p.extra_generation_params),
        )
        p.enable_hr = False
        self._reset_metadata(p)
        self.logger.info(
            "Distilled ResShift: upscale-only mode, base=%sx%s, target=%sx%s; "
            "native Hires disabled, sample_img2img will not run",
            p.width, p.height, *target,
        )

    def _reset_metadata(self, p):
        request = p._resshift_upscale
        for key in list(p.extra_generation_params):
            if (key.startswith("Hires ") or key.startswith("ResShift")
                    or key in ("Denoising strength", "Token merging ratio hr")):
                del p.extra_generation_params[key]
        if "Size" in request.original_params:
            p.extra_generation_params["Size"] = request.original_params["Size"]
        else:
            p.extra_generation_params.pop("Size", None)

    def postprocess_image_after_composite(self, p, pp, *args):
        request = getattr(p, "_resshift_upscale", None)
        if request is None or pp.index in request.completed:
            return
        self._reset_metadata(p)
        if shared.state.interrupted:
            self.logger.info("Distilled ResShift: image %s interrupted before reconstruction", pp.index + 1)
            return

        source = pp.image.convert("RGB")
        factor = 2 if request.scale <= 2 else 4
        started = time.monotonic()
        try:
            output = UpscalerDistilledResShift().upscale(source, factor)
        except InterruptedError:
            self.logger.info("Distilled ResShift: image %s reconstruction interrupted", pp.index + 1)
            return
        if shared.state.interrupted:
            return

        resized = output.size != request.target
        if resized:
            output = output.resize(request.target, Image.Resampling.LANCZOS)
        pp.image = output
        request.completed.add(pp.index)
        p.extra_generation_params.update({
            "ResShift": "Distilled",
            "ResShift scale": request.scale,
            "ResShift native scale": factor,
            "ResShift input size": f"{source.width}x{source.height}",
            "Size": f"{output.width}x{output.height}",
        })
        if resized:
            p.extra_generation_params["ResShift resize"] = "Lanczos"
        self.logger.info(
            "Distilled ResShift: image %s, RGB %sx%s -> %sx%s, student=x%s, "
            "passes=1, sample_img2img=0, completed in %.2fs",
            pp.index + 1, source.width, source.height, output.width, output.height,
            factor, time.monotonic() - started,
        )

    def postprocess(self, p, processed, *args):
        request = getattr(p, "_resshift_upscale", None)
        if request is None:
            return
        processed.denoising_strength = None
        if request.completed:
            processed.width, processed.height = request.target
        # Processed retains the applied metadata; restore the reusable request.
        p.extra_generation_params = request.original_params
        p.enable_hr = True
        del p._resshift_upscale
