import logging
import math

from PIL import Image
from modules import shared, upscaler


logger = logging.getLogger(__name__)


def enhance(image, scale):
    from forge_resshift.runtime import get_runtime

    return get_runtime().enhance(image, scale)


class UpscalerDistilledResShift(upscaler.Upscaler):
    name = "Distilled ResShift"

    def __init__(self):
        super().__init__()
        self.scalers = [upscaler.UpscalerData(self.name, "auto", self, scale=2)]

    def load_model(self, path=None):
        from forge_resshift.runtime import get_runtime

        return get_runtime().load(4 if path == "4" else 2)

    def do_upscale(self, img, selected_model=None):
        return enhance(img.convert("RGB"), 4 if selected_model == "4" else 2)

    def upscale(self, img, scale, selected_model=None):
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Distilled ResShift requires a finite, positive scale")
        source = img.convert("RGB")
        target = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
        if scale <= 1:
            return source.resize(target, Image.Resampling.LANCZOS)
        factor = 2 if scale <= 2 else 4
        output = enhance(source, factor)
        if output.size != target:
            logger.info("Distilled ResShift: one neural x%s pass, final Lanczos resize to %sx%s",
                        factor, *target)
            output = output.resize(target, Image.Resampling.LANCZOS)
        return output


def register_upscaler():
    if not any(item.name == UpscalerDistilledResShift.name for item in shared.sd_upscalers):
        shared.sd_upscalers.extend(UpscalerDistilledResShift().scalers)
