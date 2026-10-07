from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
import torch

from forge_resshift import inference
from forge_resshift.models import STUDENTS, VQGAN, ensure_asset


logger = logging.getLogger(__name__)
GIB = 1024 ** 3


@dataclass
class ModelBundle:
    scale: int
    cfg: object
    diffusion: object
    student: torch.nn.Module
    vqgan: torch.nn.Module
    conditioning: str
    alignment: int


class ResShiftRuntime:
    def __init__(self, model_dir, device, memory=None, state=None):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.memory = memory
        self.state = state
        self.bundle = None
        self.lock = threading.RLock()

    def use_bf16(self):
        # The port's autocast path is CUDA-only. Other backends use FP32.
        if self.device.type != "cuda":
            return False
        if self.memory is not None:
            return bool(self.memory.should_use_bf16(self.device))
        return bool(torch.cuda.is_bf16_supported())

    def check_interruption(self):
        if self.state is not None and self.state.interrupted:
            raise InterruptedError("Distilled ResShift interrupted")

    def load(self, scale):
        with self.lock:
            self.check_interruption()
            if scale not in STUDENTS:
                raise ValueError("Distilled ResShift supports native x2 and x4 reconstruction")
            if self.bundle is not None and self.bundle.scale == scale:
                return self.bundle
            self.bundle = None
            student_path = ensure_asset(STUDENTS[scale], self.model_dir)
            vqgan_path = ensure_asset(VQGAN, self.model_dir)
            cfg = inference.load_configs(vqgan_path, inference.CONFIGS[f"x{scale}"])
            with safe_open(str(student_path), framework="pt", device="cpu") as stream:
                metadata = stream.metadata() or {}
            conditioning = metadata.get("cond_lq", "latent")
            if conditioning not in ("latent", "pixel"):
                raise ValueError(f"Unsupported ResShift conditioning: {conditioning}")
            weights = load_file(str(student_path), device="cpu")
            with torch.random.fork_rng(devices=[]):
                student = inference.build_student(
                    cfg, torch.device("cpu"), noise_mode=metadata.get("noise_mode", "concat"),
                    noise_channels=int(metadata.get("noise_channels", "1")),
                )
                inference.load_student_weights(student, weights)
                vqgan = inference.build_autoencoder(cfg, torch.device("cpu"))
            student.requires_grad_(False)
            self.bundle = ModelBundle(
                scale, cfg, inference.build_diffusion(cfg), student, vqgan,
                conditioning, inference.swin_align(cfg),
            )
            logger.info("Distilled ResShift: loaded x%s student and VQGAN; conditioning=%s",
                        scale, conditioning)
            return self.bundle

    def _set_precision(self, bundle, amp):
        dtype = torch.bfloat16 if amp else torch.float32
        current = next(bundle.vqgan.parameters()).dtype
        if current == dtype:
            return
        if dtype == torch.float32:
            # Re-load original FP32 weights rather than upcast rounded BF16 weights.
            bundle.vqgan = inference.build_autoencoder(bundle.cfg, torch.device("cpu"))
        else:
            bundle.vqgan.to(dtype=dtype)

    @torch.inference_mode()
    def enhance(self, image, scale, seed=None):
        with self.lock:
            self.check_interruption()
            started = time.monotonic()
            bundle = self.load(scale)
            amp = self.use_bf16()
            self._set_precision(bundle, amp)
            source = image.convert("RGB")
            pixels = np.array(source, dtype=np.float32) / 127.5 - 1.0
            tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
            if seed is None:
                seed = int.from_bytes(hashlib.sha256(source.tobytes()).digest()[:8], "little")
            tile = 256
            try:
                if self.memory is not None and self.device.type != "cpu":
                    size = self.memory.module_size(bundle.student) + self.memory.module_size(bundle.vqgan)
                    self.memory.free_memory(size + 2 * GIB, self.device)
                bundle.student.to(self.device)
                bundle.vqgan.to(self.device)
                if self.memory is not None and self.device.type != "cpu":
                    available = self.memory.get_free_memory(self.device)
                    if available >= (3 if amp else 6) * GIB:
                        tile = 512
                while True:
                    self.check_interruption()
                    logger.info("Distilled ResShift: x%s, %s, %s, tile=%s, overlap=64",
                                scale, self.device, "BF16" if amp else "FP32", tile)
                    try:
                        output = inference.upscale(
                            bundle.cfg, bundle.diffusion, bundle.vqgan, bundle.student,
                            tensor, self.device, scale, bundle.alignment,
                            overlap=64, chop=tile, tile_batch=1, amp=amp,
                            seed=seed, progress=self.check_interruption,
                            shared_noise=True, cond_mode=bundle.conditioning,
                        )
                        if not torch.isfinite(output).all():
                            raise FloatingPointError("Non-finite stitched ResShift pixels")
                    except FloatingPointError as error:
                        if not amp:
                            raise RuntimeError("Distilled ResShift produced non-finite pixels in FP32") from error
                        logger.warning("Distilled ResShift: non-finite BF16 output; retrying in FP32")
                        bundle.vqgan.to("cpu")
                        self._set_precision(bundle, False)
                        bundle.vqgan.to(self.device)
                        amp, tile = False, 256
                        continue
                    except RuntimeError as error:
                        if tile > 256 and "out of memory" in str(error).lower():
                            tile = 256
                            if self.memory is not None:
                                self.memory.soft_empty_cache()
                            continue
                        raise
                    expected = (1, 3, source.height * scale, source.width * scale)
                    if tuple(output.shape) != expected:
                        raise RuntimeError(f"Unexpected ResShift output shape: {tuple(output.shape)}")
                    values = ((output[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5)
                    result = Image.fromarray(values.round().to(torch.uint8).numpy())
                    logger.info("Distilled ResShift: neural reconstruction completed in %.2fs",
                                time.monotonic() - started)
                    return result
            finally:
                bundle.student.to("cpu")
                bundle.vqgan.to("cpu")
                if self.memory is not None:
                    self.memory.soft_empty_cache()


_runtime = None


def get_runtime():
    global _runtime
    if _runtime is None:
        from backend import memory_management
        from modules import devices, shared
        from modules.paths import models_path

        _runtime = ResShiftRuntime(
            Path(models_path) / "DistilledResShift",
            devices.device_esrgan or memory_management.get_torch_device(),
            memory=memory_management, state=shared.state,
        )
    return _runtime
