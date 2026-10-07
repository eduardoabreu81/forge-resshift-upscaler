from pathlib import Path
import sys

from modules import script_callbacks


extension_root = str(Path(__file__).resolve().parents[1])
if extension_root not in sys.path:
    sys.path.insert(0, extension_root)

from forge_resshift.upscaler import register_upscaler


# Forge rebuilds sd_upscalers after loading extension scripts.
script_callbacks.on_before_ui(register_upscaler)
