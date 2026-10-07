from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


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


if __name__ == "__main__":
    unittest.main()
