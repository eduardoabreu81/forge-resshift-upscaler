import launch


# Forge provides Torch, NumPy, Pillow, PyYAML and safetensors.
# Do not replace its Torch build or resolve dependencies into its environment.
if not launch.is_installed("einops"):
    launch.run_pip("install --no-deps einops", "Distilled ResShift: einops")
