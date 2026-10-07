# Distilled ResShift for Forge Neo

A native Hires upscaler that reconstructs an image using a specialized one-step
diffusion model. It works on RGB pixels, independently of the checkpoint that
generated the source image.

## Installation and use

1. Copy this folder into `Forge/extensions/forge-resshift-upscaler`.
2. Restart Forge completely. The installer adds `einops` only if missing, without
   replacing Torch or resolving its dependencies.
3. Enable **Hires. fix**, select **Distilled ResShift** in the existing upscaler
   dropdown, and generate normally.

The first use downloads the appropriate student and VQGAN to
`Forge/models/DistilledResShift`: approximately 700 MB for either factor, or
1.18 GB if both factors are used. Internet access is needed only for missing or
invalid files. Complete files in that directory can also be supplied manually;
the download URLs and checksums are in `forge_resshift/models.py`.

Requested factors up to 2 use the x2 student; larger factors use x4. Exactly one
neural reconstruction runs, followed by a Lanczos resize when needed to reach
the requested dimensions. Factors above 4 do not gain another neural detail
pass. The normal Hires diffusion pass still follows, so its denoising strength
affects the final image.

Precision and tile size are automatic. Supported CUDA devices use BF16 when
Forge permits it; other paths use FP32. Full-image stitching stays in RAM, and
weights move back to CPU after reconstruction. Errors are reported rather than
silently substituted with interpolation. No additional panel is required.

## Validation status

The x2 and x4 checkpoints ran successfully in FP32 on CPU, including overlapping
tiles and deterministic repetition. Focused tests cover registration timing,
dimensions, downloads, conditioning, padding, invalid output and precision.

Full generation inside Forge Neo and GPU behavior have not been validated.
Visual improvement on generated photos, illustrations and text remains to be
evaluated. Compatibility with every GPU is not guaranteed.

Run the focused tests with `python -m unittest discover -s tests -v` in an
environment containing Torch, NumPy, Pillow, PyYAML, safetensors and einops.

For the upstream non-commercial license terms and attribution, see
[NOTICE.md](NOTICE.md) and [LICENSE](LICENSE).
