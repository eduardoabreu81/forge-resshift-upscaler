# Distilled ResShift for Forge Neo

A specialized one-step diffusion upscaler selected through Forge's existing
Hires controls. It works on RGB pixels, independently of the checkpoint that
generated the source image, and returns the final upscaled image.

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
pass. Selecting Distilled ResShift disables the native Hires diffusion pass for
that request. The base image is generated at the original resolution, decoded
once, then reconstructed by ResShift before individual image saving and grid
creation. There is no checkpoint VAE re-encode or second checkpoint sampling.

Hires denoising strength, steps, CFG, sampler, scheduler, checkpoint and modules
are ignored in this mode. Other upscalers retain the normal Hires behavior.
The requested scale or explicit dimensions still determine the output size.
The existing pixel-derived ResShift seed is unchanged, so identical RGB input
and ResShift parameters produce deterministic output on the same execution path.

PNG metadata records `ResShift: Distilled`, the requested and native scales,
input size and final `Size`. `ResShift resize: Lanczos` appears only when needed.
Unused Hires diffusion parameters are omitted. Interrupted reconstruction does
not mark the current base image as a completed ResShift upscale.

The terminal reports `native Hires disabled, sample_img2img will not run`, then
one `passes=1, sample_img2img=0` completion per image. Existing runtime messages
show the execution device, precision and tile size; a tile-memory retry changes
the logged tile size from 512 to 256.

Precision and tile size are automatic. Supported CUDA devices use BF16 when
Forge permits it; other paths use FP32. Full-image stitching stays in RAM, and
weights move back to CPU after reconstruction. Errors are reported rather than
silently substituted with interpolation. No additional panel is required.

## Validation status

The x2 and x4 checkpoints ran successfully in FP32 on CPU, including overlapping
tiles and deterministic repetition. Focused tests cover registration timing,
dimensions, downloads, conditioning, padding, invalid output and precision.

Focused integration tests cover upscale-only routing, unchanged RGB input,
2x/1.5x/4x sizing, multi-image batches, multiple batches, cancellation, metadata
and isolation from other upscalers. Full generation inside Forge Neo and GPU
behavior have not been validated locally.
Visual improvement on generated photos, illustrations and text remains to be
evaluated. Compatibility with every GPU is not guaranteed.

Run the focused tests with `python -m unittest discover -s tests -v` in an
environment containing Torch, NumPy, Pillow, PyYAML, safetensors and einops.

For the upstream non-commercial license terms and attribution, see
[NOTICE.md](NOTICE.md) and [LICENSE](LICENSE).
