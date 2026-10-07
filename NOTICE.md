# Third-party code and weights

This extension ports the inference core and vendored ResShift network from
[ComfyUI-Distilled-ResShift](https://github.com/sorryhyun/ComfyUI-Distilled-ResShift),
revision `628fb063669a071b3957406a351b0b8ed48e335f`.
Its code is distributed under the upstream CC BY-NC-SA 4.0 terms in `LICENSE`.
The vendored ResShift network retains S-Lab License 1.0 in
`forge_resshift/_vendor/resshift/LICENSE`.
Both licenses restrict commercial use.

The research implementation is based on
[RSD](https://github.com/Daniil-Selikhanovych/RSD),
[SinSR](https://github.com/wyf0912/SinSR), and
[ResShift](https://github.com/zsyOAOA/ResShift).

The distilled students are published by `sorryhyun` on Hugging Face and are not
represented as the RSD authors' official checkpoints. Downloads use pinned
repository revisions and SHA-256 checksums. The VQGAN comes from the upstream
ResShift v2.0 release; its checksum was recorded from that release asset.
Weights are downloaded separately on first use, under their upstream terms.

Port changes: package-qualified imports to avoid host namespace collisions;
native Forge upscaler registration; first-use downloads; automatic precision
and tile selection; CPU image stitching and shared noise; padding for small
images; invalid-output detection; safe VQGAN checkpoint loading. Inference uses
PyTorch tuple/initialization helpers and disabled stochastic depth instead of
importing timm. The vendored network is used for inference only.
