# ComfyUI-MOSS-PAG

Perturbed Attention Guidance for MOSS-Image, following the diffusers implementation.

The node **MOSS-Image Perturbed Attention Guidance** (`model/patch`) runs one extra conditional pass per step in which the self-attention of the chosen blocks returns its values, then applies `cfg + scale * (cond - perturbed)`.

- `scale`: PAG scale, 0 disables the extra pass.
- `layers`: block indices to perturb, e.g. `18, 20, 22` (MOSS-Image has 42).
- `adaptive_scale`: diffusers' `pag_adaptive_scale`.

Install by cloning into `ComfyUI/custom_nodes`.
