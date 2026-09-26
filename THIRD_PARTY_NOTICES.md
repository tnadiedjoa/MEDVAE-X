# Third-party notices

This project includes code adapted from the following projects. Their licenses
apply to the corresponding files.

| Project | License | Files in this repository |
|---|---|---|
| [MedVAE](https://github.com/StanfordMIMI/MedVAE) — Stanford MIMI Lab | MIT | `medvae_eval/cvae/medvae_standalone.py`, `jepa_adaptation/models/medvae.py`, `jepa_adaptation/models/medvae_main.py`, `jepa_adaptation/models/modules/` |
| [taming-transformers](https://github.com/CompVis/taming-transformers) / [latent-diffusion](https://github.com/CompVis/latent-diffusion) (upstream of MedVAE's autoencoder and LPIPS wrapper) | MIT | `jepa_adaptation/models/modules/diffusionmodels*.py`, `distributions.py`, `losses.py` |
| [LPIPS](https://github.com/richzhang/PerceptualSimilarity) — Richard Zhang | BSD-2-Clause | LPIPS implementation in `jepa_adaptation/models/modules/losses.py` (weights downloaded at runtime) |
| [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) — Jun-Yan Zhu, Taesung Park | BSD | `NLayerDiscriminator` in `jepa_adaptation/models/discriminator.py` |

The ARCADE dataset (Popov et al., 2024) is distributed under CC0 on
[Zenodo](https://zenodo.org/records/10390295); it is downloaded by
`scripts/download_arcade.sh` and not stored in this repository.

## MedVAE license

```
MIT License

Copyright (c) 2025 Stanford MIMI Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.```
