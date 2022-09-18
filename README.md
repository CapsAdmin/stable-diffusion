# Stable Diffusion

_Stable Diffusion was made possible thanks to a collaboration with [Stability AI](https://stability.ai/) and [Runway](https://runwayml.com/) and builds upon our previous work:_

[**High-Resolution Image Synthesis with Latent Diffusion Models**](https://ommer-lab.com/research/latent-diffusion-models/)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Dominik Lorenz](https://github.com/qp-qp)\,
[Patrick Esser](https://github.com/pesser),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
_[CVPR '22 Oral](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) |
[GitHub](https://github.com/CompVis/latent-diffusion) | [arXiv](https://arxiv.org/abs/2112.10752) | [Project page](https://ommer-lab.com/research/latent-diffusion-models/)_

![txt2img-stable2](assets/stable-samples/txt2img/merged-0006.png)
[Stable Diffusion](#stable-diffusion-v1) is a latent text-to-image diffusion
model.
Thanks to a generous compute donation from [Stability AI](https://stability.ai/) and support from [LAION](https://laion.ai/), we were able to train a Latent Diffusion Model on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database.
Similar to Google's [Imagen](https://arxiv.org/abs/2205.11487),
this model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts.
With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM.
See [this section](#stable-diffusion-v1) below and the [model card](https://huggingface.co/CompVis/stable-diffusion).

## News

### April 2022

- Thanks to [Katherine Crowson](https://github.com/crowsonkb), classifier-free guidance received a ~2x speedup and the [PLMS sampler](https://arxiv.org/abs/2202.09778) is available. See also [this PR](https://github.com/CompVis/latent-diffusion/pull/51).

- Our 1.45B [latent diffusion LAION model](#text-to-image) was integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/multimodalart/latentdiffusion)

- More pre-trained LDMs are available:
  - A 1.45B [model](#text-to-image) trained on the [LAION-400M](https://arxiv.org/abs/2111.02114) database.
  - A class-conditional model on ImageNet, achieving a FID of 3.6 when using [classifier-free guidance](https://openreview.net/pdf?id=qw8AKxfYbI) Available via a [colab notebook](https://colab.research.google.com/github/CompVis/latent-diffusion/blob/main/scripts/latent_imagenet_diffusion.ipynb) [![][colab]][colab-cin].

## Requirements

A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate ldm
```

You can also update an existing [latent diffusion](https://github.com/CompVis/latent-diffusion) environment by running

```
conda install pytorch torchvision -c pytorch
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -e .
```

## THE GUI IS OUT <br>

See [GUI Usage tutorial for dummies](GUI_TUTORIAL.md)

## The hlky's webui version [is out](https://github.com/neonsecret/stable-diffusion-webui)

_Note: Stable Diffusion v1 is a general text-to-image diffusion model and therefore mirrors biases and (mis-)conceptions that are present
in its training data.
Details on the training procedure and data, as well as the intended use of the model can be found in the corresponding [model card](Stable_Diffusion_v1_Model_Card.md)._

The weights are available via [the CompVis organization at Hugging Face](https://huggingface.co/CompVis) under [a license which contains specific use-based restrictions to prevent misuse and harm as informed by the model card, but otherwise remains permissive](LICENSE). While commercial use is permitted under the terms of the license, **we do not recommend using the provided weights for services or products without additional safety mechanisms and considerations**, since there are [known limitations and biases](Stable_Diffusion_v1_Model_Card.md#limitations-and-bias) of the weights, and research on safe and ethical deployment of general text-to-image models is an ongoing effort. **The weights are research artifacts and should be treated as such.**

[The CreativeML OpenRAIL M license](LICENSE) is an [Open RAIL M license](https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses), adapted from the work that [BigScience](https://bigscience.huggingface.co/) and [the RAIL Initiative](https://www.licenses.ai/) are jointly carrying in the area of responsible AI licensing. See also [the article about the BLOOM Open RAIL license](https://bigscience.huggingface.co/blog/the-bigscience-rail-license) on which our license is based.

# The project is completely open-source and free, and is only maintained by me. If you want to support me, I have a [ko-fi](https://ko-fi.com/neonsecret)

We currently provide the following checkpoints:

- `sd-v1-1.ckpt`: 237k steps at resolution `256x256` on [laion2B-en](https://huggingface.co/datasets/laion/laion2B-en).
  194k steps at resolution `512x512` on [laion-high-resolution](https://huggingface.co/datasets/laion/laion-high-resolution) (170M examples from LAION-5B with resolution `>= 1024x1024`).
- `sd-v1-2.ckpt`: Resumed from `sd-v1-1.ckpt`.
  515k steps at resolution `512x512` on [laion-aesthetics v2 5+](https://laion.ai/blog/laion-aesthetics/) (a subset of laion2B-en with estimated aesthetics score `> 5.0`, and additionally
  filtered to images with an original size `>= 512x512`, and an estimated watermark probability `< 0.5`. The watermark estimate is from the [LAION-5B](https://laion.ai/blog/laion-5b/) metadata, the aesthetics score is estimated using the [LAION-Aesthetics Predictor V2](https://github.com/christophschuhmann/improved-aesthetic-predictor)).
- `sd-v1-3.ckpt`: Resumed from `sd-v1-2.ckpt`. 195k steps at resolution `512x512` on "laion-aesthetics v2 5+" and 10\% dropping of the text-conditioning to improve [classifier-free guidance sampling](https://arxiv.org/abs/2207.12598).
- `sd-v1-4.ckpt`: Resumed from `sd-v1-2.ckpt`. 225k steps at resolution `512x512` on "laion-aesthetics v2 5+" and 10\% dropping of the text-conditioning to improve [classifier-free guidance sampling](https://arxiv.org/abs/2207.12598).

## The superfast and low-vram mode have been updated. The latest results are: 2048x2048 on 8 gb vram and 3200x3200 on 24 gb.

Below you can see the speed/resolution comparison table.
<br>
| resolution | steps | soft_limiter | time | vram | low vram mode |
|------------ |------- |---------- |--------------- |------ |--------------- |
| 512x512 | 50 | 100% | 1.5 minutes | 4 | no |
| 512x512 | 50 | 100% | 36 seconds | 8 | no |
| 512x512 | 50 | 100% | 30 seconds | 10 | no |
| 512x512 | 50 | 100% | 15 seconds | 24 | no |
| 1024x1024 | 50 | 100% | 15 minutes | 4 | no |
| 1024x1024 | 50 | 100% | 4 minutes | 8 | no |
| 1024x1024 | 50 | 100% | 3 minutes | 10 | no |
| 1024x1024 | 50 | 100% | 70 seconds | 24 | no |
| 2048x2048 | 50 | 100% | 25 minutes | 8 | no |
| 2048x2048 | 50 | 100% | 20 minutes | 10 | no |
| 2048x2048 | 50 | 100% | 15 minutes | 24 | no |
| 512x4096 | 50 | 100% | 2 minutes | 24 | no |
| 3840x2176 | 50 | 100% | 40 minutes | 24 | no |
| 3200x3200 | 50 | 100% | 60 minutes | 24 | no |
<br>

gpus used: gtx 1050 ti, rtx 3070, colab gpu, rtx 3090
(huge thanks to @therustysmear for helping me in these tests)

soft_limiter parameter limits vram usage, so that you can use your pc while generating images. 100% though allows the max sped.

Stable Diffusion is a latent diffusion model conditioned on the (non-pooled) text embeddings of a CLIP ViT-L/14 text encoder.
We provide a [reference script for sampling](#reference-sampling-script), but
there also exists a [diffusers integration](#diffusers-integration), which we
expect to see more active community development.

#### Reference Sampling Script

We provide a reference sampling script, which incorporates

- a [Safety Checker Module](https://github.com/CompVis/stable-diffusion/pull/36),
  to reduce the probability of explicit outputs,
- an [invisible watermarking](https://github.com/ShieldMnt/invisible-watermark)
  of the outputs, to help viewers [identify the images as machine-generated](scripts/tests/test_watermark.py).

After [obtaining the `stable-diffusion-v1-*-original` weights](#weights), link them

```
python optimizedSD/optimized_txt2img.py --prompt "an apple" --config_path optimizedSD/v1-inference_lowvram.yaml --H 512 --W 512 --seed 27 --n_iter 1 --n_samples 1 --ddim_steps 50
```

Example gradio command:

```
python optimizedSD/neongradio_ultimate.py
```

By default, this uses a guidance scale of `--scale 7.5`, [Katherine Crowson's implementation](https://github.com/CompVis/latent-diffusion/pull/51) of the [PLMS](https://arxiv.org/abs/2202.09778) sampler,
and renders images of size 512x512 (which it was trained on) in 50 steps. All supported arguments are listed below (type `python scripts/txt2img.py --help`).

```commandline
usage: txt2img.py [-h] [--prompt [PROMPT]] [--outdir [OUTDIR]] [--skip_grid] [--skip_save] [--ddim_steps DDIM_STEPS] [--plms] [--laion400m] [--fixed_code] [--ddim_eta DDIM_ETA]
                  [--n_iter N_ITER] [--H H] [--W W] [--C C] [--f F] [--n_samples N_SAMPLES] [--n_rows N_ROWS] [--scale SCALE] [--from-file FROM_FILE] [--config CONFIG] [--ckpt CKPT]
                  [--seed SEED] [--precision {full,autocast}]

optional arguments:
  -h, --help            show this help message and exit
  --prompt [PROMPT]     the prompt to render
  --outdir [OUTDIR]     dir to write results to
  --skip_grid           do not save a grid, only individual samples. Helpful when evaluating lots of samples
  --skip_save           do not save individual samples. For speed measurements.
  --ddim_steps DDIM_STEPS
                        number of ddim sampling steps
  --plms                use plms sampling
  --laion400m           uses the LAION400M model
  --fixed_code          if enabled, uses the same starting code across samples
  --ddim_eta DDIM_ETA   ddim eta (eta=0.0 corresponds to deterministic sampling
  --n_iter N_ITER       sample this often
  --H H                 image height, in pixel space
  --W W                 image width, in pixel space
  --C C                 latent channels
  --f F                 downsampling factor
  --n_samples N_SAMPLES
                        how many samples to produce for each given prompt. A.k.a. batch size
  --n_rows N_ROWS       rows in the grid (default: n_samples)
  --scale SCALE         unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
  --from-file FROM_FILE
                        if specified, load prompts from this file
  --config CONFIG       path to config which constructs model
  --ckpt CKPT           path to checkpoint of model
  --seed SEED           the seed (for reproducible sampling)
  --precision {full,autocast}
                        evaluate at this precision
```

python optimizedSD/neongradio_ultimate.py --config_path optimizedSD/v1-inference_lowvram.yaml

````
the `--config_path optimizedSD/v1-inference_lowvram.yaml` argument enables a low-vram mode which allows to generate bigger-resolution images at the slight cost of the speed.

### Description

#### Diffusers Integration

A simple way to download and sample Stable Diffusion is by using the [diffusers library](https://github.com/huggingface/diffusers/tree/main#new--stable-diffusion-is-now-fully-compatible-with-diffusers):
```py
# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4",
	use_auth_token=True
).to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt)["sample"][0]

image.save("astronaut_rides_horse.png")
````

### Image Modification with Stable Diffusion

To achieve this, the stable diffusion model is fragmented into four parts which are sent to the GPU only when needed.
After the calculation is done, they are moved back to the CPU. This allows us to run a bigger model while requiring less
VRAM.

Also I invented the sliced atttention technique, which allows to push the model's abilities even further. It works by automatically determining the slice size from your vram and image size and then allocating it one by one accordingly.
You can practically generate any image size, it just depends on the generation speed you are willing to sacrifice.

<h1 align="center">Installation</h1>

You can clone this repo and follow the same installation steps as the original (mainly creating the conda environment and
placing the weights at the specified location). <br>
So run: <br>
`conda env create -f environment.yaml` <br>
`conda activate ldm`

<h2 align="center">Additional steps for AMD Cards</h2>

After activating your conda environment, you have to update torch and torchvision wheels which were built with ROCm support (only on linux):

`pip3 install --upgrade torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.1.1`

<h2 align="center">Docker</h2>

Alternatively, if you prefer to use Docker, you can do the following:

1. Install [Docker](https://docs.docker.com/engine/install/)
   , [Docker Compose plugin](https://docs.docker.com/compose/install/),
   and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
2. Clone this repo to, e.g., `~/stable-diffusion`
3. Put your downloaded `model.ckpt` file into `~/sd-data` (it's a relative path, you can change it
   in `docker-compose.yml`)
4. `cd` into `~/stable-diffusion` and execute `docker compose up --build`

This will launch gradio on port 7860 with txt2img. You can also use `docker compose run` to execute other Python
scripts.

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models},
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

- Prompts can also be weighted to put relative emphasis on certain words.
  eg. `--prompt tabby cat:0.25 white duck:0.75 hybrid`.

- The number followed by the colon represents the weight given to the words before the colon. The weights can be both
  fractions or integers.
