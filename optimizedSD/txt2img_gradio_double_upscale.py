import argparse
import os
import re
import time
from contextlib import nullcontext
from itertools import islice
from random import randint

import gradio as gr
import numpy as np
import torch
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from transformers import logging
import mimetypes
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger

logging.set_verbosity_error()

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def load_img(image, h0, w0):
    image = image.convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def generate(
        prompt,
        ddim_steps,
        img2img_strength,
        Height,
        Width,
        scale,
        ddim_eta,
        unet_bs,
        device,
        seed,
        outdir,
        img_format,
        turbo,
        full_precision,
        sampler,
):
    C = 4
    f = 8
    start_code = None
    model.unet_bs = unet_bs
    model.turbo = turbo
    model.cdevice = device
    modelCS.cond_stage_model.device = device

    if seed == "":
        seed = randint(0, 1000000)
    seed = int(seed)
    seed_everything(seed)
    # Logging
    logger(locals(), "logs/doulbe_upscale.csv")

    if device != "cpu" and not full_precision:
        model.half()
        modelFS.half()
        modelCS.half()

    tic = time.time()
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir
    sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompt.replace("/", ""))))[:150]
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    assert prompt is not None
    data = [1 * [prompt]]

    if device != "cpu" and not full_precision:
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    all_samples = []
    seeds = ""
    with torch.no_grad():
        all_samples = list()
        for prompts in tqdm(data, desc="data"):
            with precision_scope("cuda"):
                modelCS.to(device)
                uc = None
                if scale != 1.0:
                    uc = modelCS.get_learned_conditioning(1 * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)

                subprompts, weights = split_weighted_subprompts(prompts[0])
                if len(subprompts) > 1:
                    c = torch.zeros_like(uc)
                    totalWeight = sum(weights)
                    # normalize each "sub prompt" and add it
                    for i in range(len(subprompts)):
                        weight = weights[i]
                        # if not skip_normalize:
                        weight = weight / totalWeight
                        c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                else:
                    c = modelCS.get_learned_conditioning(prompts)

                shape = [1, C, Height // f, Width // f]

                if device != "cpu":
                    mem = torch.cuda.memory_allocated() / 1e6
                    modelCS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)

                samples_ddim = model.sample(
                    S=ddim_steps,
                    conditioning=c,
                    seed=seed,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    eta=ddim_eta,
                    x_T=start_code,
                    sampler=sampler,
                )

                modelFS.to(device)

                x_samples_ddim = modelFS.decode_first_stage(samples_ddim[0].unsqueeze(0))
                x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                Image.fromarray(x_sample.astype(np.uint8)).save(
                    os.path.join(sample_path, "seed_" + str(seed) + "_step1_" + f"{base_count:05}.{img_format}")
                )

                ### STEP 2

                init_image = repeat(load_img(Image.fromarray(x_sample.astype(np.uint8)), Height*2, Width*2).to(device),
                                    "1 ... -> b ...", b=1)
                init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))

                modelFS.cpu()
                model.to(device)

                z_enc = model.stochastic_encode(
                    init_latent, torch.tensor([int(img2img_strength * ddim_steps)]).to(device), seed, ddim_eta,
                    ddim_steps
                ).to(device)

                samples_ddim = model.sample(
                    int(img2img_strength * ddim_steps // 3),
                    c,
                    z_enc,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    sampler="ddim"
                )

                modelFS.to(device)
                model.cpu()
                modelCS.to("cpu")
                torch.cuda.empty_cache()

                x_samples_ddim = modelFS.decode_first_stage(samples_ddim[0].unsqueeze(0))
                x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                Image.fromarray(x_sample.astype(np.uint8)).save(
                    os.path.join(sample_path, "seed_" + str(seed) + "_step2_" + f"{base_count:05}.{img_format}")
                )

                ### STEP 3

                init_image = repeat(
                    load_img(Image.fromarray(x_sample.astype(np.uint8)), Height * 3, Width * 3).to(device),
                    "1 ... -> b ...", b=1)
                init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))

                modelFS.cpu()
                model.to(device)

                z_enc = model.stochastic_encode(
                    init_latent, torch.tensor([int(img2img_strength * ddim_steps)]).to(device), seed, ddim_eta,
                    ddim_steps
                ).to(device)

                samples_ddim = model.sample(
                    int(img2img_strength * ddim_steps // 3),
                    c,
                    z_enc,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    sampler="ddim"
                )

                print("saving images")
                model.cpu()
                modelFS.to(device)

                x_samples_ddim = modelFS.decode_first_stage(samples_ddim[0].unsqueeze(0))
                x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                all_samples.append(x_sample.to("cpu"))
                x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                Image.fromarray(x_sample.astype(np.uint8)).save(
                    os.path.join(sample_path, "seed_" + str(seed) + "_step3_" + f"{base_count:05}.{img_format}")
                )

                if device != "cpu":
                    mem = torch.cuda.memory_allocated() / 1e6
                    modelFS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)

                del samples_ddim
                del x_sample
                del x_samples_ddim
                print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    toc = time.time()

    time_taken = (toc - tic) / 60.0
    grid = torch.cat(all_samples, 0)
    grid = make_grid(grid, nrow=1)
    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()

    txt = (
            "Samples finished in "
            + str(round(time_taken, 3))
            + " minutes and exported to "
            + sample_path
            + "\nSeeds used = "
            + seeds[:-1]
    )
    return Image.fromarray(grid.astype(np.uint8)), txt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='txt2img using gradio')
    parser.add_argument('--config_path', default="optimizedSD/v1-inference.yaml", type=str, help='config path')
    parser.add_argument('--ckpt_path', default="models/ldm/stable-diffusion-v1/model.ckpt", type=str, help='ckpt path')
    parser.add_argument('--outputs_path', default="outputs/txt2img-samples", type=str, help='output imgs path')
    args = parser.parse_args()
    config = args.config_path
    ckpt = args.ckpt_path
    sd = load_model_from_config(f"{ckpt}")
    li, lo = [], []
    for key, v_ in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{config}")

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd

    demo = gr.Interface(
        fn=generate,
        inputs=[
            "text",
            gr.Slider(1, 1000, value=50),
            gr.Slider(0, 1, value=0.1, step=0.01),
            gr.Slider(64, 4096, value=512, step=64),
            gr.Slider(64, 4096, value=512, step=64),
            gr.Slider(0, 50, value=7.5, step=0.1),
            gr.Slider(0, 1, step=0.01),
            gr.Slider(1, 2, value=1, step=1),
            gr.Radio(["cuda", "cpu"], value="cuda"),
            "text",
            gr.Text(value=args.outputs_path),
            gr.Radio(["png", "jpg"], value='png'),
            gr.Checkbox(value=True),
            "checkbox",
            gr.Radio(["ddim", "plms"], value="plms"),
        ],
        outputs=["image", "text"],
    )
    demo.launch(share=True)
