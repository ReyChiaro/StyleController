import os
import hydra
import torch
import numpy as np

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from diffusers import FluxKontextPipeline
from diffusers.pipelines.flux.pipeline_flux_kontext import calculate_shift, retrieve_timesteps
from models.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image

from tuner import FluxKontextFinetuner, QwenImageEditFinetuner
from utils import (
    neighbor_interpolate_states,
    bspline_interpolate_states,
    lora_scale_states,
)


@torch.no_grad()
def flux_styctrl_infer(
    pipeline: FluxKontextPipeline,
    content_image: Image.Image,
    style_image: Image.Image,
    prompt: str,
    joint_attention_kwargs: dict,
    num_inference_steps: int,
    generator: torch.Generator,
    height: int = 1024,
    width: int = 1024,
    guidance_scale: float = 3.5,
    max_sequence_length: int = 512,
) -> Image.Image:
    device = pipeline._execution_device
    dtype = pipeline.transformer.dtype
    batch_size = 1
    num_images_per_prompt = 1

    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    multiple_of = pipeline.vae_scale_factor * 2
    height = height // multiple_of * multiple_of
    width = width // multiple_of * multiple_of

    num_channels_latents = pipeline.transformer.config.in_channels // 4
    latents, _, latent_ids, _ = pipeline.prepare_latents(
        None,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        None,
    )

    content_tensor = pipeline.image_processor.preprocess(content_image, height=height, width=width)
    style_tensor = pipeline.image_processor.preprocess(style_image, height=height, width=width)
    content_tensor = content_tensor.to(device=device, dtype=prompt_embeds.dtype)
    style_tensor = style_tensor.to(device=device, dtype=prompt_embeds.dtype)

    content_latents = pipeline._encode_vae_image(content_tensor, generator=generator)
    style_latents = pipeline._encode_vae_image(style_tensor, generator=generator)
    content_latent_height, content_latent_width = content_latents.shape[2:]
    style_latent_height, style_latent_width = style_latents.shape[2:]

    content_latents = pipeline._pack_latents(
        content_latents,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        content_latent_height,
        content_latent_width,
    )
    style_latents = pipeline._pack_latents(
        style_latents,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        style_latent_height,
        style_latent_width,
    )
    image_latents = torch.cat([content_latents, style_latents], dim=1)

    content_ids = pipeline._prepare_latent_image_ids(
        batch_size * num_images_per_prompt,
        content_latent_height // 2,
        content_latent_width // 2,
        device,
        prompt_embeds.dtype,
    )
    style_ids = pipeline._prepare_latent_image_ids(
        batch_size * num_images_per_prompt,
        style_latent_height // 2,
        style_latent_width // 2,
        device,
        prompt_embeds.dtype,
    )
    content_ids[..., 0] = 1
    style_ids[..., 0] = 2
    img_ids = torch.cat([latent_ids, content_ids, style_ids], dim=0)

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(
        latents.shape[1],
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    pipeline.scheduler.set_begin_index(0)

    guidance = None
    if pipeline.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0])

    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for t in timesteps:
            latent_model_input = torch.cat([latents, image_latents], dim=1)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            noise_pred = pipeline.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : latents.size(1)]
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            progress_bar.update()

    latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    image = pipeline.vae.decode(latents, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(image, output_type="pil")[0]
    pipeline.maybe_free_model_hooks()
    return image


@hydra.main(version_base="v1.2", config_path="configs", config_name="default")
def inference(cfgs: OmegaConf):

    # -- Initialize Training Framework -- #
    accelerator = Accelerator(
        mixed_precision=cfgs.mixed_precision,
        gradient_accumulation_steps=cfgs.gradient_accumulation,
        log_with=cfgs.log_with,
        project_config=ProjectConfiguration(
            project_dir=cfgs.project_dir,
            logging_dir=cfgs.logging_dir,
        ),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(cfgs.project_name)

    logger = get_logger(__name__, log_level="INFO")
    logger.info(f"# -- Configs -- #")
    logger.info(OmegaConf.to_yaml(cfgs))

    # -- Initialize Finetuner -- #
    finetuner: QwenImageEditFinetuner = instantiate(cfgs.finetuner, device=accelerator.device, logger=logger)

    # -- Add Single LoRA -- #
    if cfgs.load_lora:
        finetuner.register_styctrl(
            lora_target_modules=cfgs.styctrl.lora_target_modules,
            device=accelerator.device,
            dtype=finetuner.dtype,
        )
        finetuner.add_styctrl(
            rank=cfgs.styctrl.rank,
            adapter_name=cfgs.styctrl.adapter_name,
            proj_type=cfgs.styctrl.proj_type,
            bias=cfgs.styctrl.bias,
            checkpoint_path=cfgs.styctrl.checkpoint_path,
            device=accelerator.device,
            dtype=finetuner.dtype,
        )
        finetuner.activate_styctrl(cfgs.styctrl.adapter_name)

    # -- Inference -- #
    if isinstance(finetuner, FluxKontextFinetuner):
        pipeline = FluxKontextPipeline(
            transformer=finetuner.transformer,
            vae=finetuner.vae,
            scheduler=finetuner.scheduler_val,
            text_encoder=finetuner.text_encoder,
            tokenizer=finetuner.tokenizer,
            text_encoder_2=finetuner.text_encoder_2,
            tokenizer_2=finetuner.tokenizer_2,
        )
    else:
        pipeline = QwenImageEditPlusPipeline(
            transformer=finetuner.transformer,
            vae=finetuner.vae,
            scheduler=finetuner.scheduler_val,
            text_encoder=finetuner.text_encoder,
            tokenizer=finetuner.tokenizer,
            processor=finetuner.processor,
        )

    interp_type = cfgs.inference.interp_type
    anchor_states = cfgs.inference.anchor_states
    output_dir = cfgs.inference.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ----------- Main Papaer Inference ----------- #
    infer_pairs = os.listdir(cfgs.valset.result_dir)
    infer_pairs = list(set(["_".join(p.split("_")[:2]) for p in infer_pairs]))
    infer_pairs.sort()

    max_infer_num = min(cfgs.inference.max_infer_num, len(infer_pairs))
    infer_start = cfgs.inference.infer_start

    infer_pairs = [(p.split("_")[0], p.split("_")[1]) for p in infer_pairs][infer_start:max_infer_num]
    infer_pairs = [
        (os.path.join(cfgs.valset.content_dir, cn) + ".jpg", os.path.join(cfgs.valset.style_dir, sn) + ".jpg")
        for (cn, sn) in infer_pairs
    ]
    logger.info(f"Total infer: {max_infer_num-infer_start}")
    # ------------------- END ------------------- #

    # ----------- Supp Inference ----------- #
    # with open("/temp/xr/ECCV26/styctrl/figs/inference/content_style.txt") as f:
    #     infer_pairs = f.read().split("\n")[:-1]
    # infer_pairs = [(p.split()[0], p.split()[1]) for p in infer_pairs][::5]
    # logger.info(f"Total infer: {len(infer_pairs)}")
    # ------------------- END ------------------- #


    if cfgs.inference.continuous:
        query_points = np.linspace(0, 1, 10, endpoint=True)
    else:
        query_points = [cfgs.inference.query_point]

    for cnt_path, sty_path in infer_pairs:
        cnt_id = Path(cnt_path).stem
        sty_id = Path(sty_path).stem
        save_dir = os.path.join(output_dir, f"{cnt_id}_{sty_id}")
        if cfgs.inference.continuous:
            os.makedirs(save_dir, exist_ok=True)

        cnt_img = Image.open(cnt_path).convert("RGB").resize([1024, 1024])
        sty_img = Image.open(sty_path).convert("RGB").resize([1024, 1024])

        output_list = []
        for i, query_point in enumerate(query_points):
            interp_states = {}
            if interp_type == "neighbor":
                interp_states = neighbor_interpolate_states(
                    anchor_states,
                    query_point,
                    accelerator.device,
                    torch.bfloat16,
                )
            elif interp_type == "bspline":
                interp_states = bspline_interpolate_states(
                    anchor_states,
                    query_point,
                    X=cfgs.inference.X,
                    order=cfgs.inference.k,
                    device=accelerator.device,
                    dtype=torch.bfloat16,
                )
            elif interp_type == "lora_scale":
                interp_states = lora_scale_states(
                    anchor_states,
                    query_point,
                    device=accelerator.device,
                    dtype=torch.bfloat16,
                )
            else:
                logger.info(f"No interpolated state is loaded.")
                pass
                # raise KeyError(f"Cannot load interpolated states with {interp_type=}")
            pipeline.transformer.load_state_dict(interp_states, strict=False)
            for _,c in pipeline.components.items():
                if hasattr(c, "parameters"):
                    for p in c.parameters():
                        p.requires_grad_(False)

            w = torch.tensor([query_point], device=accelerator.device, dtype=torch.bfloat16)
            if cfgs.load_lora:
                attention_kwargs = {
                    "lora_layer_indices": finetuner.lora_layer_indices,
                    "enable_lora": True,
                    "w": w,
                }
            else:
                attention_kwargs = {}
            if isinstance(finetuner, FluxKontextFinetuner):
                output = flux_styctrl_infer(
                    pipeline=pipeline,
                    content_image=cnt_img,
                    style_image=sty_img,
                    prompt=finetuner.DEFAULT_PROMPT,
                    joint_attention_kwargs=attention_kwargs,
                    num_inference_steps=16,
                    generator=torch.Generator(device=accelerator.device).manual_seed(42),
                    height=1024,
                    width=1024,
                )
            else:
                output: Image.Image = pipeline(
                    image=[cnt_img, sty_img],
                    prompt=finetuner.DEFAULT_PROMPT,
                    attention_kwargs=attention_kwargs,
                    num_inference_steps=16,
                    generator=torch.Generator(device=accelerator.device).manual_seed(42),
                ).images[0]

            if cfgs.inference.continuous:
                save_path = os.path.join(save_dir, f"{i:02d}.jpg")
                output.save(save_path)
                output_list.append(output)
                logger.info(f"C {cnt_id}, S {sty_id} [{i+1:02d}/{len(query_points)}] Save to {save_path}")
            else:
                output.save(save_dir + ".jpg")
                logger.info(f"C {cnt_id}, S {sty_id} [{i+1:02d}/{len(query_points)}] Save to {save_dir}.jpg")

        if cfgs.inference.continuous:
            output_list[0].save(
                f"{save_dir}/output.gif",
                save_all=True,
                append_images=output_list[1:],  # Append all images after the first one
                duration=100,
                loop=0,
            )
            logger.info(f"Content {cnt_id}, Style {sty_id} GIF Save to {save_dir}/output.gif")


if __name__ == "__main__":
    inference()
