import copy
import json
import logging
import math
import os
import prodigyopt
import torch
import torchvision.transforms.functional as T
import numpy as np
from accelerate import Accelerator
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from typing import Literal, Optional

from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.image_processor import VaeImageProcessor
from diffusers import FluxKontextPipeline
from diffusers.pipelines.flux.pipeline_flux_kontext import calculate_shift, retrieve_timesteps
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from transformers.models.qwen2 import Qwen2Tokenizer

from models.styctrl import (
    register_styctrl,
    add_styctrl,
    activate,
    deactivate,
)
from models.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from models.transformer_flux_styctrl import FluxStyCtrlTransformer2DModel
from models.transformer_qwenimage import QwenImageTransformer2DModel
from utils import summarize_model


def calculate_dimensions(
    target_image_size: int,
    wh_ratio: float,
    multiple_of: int,
) -> tuple[int]:
    w = math.sqrt(target_image_size * wh_ratio)
    h = w / wh_ratio

    w = int(w // multiple_of * multiple_of)
    h = int(h // multiple_of * multiple_of)

    return h, w


def retrieve_latents(
    vae_output: AutoencoderKLOutput,
    generator: torch.Generator,
    sample_mode: str = "sample",
) -> torch.Tensor:
    if hasattr(vae_output, "latent_dist") and sample_mode == "sample":
        return vae_output.latent_dist.sample(generator)
    elif hasattr(vae_output, "latent_dist") and sample_mode == "argmax":
        return vae_output.latent_dist.mode()
    elif hasattr(vae_output, "latents"):
        return vae_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class QwenImageEditFinetuner:

    CONDITION_IMAGE_SIZE = 384 * 384
    VAE_IMAGE_SIZE = 512 * 512
    DEFAULT_PROMPT = (
        "Apply the visual style of the second reference image to the first content image "
        "while preserving the content structure and layout."
    )
    # VAE_IMAGE_SIZE = 1024 * 1024

    def __init__(
        self,
        pretrained_pipeline: str,
        lora_layer_range: list[int] = [0, 59],
        load_dit: bool = True,
        load_text_encoder: bool = False,
        timestep_weighting_scheme: str = "logit_normal",
        lr: float = 1.0,
        max_grad_norm: float = 1.0,
        mu_logit_mean: float = 0.0,
        mu_logit_std: float = 1.0,
        mu_mode_scale: float = 1.29,
        modules_require_grad: list[Literal["lora_A", "lora_B", "projector"]] = [
            "lora_A",
            "lora_B",
        ],
        torch_dtype: str = "bf16",
        device: str | int = "cpu",
        logger: logging.Logger = None,
    ):
        r"""
        Args:
            @param pretrained_pipeline:

            # LoRA configs
            @param lora_rank:
            @param lora_target_modules:
            @param lora_layer_range: List of length 2 (default [0, 59])
            @param lora_checkpoint:

            # Modules
            @param load_dit:
            @param modules_require_grad: lora_A | lora_B | in_weight_proj | mid_weight_proj | out_weight_proj
            @param load_text_encoder: in_weight | mid_weight | out_weight

            # Training
            @param lr:
            @param max_grad_norm:

            # Sampling parameters
            @param timestep_weighting_scheme:
            @param mu_logit_mean:
            @param mu_logit_std:
            @param mu_mode_scale:

            # Other args
            @param device: Pipeline and model device (default cpu)
            @param torch_dtype: Pipeline and weights dtype (default bf16)
            @param logger: DDP logger (default None)
        """
        self.device = device
        self.dtype = torch.float32
        if torch_dtype == "fp16":
            self.dtype = torch.float16
        elif torch_dtype == "bf16":
            self.dtype = torch.bfloat16

        self.load_dit = load_dit
        self.load_text_encoder = load_text_encoder

        self.modules_require_grad = modules_require_grad
        self.lora_layer_indices = list(range(lora_layer_range[0], lora_layer_range[1] + 1))
        self.lr = lr
        self.max_grad_norm = max_grad_norm

        self.timestep_weighting_scheme = timestep_weighting_scheme
        self.mu_logit_mean = mu_logit_mean
        self.mu_logit_std = mu_logit_std
        self.mu_mode_scale = mu_mode_scale

        self.logger = logger or logging.getLogger(__name__)

        # -- Copied from QwenImageEditPlusPipeline
        self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 64
        self.default_sample_size = 128
        # -- End copy

        # -- Initialize modules -- #
        if self.load_dit:
            self.logger.info(f"# -- Initialize Model -- #")
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                pretrained_pipeline,
                subfolder="scheduler",
                torch_dtype=self.dtype,
            )
            # Scheduler for training and evaluation donot share timesteps
            self.scheduler_val = copy.deepcopy(self.scheduler)
            self.logger.info(f"Scheduler initialized")
            self.vae = (
                AutoencoderKLQwenImage.from_pretrained(
                    pretrained_pipeline,
                    subfolder="vae",
                    torch_dtype=self.dtype,
                )
                .to(device)
                .requires_grad_(False)
            )
            self.logger.info(f"VAE initialized")

            self.transformer = (
                QwenImageTransformer2DModel.from_pretrained(
                    pretrained_pipeline,
                    subfolder="transformer",
                    torch_dtype=self.dtype,
                )
                .to(device)
                .requires_grad_(False)
            )
            self.logger.info(f"Transformer initialized")

            # -- Initialize useful parameters
            self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)  # 8
            self.latent_channels = self.vae.config.z_dim
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        if self.load_text_encoder:
            self.tokenizer = Qwen2Tokenizer.from_pretrained(
                pretrained_pipeline,
                subfolder="tokenizer",
                torch_dtype=self.dtype,
            )
            self.logger.info(f"Tokenizer initialized")
            self.text_encoder = (
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    pretrained_pipeline,
                    subfolder="text_encoder",
                    torch_dtype=self.dtype,
                )
                .to(device)
                .requires_grad_(False)
            )
            self.logger.info(f"Text Encoder initialized")
            self.processor = Qwen2_5_VLProcessor.from_pretrained(
                pretrained_pipeline,
                subfolder="processor",
                torch_dtype=self.dtype,
            )
            self.logger.info(f"Processor initialized")
        self.logger.info(f"# -- Model initialized -- #")

        self.active_adapter = None

    def register_styctrl(
        self,
        lora_target_modules: list[str],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        if not self.load_dit:
            self.logger.warning(f"DiT has not been loaded.")
            return
        register_styctrl(
            model=self.transformer,
            target_modules=lora_target_modules,
            device=device,
            dtype=dtype,
        )

    def add_styctrl(
        self,
        rank: int,
        adapter_name: str = "default",
        proj_type: Literal["scale", "linear", "none"] = "none",
        bias: bool = False,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Add a LoRA module into DiT. Just add a module with parameters and adapter_name into ParameterDict or ModuleDict in LoRA Adapter, do not enable training, no gradient is required by default. If checkpoint_path is provided, then load it into current LoRA module.

        Args:
            @param checkpoint_path (str): Expected to have key format `transformer_blocks.0.attn.to_q.lora_A.adapter_name`
        """
        device = device or self.device
        dtype = dtype or self.dtype

        add_styctrl(
            model=self.transformer,
            rank=rank,
            lora_layer_indices=self.lora_layer_indices,
            adapter_name=adapter_name,
            proj_type=proj_type,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            styctrl_states = load_file(checkpoint_path, device="cpu")
            self.transformer.load_state_dict(styctrl_states, strict=False)
            self.logger.info(f"Checkpoint loading finished.")

        for n, p in self.transformer.named_parameters():
            if adapter_name in n:
                p = p.to(device=device, dtype=dtype)
                p.requires_grad_(False)

        self.logger.info(f"LoRA {adapter_name} has been added into DiT.")

    def activate_styctrl(self, adapter_name: str):
        self.active_adapter = adapter_name
        activate(self.transformer, adapter_name)

    def deactivate_styctrl(self):
        self.active_adapter = None
        deactivate(self.transformer)

    def enable_training(
        self,
        adapter_name: str = "default",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype
        for n, p in self.transformer.named_parameters():
            if adapter_name in n and any(m in n for m in self.modules_require_grad):
                p = p.to(device=device, dtype=dtype)
                p.requires_grad_(True)

        dit_info = summarize_model(self.transformer)
        self.logger.info(f"DiT StyCtrlLoRA Summary")
        for k, v in dit_info.items():
            self.logger.info(f"{k}: {v}")

    def disable_training(
        self,
        adapter_name: str = "default",
    ):
        for n, p in self.transformer.named_parameters():
            if adapter_name in n:
                p.requires_grad_(False)

    def save_styctrl_checkpoint(
        self,
        accelerator: Accelerator,
        output_dir: str,
        global_step: int,
        save_modules: list[Literal["lora_A", "lora_B", "projector"]],
    ):
        if self.active_adapter is None:
            self.logger.warning(f"Current active lora is None, no checkpoint will be saved.")
            return
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(checkpoint_path, exist_ok=True)

        lora_state_dict = {}
        for n, p in accelerator.unwrap_model(self.transformer).named_parameters():
            if self.active_adapter in n and any(m in n for m in save_modules):
                lora_state_dict[n] = p
        save_path = os.path.join(checkpoint_path, f"styctrl_lora_{self.active_adapter}.safetensors")
        save_file(lora_state_dict, save_path)
        self.logger.info(f"Save StyCtrlLoRA {self.active_adapter} to {save_path}")

    def get_sigmas(self, timesteps: torch.Tensor, latent_ndim: int = 4) -> torch.Tensor:
        device = timesteps.device
        sched_sigmas = self.scheduler.sigmas.to(device)
        sched_timesteps = self.scheduler.timesteps.to(device)
        indices = [(sched_timesteps == t).nonzero().item() for t in timesteps]
        sigmas = sched_sigmas[indices].flatten()
        while sigmas.ndim < latent_ndim:
            sigmas = sigmas.unsqueeze(-1)
        return sigmas

    def preprocess_images(
        self,
        target_size: int,
        cnt_images: torch.Tensor | list[torch.Tensor],
        sty_images: torch.Tensor | list[torch.Tensor],
        res_images: torch.Tensor | list[torch.Tensor],
        return_shapes: bool = False,
    ) -> tuple[torch.Tensor]:
        def _get_size(images: torch.Tensor) -> tuple[int]:
            h, w = images.shape[-2:]
            h, w = calculate_dimensions(target_size, w / h, multiple_of=self.vae_scale_factor * 2)
            return h, w

        if not isinstance(cnt_images, list):
            cnt_images = [cnt_images]
        if not isinstance(sty_images, list):
            sty_images = [sty_images]
        if not isinstance(res_images, list):
            res_images = [res_images]
        cnt_shapes = [_get_size(cnt) for cnt in cnt_images]
        sty_shapes = [_get_size(sty) for sty in sty_images]
        res_shapes = [_get_size(res) for res in res_images]

        processed_cnt = [
            self.image_processor.preprocess(cnt, height=shape[0], width=shape[1]).unsqueeze(2).to(dtype=self.dtype)
            for cnt, shape in zip(cnt_images, cnt_shapes)
        ]

        processed_sty = [
            self.image_processor.preprocess(sty, height=shape[0], width=shape[1]).unsqueeze(2).to(dtype=self.dtype)
            for sty, shape in zip(sty_images, sty_shapes)
        ]

        processed_res = [
            self.image_processor.preprocess(res, height=shape[0], width=shape[1]).unsqueeze(2).to(dtype=self.dtype)
            for res, shape in zip(res_images, res_shapes)
        ]

        processed_cnt = torch.cat(processed_cnt, dim=0)
        processed_sty = torch.cat(processed_sty, dim=0)
        processed_res = torch.cat(processed_res, dim=0)

        if return_shapes:
            return processed_cnt, processed_sty, processed_res, cnt_shapes, sty_shapes, res_shapes
        return processed_cnt, processed_sty, processed_res

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImageEditPipeline._extract_masked_hidden
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImageEditPipeline._get_qwen_prompt_embeds
    def encode_qwen_prompt(
        self,
        text_encoder: Optional[Qwen2_5_VLForConditionalGeneration],
        processor: Optional[Qwen2_5_VLProcessor],
        image: list[torch.Tensor],
        prompt: str | list[str] = "",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = dtype or text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        if isinstance(image, list):
            base_img_prompt = ""
            for i, img in enumerate(image):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif image is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        template = self.prompt_template_encode

        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]

        model_inputs = processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)

        outputs = text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.QwenImageEditPipeline.encode_prompt
    def encode_prompt(
        self,
        text_encoder: Optional[Qwen2_5_VLForConditionalGeneration],
        processor: Optional[Qwen2_5_VLProcessor],
        image: list[torch.Tensor],
        prompt: str | list[str] = "",
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 1024,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        txt_latents, txt_latent_masks = self.encode_qwen_prompt(text_encoder, processor, image, prompt, device)

        _, seq_len, _ = txt_latents.shape
        txt_latents = txt_latents.repeat(1, num_images_per_prompt, 1)
        txt_latents = txt_latents.view(batch_size * num_images_per_prompt, seq_len, -1)
        txt_latent_masks = txt_latent_masks.repeat(1, num_images_per_prompt, 1)
        txt_latent_masks = txt_latent_masks.view(batch_size * num_images_per_prompt, seq_len)

        return txt_latents, txt_latent_masks

    def encode_vae_images(self, images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        dtype = images.dtype
        image_latents = retrieve_latents(self.vae.encode(images), generator=generator, sample_mode="sample")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents.to(dtype=dtype)

    def pack_vae_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # Input is 5D tensor with frame dimension equals to 1
        B, C, _, H, W = latents.shape
        latents = latents.view(B, C, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(B, (H // 2 * W // 2), C * 4)
        return latents

    def unpack_vae_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        B, _, C = latents.shape
        latents = latents.view(B, height // 2, width // 2, C // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(B, C // 4, 1, height, width)
        return latents

    def prepare_vae_latents(
        self,
        images: torch.Tensor,
        pack_latent: bool,
        generator: torch.Generator,
    ) -> torch.Tensor:
        # 5D tensor
        # all_image_latents = []
        # for image in images:
        # The frame dimension is 1, C=16
        image_latents = self.encode_vae_images(images, generator)
        if pack_latent:
            # Output of image latent shape [B, L, C=64]
            image_latents = self.pack_vae_latents(image_latents)

        return image_latents

    @torch.no_grad()
    def evaluate(
        self,
        accelerator: Accelerator,
        val_loader: DataLoader,
        adapter_name: str,
        output_dir: str,
        global_step: int,
        max_num_val: int = 4,
        recover_train: bool = True,
    ):
        self.disable_training(adapter_name)
        pipeline = QwenImageEditPlusPipeline(
            transformer=accelerator.unwrap_model(self.transformer),
            vae=self.vae,
            scheduler=self.scheduler_val,
            text_encoder=self.text_encoder if self.load_text_encoder else None,
            tokenizer=self.tokenizer if self.load_text_encoder else None,
            processor=self.processor if self.load_text_encoder else None,
        )
        pipeline.to(dtype=torch.float32)

        val_save_dir = os.path.join(output_dir, f"eval-{global_step}")
        self.logger.info(f"{val_save_dir=}")
        if accelerator.is_main_process:
            os.makedirs(val_save_dir, exist_ok=True)

        # Only evaluate a few samples to save time
        info = {}
        for i, batch in enumerate(val_loader):
            if i >= max_num_val:
                break
            if len(batch) == 4:
                cnt_images, sty_images, res_images, strengths = batch
                strengths = strengths.to(dtype=self.dtype)
                txt_latents, txt_latent_masks = self.encode_prompt(
                    text_encoder=self.text_encoder,
                    processor=self.processor,
                    image=[cnt_images, sty_images],
                    prompt=self.DEFAULT_PROMPT,
                    device=self.device,
                )
            elif len(batch) == 6:
                cnt_images, sty_images, res_images, strengths, txt_latents, txt_latent_masks = batch
                strengths = strengths.to(dtype=self.dtype)
                txt_latents = txt_latents.to(dtype=self.dtype)
                txt_latent_masks = txt_latent_masks.to(dtype=torch.long)
            else:
                raise ValueError(f"Batch should contains only 4 or 6 items.")

            for j in range(cnt_images.shape[0]):
                w = strengths[j].unsqueeze(0)

                attention_kwargs = {
                    "lora_layer_indices": self.lora_layer_indices,
                    "enable_lora": True,
                    "w": w,
                }

                cnt_pil = T.to_pil_image(cnt_images[j])
                sty_pil = T.to_pil_image(sty_images[j])

                image = [cnt_pil, sty_pil]

                output = pipeline(
                    image=image,
                    prompt=None,
                    attention_kwargs=attention_kwargs,
                    num_inference_steps=16,
                    prompt_embeds=txt_latents[j].unsqueeze(0),
                    prompt_embeds_mask=txt_latent_masks[j].unsqueeze(0),
                    generator=torch.Generator(self.device).manual_seed(42),
                )

                gt_h, gt_w = res_images[j].shape[-2:]
                cnt = T.resize(cnt_images[j], [gt_h, gt_w])
                sty = T.resize(sty_images[j], [gt_h, gt_w])
                pred = T.resize(T.to_tensor(output.images[0]), [gt_h, gt_w])
                concat = torch.cat([cnt.cpu(), sty.cpu(), pred.cpu(), res_images[j].cpu()], dim=-1)

                if accelerator.is_main_process:
                    image_name = f"sample_{i}_{j}.jpg"
                    info[image_name] = {
                        "enable_lora": True,
                        "w": strengths[j].item(),
                    }
                    save_image(concat, os.path.join(val_save_dir, image_name))
        with open(os.path.join(val_save_dir, "eval_info.json"), "w") as f:
            json.dump(info, f, indent=4)

        if recover_train:
            pipeline.to(dtype=torch.bfloat16)
            self.enable_training(adapter_name)

    def _finetune_step(self, batch: tuple[torch.Tensor], generator: Optional[torch.Generator] = None):
        if len(batch) == 4:
            cnt_images, sty_images, res_images, strengths = batch
            cnt_images = cnt_images.to(dtype=self.dtype)
            sty_images = sty_images.to(dtype=self.dtype)
            res_images = res_images.to(dtype=self.dtype)
            strengths = strengths.to(dtype=self.dtype)
            txt_latents, txt_latent_masks = self.encode_prompt(
                text_encoder=self.text_encoder,
                processor=self.processor,
                image=[cnt_images, sty_images],
                prompt=self.DEFAULT_PROMPT,
                device=self.device,
            )
        elif len(batch) == 6:
            cnt_images, sty_images, res_images, strengths, txt_latents, txt_latent_masks = batch
            cnt_images = cnt_images.to(dtype=self.dtype)
            sty_images = sty_images.to(dtype=self.dtype)
            res_images = res_images.to(dtype=self.dtype)
            strengths = strengths.to(dtype=self.dtype)
            txt_latents = txt_latents.to(dtype=self.dtype)
            txt_latent_masks = txt_latent_masks.to(dtype=torch.long)
        else:
            raise ValueError(f"Batch should contains only 4 or 6 items.")

        # For QwenImage, the output images are list of 5D [B, C, F, H, W], where F stands for frames
        cnt_images, sty_images, res_images, cnt_shapes, sty_shapes, res_shapes = self.preprocess_images(
            self.VAE_IMAGE_SIZE,
            cnt_images,
            sty_images,
            res_images,
            return_shapes=True,
        )
        batch_size = cnt_images.shape[0]
        dtype = cnt_images.dtype
        device = cnt_images.device

        # Pack latent, shape [B, L, C]
        cnt_latents = self.prepare_vae_latents(cnt_images, True, generator)
        sty_latents = self.prepare_vae_latents(sty_images, True, generator)

        # No pack latent, shape [B, C, 1, H, W]
        res_latents = self.prepare_vae_latents(res_images, False, generator)

        # cnt_latents and sty_latents should be concatenated alternately
        img_latents = [torch.cat([cnt, sty], dim=0) for cnt, sty in zip(cnt_latents, sty_latents)]

        # Shape [B, 2 * L, C]
        img_latents = torch.stack(img_latents, dim=0)

        # Noises latents
        eps_latents = torch.randn_like(res_latents, dtype=dtype, device=device)

        # Sample timesteps (integer in [0, 1000]) and sigmas (float in [0, 1])
        u = compute_density_for_timestep_sampling(
            self.timestep_weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.mu_logit_mean,
            logit_std=self.mu_logit_std,
            mode_scale=self.mu_mode_scale,
            device=device,
            generator=generator,
        )
        indices = (u * self.scheduler.config.num_train_timesteps).long().to("cpu")
        timesteps = self.scheduler.timesteps[indices].to(device)
        sigmas = self.get_sigmas(timesteps, latent_ndim=res_latents.ndim).to(device, dtype=dtype)

        noisy_latents = (1.0 - sigmas) * res_latents + sigmas * eps_latents

        # Shape [B, L, C]
        noisy_latents = self.pack_vae_latents(noisy_latents)

        # Calculate image shapes for generating RoPE in transformer.
        # For each sample the image will be concatenated into [latent, img_latent], where latent shape comes from cnt_shapes, and img_latent shape comes from cnt_shapes and sty_shapes
        image_shapes = []
        for (res_h, res_w), (cnt_h, cnt_w), (sty_h, sty_w) in zip(res_shapes, cnt_shapes, sty_shapes):
            image_shapes.append(
                [
                    (1, res_h // self.vae_scale_factor // 2, res_w // self.vae_scale_factor // 2),
                    (1, cnt_h // self.vae_scale_factor // 2, cnt_w // self.vae_scale_factor // 2),
                    (1, sty_h // self.vae_scale_factor // 2, sty_w // self.vae_scale_factor // 2),
                ]
            )
        txt_seq_lens = txt_latent_masks.sum(dim=1).tolist()

        # Attention kwargs is used to pass values into LoRA
        attention_kwargs = {
            "lora_layer_indices": self.lora_layer_indices,
            "enable_lora": True,
            "w": strengths,
        }
        # Predict vector field
        latents = torch.cat([noisy_latents, img_latents], dim=1)
        pred_v = self.transformer(
            hidden_states=latents,
            encoder_hidden_states=txt_latents,
            encoder_hidden_states_mask=txt_latent_masks,
            timestep=timesteps / 1000,
            img_shapes=image_shapes,
            txt_seq_lens=txt_seq_lens,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        pred_v = pred_v[:, : noisy_latents.shape[1]]

        pred_v = self.unpack_vae_latents(
            pred_v,
            height=res_latents.shape[-2],
            width=res_latents.shape[-1],
        )
        target_v = eps_latents - res_latents

        loss_w = compute_loss_weighting_for_sd3(
            weighting_scheme=self.timestep_weighting_scheme,
            sigmas=sigmas,
        )

        flow_matching_loss = torch.mean(
            (loss_w.float() * (pred_v.float() - target_v.float()) ** 2).reshape(batch_size, -1),
            dim=1,
        )
        flow_matching_loss = flow_matching_loss.mean()

        return {
            "flow_matching_loss": flow_matching_loss,
            "predict_v": pred_v,
        }

    def finetune(
        self,
        accelerator: Accelerator,
        max_training_steps: int,
        train_loader: DataLoader,
        adapter_name: str = "default",
        val_loader: Optional[DataLoader] = None,
        generator: Optional[torch.Generator] = None,
        checkpointing_steps: int = 500,
        validation_steps: int = 500,
        output_dir: str = ".",
    ):
        world_size = accelerator.num_processes
        requires_val = val_loader is not None
        training_adapter_name = adapter_name
        self.disable_training(training_adapter_name)
        self.enable_training(training_adapter_name)

        num_steps_per_device = len(train_loader)
        num_steps_per_epoch = num_steps_per_device * world_size
        num_epochs = math.ceil(max_training_steps / num_steps_per_epoch)

        self.logger.info(f"# -- Finetune Params -- #")
        self.logger.info(f"{world_size=}")
        self.logger.info(f"{requires_val=}")
        self.logger.info(f"{num_steps_per_device=}")
        self.logger.info(f"{num_steps_per_epoch=}")
        self.logger.info(f"{num_epochs=}")
        self.logger.info(f"{max_training_steps=}")

        training_models = []
        optimizer_params = []
        training_models.append(self.transformer)
        optimizer_params.append(
            {"params": [p for p in self.transformer.parameters() if p.requires_grad], "lr": self.lr}
        )

        self.logger.info(f"{self.lr=}")
        optimizer = prodigyopt.Prodigy(optimizer_params)

        optimizer = accelerator.prepare(optimizer)
        self.vae = accelerator.prepare(self.vae)
        self.transformer = accelerator.prepare(self.transformer)

        self.logger.info(f"# -- Start Finetune -- #")
        global_steps = 0
        step_digit = len(str(max_training_steps))

        if requires_val:
            self.evaluate(accelerator, val_loader, training_adapter_name, output_dir, global_steps)

        train_dynamics_file = os.path.join(output_dir, "dynamics.jsonl")
        for epoch in range(num_epochs):
            for train_idx, batch in enumerate(train_loader):
                avg_loss = []
                with accelerator.accumulate(*training_models):
                    output = self._finetune_step(batch, generator)
                    fm_loss = output["flow_matching_loss"]
                    avg_loss.append(fm_loss.detach().cpu().item())

                    accelerator.backward(fm_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.transformer.parameters(), max_norm=self.max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()
                # End accumulation
                if accelerator.sync_gradients:
                    global_steps += 1
                    avg_loss = sum(avg_loss) / len(avg_loss)
                    self.logger.info(f"Step [{global_steps:0{step_digit}d}/{max_training_steps}] Loss {avg_loss:.4f}")

                    accelerator.log({"flow_matching_loss": avg_loss}, step=global_steps)
                    dynamic_line = {"flow_matching_loss": avg_loss, "step": global_steps}
                    with open(train_dynamics_file, "a") as f:
                        f.write(json.dumps(dynamic_line) + "\n")

                    if global_steps % checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            self.save_styctrl_checkpoint(
                                accelerator,
                                output_dir,
                                global_steps,
                                save_modules=self.modules_require_grad,
                            )
                    if requires_val and global_steps % validation_steps == 0:
                        self.logger.info(f"Step [{global_steps:0{step_digit}d}/{max_training_steps}] Start Evaluation")
                        self.evaluate(
                            accelerator,
                            val_loader,
                            training_adapter_name,
                            output_dir,
                            global_steps,
                        )
                        self.logger.info(f"Step [{global_steps:0{step_digit}d}/{max_training_steps}] End Evaluation")
                if global_steps >= max_training_steps:
                    break
            # End batch
            if global_steps >= max_training_steps:
                break
        # End epoch
        accelerator.wait_for_everyone()
        # Last val
        if requires_val:
            self.evaluate(accelerator, val_loader, training_adapter_name, output_dir, global_steps)

        self.logger.info(f"Training Finished. Output dir: {output_dir}")
        accelerator.end_training()


class FluxKontextFinetuner(QwenImageEditFinetuner):

    VAE_IMAGE_SIZE = 512 * 512
    DEFAULT_PROMPT = QwenImageEditFinetuner.DEFAULT_PROMPT

    def __init__(
        self,
        pretrained_pipeline: str = "black-forest-labs/FLUX.1-Kontext-dev",
        lora_layer_range: list[int] = [0, 18],
        load_dit: bool = True,
        load_text_encoder: bool = True,
        timestep_weighting_scheme: str = "logit_normal",
        lr: float = 1.0,
        max_grad_norm: float = 1.0,
        mu_logit_mean: float = 0.0,
        mu_logit_std: float = 1.0,
        mu_mode_scale: float = 1.29,
        modules_require_grad: list[Literal["lora_A", "lora_B", "projector"]] = ["lora_A", "lora_B"],
        torch_dtype: str = "bf16",
        device: str | int = "cpu",
        logger: logging.Logger = None,
    ):
        self.device = device
        self.dtype = torch.float32
        if torch_dtype == "fp16":
            self.dtype = torch.float16
        elif torch_dtype == "bf16":
            self.dtype = torch.bfloat16

        self.load_dit = load_dit
        self.load_text_encoder = load_text_encoder
        self.modules_require_grad = modules_require_grad
        self.lora_layer_indices = list(range(lora_layer_range[0], lora_layer_range[1] + 1))
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.timestep_weighting_scheme = timestep_weighting_scheme
        self.mu_logit_mean = mu_logit_mean
        self.mu_logit_std = mu_logit_std
        self.mu_mode_scale = mu_mode_scale
        self.logger = logger or logging.getLogger(__name__)

        if self.load_dit:
            self.logger.info("# -- Initialize FLUX Kontext Model -- #")
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                pretrained_pipeline, subfolder="scheduler", torch_dtype=self.dtype
            )
            self.scheduler_val = copy.deepcopy(self.scheduler)
            self.vae = (
                AutoencoderKL.from_pretrained(pretrained_pipeline, subfolder="vae", torch_dtype=self.dtype)
                .to(device)
                .requires_grad_(False)
            )
            self.transformer = (
                FluxStyCtrlTransformer2DModel.from_pretrained(
                    pretrained_pipeline, subfolder="transformer", torch_dtype=self.dtype
                )
                .to(device)
                .requires_grad_(False)
            )
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.latent_channels = self.vae.config.latent_channels
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        if self.load_text_encoder:
            self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_pipeline, subfolder="tokenizer")
            self.text_encoder = (
                CLIPTextModel.from_pretrained(pretrained_pipeline, subfolder="text_encoder", torch_dtype=self.dtype)
                .to(device)
                .requires_grad_(False)
            )
            self.tokenizer_2 = T5TokenizerFast.from_pretrained(pretrained_pipeline, subfolder="tokenizer_2")
            self.text_encoder_2 = (
                T5EncoderModel.from_pretrained(pretrained_pipeline, subfolder="text_encoder_2", torch_dtype=self.dtype)
                .to(device)
                .requires_grad_(False)
            )
        self.active_adapter = None

    def encode_prompt(
        self,
        batch_size: int,
        prompt: str | list[str] = DEFAULT_PROMPT,
        max_sequence_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.load_text_encoder:
            raise ValueError("FluxKontextFinetuner requires load_text_encoder=true unless prompt embeddings are added.")

        prompt = [prompt] * batch_size if isinstance(prompt, str) else prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        pooled_prompt_embeds = self.text_encoder(
            text_inputs.input_ids.to(self.device), output_hidden_states=False
        ).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=self.device, dtype=self.dtype)

        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder_2(text_inputs_2.input_ids.to(self.device), output_hidden_states=False)[0]
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=self.device, dtype=self.dtype)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
        latent_image_ids = latent_image_ids.reshape(height * width, 3)
        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def pack_vae_latents(latents: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = latents.shape
        latents = latents.view(bsz, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(bsz, (height // 2) * (width // 2), channels * 4)

    @staticmethod
    def unpack_vae_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        bsz, _, channels = latents.shape
        latents = latents.view(bsz, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(bsz, channels // 4, height, width)

    def encode_vae_images(self, images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        image_latents = retrieve_latents(self.vae.encode(images), generator=generator, sample_mode="sample")
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return image_latents.to(dtype=images.dtype)

    def _resize_to_area(self, images: torch.Tensor, target_size: int, width_scale: float = 1.0) -> torch.Tensor:
        _, height, width = images.shape[-3:]
        new_h, new_w = calculate_dimensions(
            target_size * width_scale, width * width_scale / height, self.vae_scale_factor * 2
        )
        return T.resize(images, [new_h, new_w])

    def _finetune_step(self, batch: tuple[torch.Tensor], generator: Optional[torch.Generator] = None):
        if len(batch) < 4:
            raise ValueError("FluxKontextFinetuner expects batches with content, style, result, and strength.")
        cnt_images, sty_images, res_images, strengths = batch[:4]
        cnt_images = cnt_images.to(device=self.device, dtype=self.dtype)
        sty_images = sty_images.to(device=self.device, dtype=self.dtype)
        res_images = res_images.to(device=self.device, dtype=self.dtype)
        strengths = strengths.to(device=self.device, dtype=self.dtype)
        batch_size = cnt_images.shape[0]

        cnt_images = self._resize_to_area(cnt_images, self.VAE_IMAGE_SIZE)
        sty_images = T.resize(sty_images, list(cnt_images.shape[-2:]))
        res_images = self._resize_to_area(res_images, self.VAE_IMAGE_SIZE)

        cnt_images = cnt_images * 2.0 - 1.0
        sty_images = sty_images * 2.0 - 1.0
        res_images = res_images * 2.0 - 1.0
        dtype, device = res_images.dtype, res_images.device

        cnt_latents = self.pack_vae_latents(self.encode_vae_images(cnt_images, generator))
        sty_latents = self.pack_vae_latents(self.encode_vae_images(sty_images, generator))
        image_latents = torch.cat([cnt_latents, sty_latents], dim=1)
        res_latents = self.encode_vae_images(res_images, generator)
        eps_latents = torch.randn_like(res_latents, dtype=dtype, device=device)

        u = compute_density_for_timestep_sampling(
            self.timestep_weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.mu_logit_mean,
            logit_std=self.mu_logit_std,
            mode_scale=self.mu_mode_scale,
            device=device,
            generator=generator,
        )
        indices = (u * self.scheduler.config.num_train_timesteps).long().to("cpu")
        timesteps = self.scheduler.timesteps[indices].to(device)
        sigmas = self.get_sigmas(timesteps, latent_ndim=res_latents.ndim).to(device, dtype=dtype)
        noisy_latents = (1.0 - sigmas) * res_latents + sigmas * eps_latents
        noisy_latents = self.pack_vae_latents(noisy_latents)

        prompt_embeds, pooled_prompt_embeds, txt_ids = self.encode_prompt(batch_size)
        latent_ids = self._prepare_latent_image_ids(
            batch_size, res_latents.shape[-2] // 2, res_latents.shape[-1] // 2, device, dtype
        )
        cnt_latent_h = cnt_images.shape[-2] // self.vae_scale_factor
        cnt_latent_w = cnt_images.shape[-1] // self.vae_scale_factor
        sty_latent_h = sty_images.shape[-2] // self.vae_scale_factor
        sty_latent_w = sty_images.shape[-1] // self.vae_scale_factor
        cnt_ids = self._prepare_latent_image_ids(batch_size, cnt_latent_h // 2, cnt_latent_w // 2, device, dtype)
        sty_ids = self._prepare_latent_image_ids(batch_size, sty_latent_h // 2, sty_latent_w // 2, device, dtype)
        cnt_ids[..., 0] = 1
        sty_ids[..., 0] = 2
        img_ids = torch.cat([latent_ids, cnt_ids, sty_ids], dim=0)

        latent_model_input = torch.cat([noisy_latents, image_latents], dim=1)
        guidance = None
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([batch_size], 3.5, device=device, dtype=torch.float32)

        pred_v = self.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=timesteps / 1000,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs={
                "lora_layer_indices": self.lora_layer_indices,
                "enable_lora": True,
                "w": strengths,
            },
            return_dict=False,
        )[0]
        pred_v = pred_v[:, : noisy_latents.shape[1]]
        pred_v = self.unpack_vae_latents(pred_v, height=res_latents.shape[-2], width=res_latents.shape[-1])
        target_v = eps_latents - res_latents

        loss_w = compute_loss_weighting_for_sd3(self.timestep_weighting_scheme, sigmas=sigmas)
        flow_matching_loss = torch.mean(
            (loss_w.float() * (pred_v.float() - target_v.float()) ** 2).reshape(batch_size, -1),
            dim=1,
        ).mean()
        return {"flow_matching_loss": flow_matching_loss, "predict_v": pred_v}

    @torch.no_grad()
    def _evaluate_one_image(
        self,
        pipeline: FluxKontextPipeline,
        cnt_image,
        sty_image,
        strength: torch.Tensor,
        generator: torch.Generator,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 16,
        guidance_scale: float = 3.5,
    ):
        device = pipeline._execution_device
        prompt_embeds, pooled_prompt_embeds, txt_ids = pipeline.encode_prompt(
            prompt=self.DEFAULT_PROMPT,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        dtype = prompt_embeds.dtype
        batch_size = 1
        multiple_of = pipeline.vae_scale_factor * 2
        height = height // multiple_of * multiple_of
        width = width // multiple_of * multiple_of

        num_channels_latents = pipeline.transformer.config.in_channels // 4
        latents, _, latent_ids, _ = pipeline.prepare_latents(
            None,
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            None,
        )

        cnt_tensor = pipeline.image_processor.preprocess(cnt_image, height=height, width=width).to(
            device=device, dtype=dtype
        )
        sty_tensor = pipeline.image_processor.preprocess(sty_image, height=height, width=width).to(
            device=device, dtype=dtype
        )
        cnt_latents = pipeline._encode_vae_image(cnt_tensor, generator=generator)
        sty_latents = pipeline._encode_vae_image(sty_tensor, generator=generator)
        cnt_latent_h, cnt_latent_w = cnt_latents.shape[2:]
        sty_latent_h, sty_latent_w = sty_latents.shape[2:]
        cnt_latents = pipeline._pack_latents(cnt_latents, batch_size, num_channels_latents, cnt_latent_h, cnt_latent_w)
        sty_latents = pipeline._pack_latents(sty_latents, batch_size, num_channels_latents, sty_latent_h, sty_latent_w)
        image_latents = torch.cat([cnt_latents, sty_latents], dim=1)

        cnt_ids = pipeline._prepare_latent_image_ids(batch_size, cnt_latent_h // 2, cnt_latent_w // 2, device, dtype)
        sty_ids = pipeline._prepare_latent_image_ids(batch_size, sty_latent_h // 2, sty_latent_w // 2, device, dtype)
        cnt_ids[..., 0] = 1
        sty_ids[..., 0] = 2
        img_ids = torch.cat([latent_ids, cnt_ids, sty_ids], dim=0)

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
            guidance = torch.full([batch_size], guidance_scale, device=device, dtype=torch.float32)

        joint_attention_kwargs = {
            "lora_layer_indices": self.lora_layer_indices,
            "enable_lora": True,
            "w": strength.to(device=device, dtype=self.dtype).reshape(1),
        }
        for t in timesteps:
            latent_model_input = torch.cat([latents, image_latents], dim=1)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            noise_pred = pipeline.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : latents.size(1)]
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
        latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        image = pipeline.vae.decode(latents, return_dict=False)[0]
        return pipeline.image_processor.postprocess(image, output_type="pil")[0]

    @torch.no_grad()
    def evaluate(
        self,
        accelerator: Accelerator,
        val_loader: DataLoader,
        adapter_name: str,
        output_dir: str,
        global_step: int,
        max_num_val: int = 4,
        recover_train: bool = True,
    ):
        self.disable_training(adapter_name)
        pipeline = FluxKontextPipeline(
            transformer=accelerator.unwrap_model(self.transformer),
            vae=self.vae,
            scheduler=self.scheduler_val,
            text_encoder=self.text_encoder if self.load_text_encoder else None,
            tokenizer=self.tokenizer if self.load_text_encoder else None,
            text_encoder_2=self.text_encoder_2 if self.load_text_encoder else None,
            tokenizer_2=self.tokenizer_2 if self.load_text_encoder else None,
        )
        pipeline.to(dtype=torch.float32)
        val_save_dir = os.path.join(output_dir, f"eval-{global_step}")
        if accelerator.is_main_process:
            os.makedirs(val_save_dir, exist_ok=True)
        for i, batch in enumerate(val_loader):
            if i >= max_num_val:
                break
            cnt_images, sty_images, res_images, strengths = batch[:4]
            for j in range(cnt_images.shape[0]):
                output = self._evaluate_one_image(
                    pipeline=pipeline,
                    cnt_image=T.to_pil_image(cnt_images[j]),
                    sty_image=T.to_pil_image(sty_images[j]),
                    strength=strengths[j],
                    generator=torch.Generator(self.device).manual_seed(42),
                    num_inference_steps=16,
                )
                gt_h, gt_w = res_images[j].shape[-2:]
                pred = T.resize(T.to_tensor(output), [gt_h, gt_w])
                concat = torch.cat([cnt_images[j].cpu(), sty_images[j].cpu(), pred.cpu(), res_images[j].cpu()], dim=-1)
                if accelerator.is_main_process:
                    save_image(concat, os.path.join(val_save_dir, f"sample_{i}_{j}.jpg"))
        if recover_train:
            pipeline.to(dtype=torch.bfloat16)
            self.enable_training(adapter_name)
