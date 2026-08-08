from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    apply_rotary_emb,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.models.transformers.transformer_flux import FluxPosEmbed
from diffusers.utils import USE_PEFT_BACKEND, is_torch_npu_available, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph

from .styctrl import StyCtrlLoRA


logger = logging.get_logger(__name__)


def _linear(module: nn.Module, x: torch.Tensor, enable_lora: bool, w: Optional[torch.Tensor]) -> torch.Tensor:
    if enable_lora and isinstance(module, StyCtrlLoRA):
        return module(x, enable_lora=True, w=w)
    return module(x)


def _feed_forward(ff: FeedForward, x: torch.Tensor, enable_lora: bool, w: Optional[torch.Tensor]) -> torch.Tensor:
    for module in ff.net:
        x = _linear(module, x, enable_lora, w) if isinstance(module, StyCtrlLoRA) else module(x)
    return x


class FluxStyCtrlAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0.")

    def __call__(
        self,
        attn: "FluxStyCtrlAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        enable_lora: bool = False,
        w: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if attn.fused_projections and enable_lora:
            raise RuntimeError("StyCtrlLoRA requires unfused FLUX attention projections.")

        query = _linear(attn.to_q, hidden_states, enable_lora, w)
        key = _linear(attn.to_k, hidden_states, enable_lora, w)
        value = _linear(attn.to_v, hidden_states, enable_lora, w)

        encoder_query = encoder_key = encoder_value = None
        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            encoder_query = _linear(attn.add_q_proj, encoder_hidden_states, enable_lora, w)
            encoder_key = _linear(attn.add_k_proj, encoder_hidden_states, enable_lora, w)
            encoder_value = _linear(attn.add_v_proj, encoder_hidden_states, enable_lora, w)

        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))

        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            encoder_query = attn.norm_added_q(encoder_query.unflatten(-1, (attn.heads, -1)))
            encoder_key = attn.norm_added_k(encoder_key.unflatten(-1, (attn.heads, -1)))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is None:
            return hidden_states

        encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
            [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
        )
        hidden_states = _linear(attn.to_out[0], hidden_states, enable_lora, w)
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = _linear(attn.to_add_out, encoder_hidden_states, enable_lora, w)
        return hidden_states, encoder_hidden_states


class FluxStyCtrlAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = FluxStyCtrlAttnProcessor
    _available_processors = [FluxStyCtrlAttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only: Optional[bool] = None,
        pre_only: bool = False,
        elementwise_affine: bool = True,
        processor=None,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias
        self.fused_projections = False

        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.pre_only:
            self.to_out = torch.nn.ModuleList([torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias), torch.nn.Dropout(dropout)])

        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        self.set_processor(processor or self._default_processor_cls())

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, **kwargs):
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)


@maybe_allow_in_graph
class FluxStyCtrlSingleTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)
        self.attn = FluxStyCtrlAttention(
            query_dim=dim, dim_head=attention_head_dim, heads=num_attention_heads, out_dim=dim, bias=True, eps=1e-6, pre_only=True
        )

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        joint_attention_kwargs = joint_attention_kwargs or {}
        enable_lora = joint_attention_kwargs.get("enable_lora", False)
        w = joint_attention_kwargs.get("w", None)
        mlp_hidden_states = self.act_mlp(_linear(self.proj_mlp, norm_hidden_states, enable_lora, w))
        attn_output = self.attn(hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb, **joint_attention_kwargs)
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = residual + gate.unsqueeze(1) * _linear(self.proj_out, hidden_states, enable_lora, w)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]


@maybe_allow_in_graph
class FluxStyCtrlTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)
        self.attn = FluxStyCtrlAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}
        enable_lora = joint_attention_kwargs.get("enable_lora", False)
        w = joint_attention_kwargs.get("w", None)

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * _feed_forward(self.ff, norm_hidden_states, enable_lora, w)

        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * _feed_forward(
            self.ff_context, norm_encoder_hidden_states, enable_lora, w
        )
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


class FluxStyCtrlTransformer2DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    CacheMixin,
    AttentionMixin,
):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxStyCtrlTransformerBlock", "FluxStyCtrlSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["FluxStyCtrlTransformerBlock", "FluxStyCtrlSingleTransformerBlock"]
    _cp_plan = {
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "img_ids": ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
            "txt_ids": ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        text_time_guidance_cls = CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                FluxStyCtrlTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxStyCtrlSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        if joint_attention_kwargs is None:
            joint_attention_kwargs = {}
        lora_layer_indices = joint_attention_kwargs.pop(
            "lora_layer_indices",
            list(range(max(len(self.transformer_blocks), len(self.single_transformer_blocks)))),
        )
        double_block_kwargs = [{} for _ in range(len(self.transformer_blocks))]
        single_block_kwargs = [{} for _ in range(len(self.single_transformer_blocks))]
        for i in lora_layer_indices:
            if i < len(double_block_kwargs):
                double_block_kwargs[i] = joint_attention_kwargs
            if i < len(single_block_kwargs):
                single_block_kwargs[i] = joint_attention_kwargs

        hidden_states = _linear(
            self.x_embedder,
            hidden_states,
            joint_attention_kwargs.get("enable_lora", False),
            joint_attention_kwargs.get("w", None),
        )
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_text_embed(timestep, pooled_projections) if guidance is None else self.time_text_embed(timestep, guidance, pooled_projections)
        encoder_hidden_states = _linear(
            self.context_embedder,
            encoder_hidden_states,
            joint_attention_kwargs.get("enable_lora", False),
            joint_attention_kwargs.get("w", None),
        )

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        ids = torch.cat((txt_ids, img_ids), dim=0)
        if is_torch_npu_available():
            freqs_cos, freqs_sin = self.pos_embed(ids.cpu())
            image_rotary_emb = (freqs_cos.npu(), freqs_sin.npu())
        else:
            image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=double_block_kwargs[index_block],
            )
            if controlnet_block_samples is not None:
                interval_control = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=single_block_kwargs[index_block],
            )
            if controlnet_single_block_samples is not None:
                interval_control = int(np.ceil(len(self.single_transformer_blocks) / len(controlnet_single_block_samples)))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        enable_lora = joint_attention_kwargs.get("enable_lora", False)
        w = joint_attention_kwargs.get("w", None)
        output = _linear(self.proj_out, hidden_states, enable_lora, w)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
