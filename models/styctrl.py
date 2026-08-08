import math
import torch
import logging
import torch.nn as nn

from typing import Literal, Optional


logger = logging.getLogger(__name__)


class HadamardLayer(nn.Module):

    def __init__(
        self,
        features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(features, device=device, dtype=dtype))

        nn.init.normal_(self.weight, mean=1.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.to(dtype=self.dtype)
        x = x * self.weight + self.bias
        return x.to(x_dtype)


class StyCtrlLoRA(nn.Module):

    def __init__(self, base_model: nn.Linear):
        super().__init__()

        self.base_model = base_model

        self.dtype = self.base_model.weight.dtype
        self.in_features = self.base_model.in_features
        self.out_features = self.base_model.out_features

        self.projector = nn.ModuleDict()
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.bias = nn.ParameterDict()

        self.active_adapter = "styctrl"
        self.proj_types = {}

    def get_current_adapters(self) -> list[str]:
        return [k for k in self.lora_A.keys()]

    def add_adapter(
        self,
        rank: int,
        adapter_name: str = "default",
        proj_type: Literal["none", "low_rank_linear", "low_rank_scale", "in_scale"] = "none",
        bias: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        if adapter_name in self.get_current_adapters():
            logger.warning(
                f"{__class__.__name__}: {adapter_name=} already exists in {__class__}, existing adapters are: {self.get_current_adapters()}."
            )
            return

        device = device or self.base_model.weight.device
        dtype = dtype or self.base_model.weight.dtype

        # Add strength projector
        if proj_type == "low_rank_linear":
            # Add linear projector in the low rank space
            self.projector[adapter_name] = nn.Linear(rank, rank, bias=True, device=device, dtype=dtype)
            nn.init.eye_(self.projector[adapter_name].weight)
            nn.init.zeros_(self.projector[adapter_name].bias)
        elif proj_type == "low_rank_scale":
            self.projector[adapter_name] = HadamardLayer(rank, device=device, dtype=dtype)
        elif proj_type == "in_scale":
            self.projector[adapter_name] = HadamardLayer(self.in_features, device=device, dtype=dtype)
        elif proj_type in ["none", "lora"]:
            self.projector[adapter_name] = nn.Identity()
        else:
            raise KeyError(f"{proj_type=} has not been implemented.")
        self.proj_types[adapter_name] = proj_type

        # Add main LoRA linear projectors
        self.lora_A[adapter_name] = nn.Parameter(torch.empty(rank, self.in_features)).to(device=device, dtype=dtype)
        self.lora_B[adapter_name] = nn.Parameter(torch.empty(self.out_features, rank)).to(device=device, dtype=dtype)

        nn.init.normal_(self.lora_A[adapter_name].data, mean=0.0, std=0.02)
        nn.init.normal_(self.lora_B[adapter_name].data, mean=0.0, std=0.02)

        # Add bias
        if bias:
            init_bias = 1 / math.sqrt(rank)
            self.bias[adapter_name] = nn.Parameter(torch.zeros((self.out_features,), dtype=self.dtype, device=device))
            nn.init.uniform_(self.bias[adapter_name], -init_bias, init_bias)

    def activate(self, adapter_name: str):
        if adapter_name not in self.get_current_adapters():
            logger.warning(
                f"{__class__.__name__}: {adapter_name=} not in current adapter list: {self.get_current_adapters()}"
            )
            return
        self.active_adapter = adapter_name

    def deactivate(self):
        self.active_adapter = None

    def forward(
        self,
        x: torch.Tensor,
        enable_lora: bool = False,
        w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.to(self.dtype)
        out = self.base_model(x)

        if enable_lora:
            adapter_name = self.active_adapter
            proj_type = self.proj_types[adapter_name]

            if w is not None and proj_type == "in_scale":
                w = w.to(self.dtype)
                w = w.unsqueeze(-1).repeat(1, self.in_features)
                w = self.projector[adapter_name](w)
                x = w * x

            lora_out = x @ self.lora_A[adapter_name].T

            if w is not None and proj_type == "low_rank_linear":
                w = w.to(self.dtype)
                while w.ndim < lora_out.ndim:
                    w = w.unsqueeze(-1)
                lora_out = self.projector[adapter_name](w * lora_out)

            if w is not None and proj_type == "low_rank_scale":
                w = w.to(self.dtype)
                w = w.unsqueeze(-1).repeat(1, lora_out.shape[-1])
                w = self.projector[adapter_name](w)
                lora_out = w * lora_out
            
            lora_out = lora_out @ self.lora_B[adapter_name].T

            if w is not None and proj_type == "lora":
                w = w.to(self.dtype)
                lora_out = w * lora_out

            out = out + lora_out

        out = out.to(x_dtype)
        return out


def register_styctrl(
    model: nn.Module,
    target_modules: list[str],
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    r"""
    This method only insert LoRA adapter into model, not add or activate any adapter.
    If there are already lora_adapter in model, this method do nothing.
    """
    for name, module in model.named_modules():
        if not any(target in name for target in target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue

        splited_name = name.split(".")
        parent_node = model.get_submodule(".".join(splited_name[:-1]))
        target_name = splited_name[-1]

        lora_layer = StyCtrlLoRA(base_model=module).to(device=device, dtype=dtype)

        module.requires_grad_(False)
        setattr(parent_node, target_name, lora_layer)
        logger.debug(f"Inject LoRA into {name}.")


def add_styctrl(
    model: nn.Module,
    rank: int,
    lora_layer_indices: list[int],
    adapter_name: str = "default",
    proj_type: Literal["none", "linear", "scale"] = "none",
    bias: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    for name, module in model.named_modules():
        if not any(
            f"transformer_blocks.{i}." in name or f"single_transformer_blocks.{i}." in name
            for i in lora_layer_indices
        ):
            continue
        if isinstance(module, StyCtrlLoRA):
            module.add_adapter(
                rank=rank,
                adapter_name=adapter_name,
                proj_type=proj_type,
                bias=bias,
                device=device,
                dtype=dtype,
            )


def activate(model: nn.Module, adapter_name: str) -> None:
    for n, module in model.named_modules():
        if isinstance(module, StyCtrlLoRA):
            logger.debug(f"Activate LoRA in {n}")
            module.activate(adapter_name)


def deactivate(model: nn.Module) -> None:
    for n, module in model.named_modules():
        if isinstance(module, StyCtrlLoRA):
            logger.debug(f"Deactivate LoRA in {n}")
            module.deactivate()
