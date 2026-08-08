import math
import torch
import numpy as np

from scipy.interpolate import make_interp_spline
from safetensors.torch import load_file


def freeze_parameters(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


def summarize_model(model):
    # Calculate parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    # Calculate size on disk/memory (assuming float32 = 4 bytes)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio(%)": trainable_params / total_params,
        "model_size_mb(MB)": size_all_mb,
    }


def lora_scale_states(
    anchor_states: list[str],
    query_point: float,
    device: torch.device,
    dtype: torch.dtype,
):
    state = load_file(anchor_states[0])
    new_state = {}
    for k, p in state.items():
        if "projector.weight" in k:
            new_state[k] = torch.full_like(p, fill_value=2 * query_point, device=device, dtype=dtype)
        elif "projector.bias" in k:
            new_state[k] = torch.zeros_like(p, device=device, dtype=dtype)
    return new_state


def neighbor_interpolate_states(
    anchor_states: list[str],
    query_point: float,
    device: torch.device,
    dtype: torch.dtype,
):
    num_levels = len(anchor_states)
    level = math.floor(query_point * num_levels)

    interp_state = {}
    if level == 0:
        state = load_file(anchor_states[0])
        w = query_point * num_levels - level
        for k, p in state.items():
            interp_state[k] = (w * p).to(device=device, dtype=dtype)
    elif level == num_levels:
        interp_state = load_file(anchor_states[-1])
        for k, p in interp_state.items():
            interp_state[k] = p.to(device=device, dtype=dtype)
    else:
        state1 = load_file(anchor_states[level - 1])
        state2 = load_file(anchor_states[level])
        w = query_point * num_levels - level
        for k, p1 in state1.items():
            p2 = state2[k]
            interp_state[k] = ((1 - w) * p1 + w * p2).to(device=device, dtype=dtype)
    return interp_state


def bspline_interpolate_states(
    anchor_states: list[str],
    query_point: float,
    X,
    device: torch.device,
    dtype: torch.dtype,
    order=3,
):
    states = load_file(anchor_states[0])
    states = {k: [p] for k, p in states.items()}
    for sp in anchor_states[1:]:
        s = load_file(sp)
        for k, p in s.items():
            states[k].append(p)

    interp_state = {}
    # X = [0.2, 0.4, 0.6, 0.8, 1.0]
    for k, pl in states.items():
        Y = torch.stack(pl, dim=0).to(torch.float32).numpy()
        bsp = make_interp_spline(X, Y, k=order)
        y = torch.from_numpy(bsp(np.array(query_point))).to(device=device, dtype=dtype)
        interp_state[k] = y
    return interp_state
