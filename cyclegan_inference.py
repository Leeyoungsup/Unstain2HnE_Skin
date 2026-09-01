from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from cyclegan_core import ResNetGenerator
from two_stage_virtual_staining import image_to_od, prepare_physical_view


class SingleODToRGB(nn.Module):
    """Adapts the trained 3-channel OD CycleGAN to the common 1-channel WSI runner."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, od1):
        return self.generator(od1.repeat(1, 3, 1, 1))


class IdentityColorizer(nn.Module):
    def forward(self, image):
        return image


def load_cyclegan_models(checkpoint_path, device=None):
    """Load a trusted local paired CycleGAN checkpoint for A→B inference."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    params = state["params"]

    generator_ab = ResNetGenerator(
        3, params["ngf"], params["residual_blocks"]
    )
    generator_ba = ResNetGenerator(
        3, params["ngf"], params["residual_blocks"]
    )
    generator_ab.load_state_dict(state["G_AB"])
    generator_ba.load_state_dict(state["G_BA"])
    generator_ab = generator_ab.to(device).eval()
    generator_ba = generator_ba.to(device).eval()
    wsi_generator = SingleODToRGB(generator_ab).to(device).eval()
    identity = IdentityColorizer().to(device).eval()
    for model in (generator_ab, generator_ba):
        for parameter in model.parameters():
            parameter.requires_grad = False

    metadata = {
        "device": str(device),
        "epoch": int(state["epoch"]) + 1,
        "unstain_od_max": float(state["od_max"]),
        "input_size": int(params["input_size"]),
        "original_size": int(params["original_size"]),
        "source_mpp": float(params["source_mpp"]),
        "target_mpp": float(params["target_mpp"]),
        "paired_training": bool(params.get("paired_training", False)),
    }
    return wsi_generator, identity, generator_ba, metadata


@torch.inference_mode()
def infer_cyclegan_patch(image, wsi_generator, generator_ba, metadata):
    if image.size[0] != image.size[1]:
        raise ValueError(f"Patch must be square, got {image.size}")
    params = {
        "input_size": metadata["input_size"],
        "original_size": image.size[0],
        "source_mpp": metadata["source_mpp"],
        "target_mpp": metadata["target_mpp"],
    }
    prepared = prepare_physical_view(image.convert("RGB"), params)
    od1 = image_to_od(prepared, metadata["unstain_od_max"]).unsqueeze(0)
    device = torch.device(metadata["device"])
    amp_enabled = device.type == "cuda"
    with torch.amp.autocast(device.type, enabled=amp_enabled):
        generated = wsi_generator(od1.to(device))
        recovered_od3 = generator_ba(generated)
    generated_rgb = (
        ((generated[0].float().cpu().clamp(-1, 1) + 1) * 127.5)
        .permute(1, 2, 0).byte().numpy()
    )
    recovered_od = recovered_od3[0].float().cpu().mean(0)
    return od1[0], recovered_od, generated_rgb
