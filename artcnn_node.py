import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort


class ArtCNNONNXError(RuntimeError):
    pass


_SESSION_CACHE: Dict[str, ort.InferenceSession] = {}


def _onnx_folder() -> str:
    node_dir = os.path.dirname(os.path.abspath(__file__))
    comfy_root = os.path.abspath(os.path.join(node_dir, os.pardir, os.pardir))
    return os.path.join(comfy_root, "models", "onnx", "Upscale")


def _list_models(include_none=False):
    folder = _onnx_folder()
    os.makedirs(folder, exist_ok=True)
    models = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(".onnx") and os.path.isfile(os.path.join(folder, f))
    )
    if include_none:
        return ["None"] + models
    return models or ["<no .onnx models found>"]


def _get_session(path: str):
    key = os.path.abspath(path)
    if key in _SESSION_CACHE:
        return _SESSION_CACHE[key]

    available = ort.get_available_providers()
    if "DmlExecutionProvider" not in available:
        raise ArtCNNONNXError(
            "onnxruntime-directml is required for this node. "
            f"Available providers: {available}"
        )

    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        key,
        sess_options=so,
        providers=["DmlExecutionProvider"],
    )

    active = session.get_providers()
    if not active or active[0] != "DmlExecutionProvider":
        raise ArtCNNONNXError(
            "DirectML was requested but is not the active execution provider. "
            f"Session providers: {active}"
        )

    _SESSION_CACHE[key] = session
    return session


def _shape_dim_is_int(v):
    return isinstance(v, (int, np.integer))


def _model_io_channels(session):
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    if len(inp.shape) != 4 or len(out.shape) != 4:
        raise ArtCNNONNXError(
            f"ArtCNN model must have 4D input/output, got input={inp.shape}, output={out.shape}"
        )
    in_ch = inp.shape[1] if _shape_dim_is_int(inp.shape[1]) else None
    out_ch = out.shape[1] if _shape_dim_is_int(out.shape[1]) else None
    if in_ch is None or out_ch is None:
        raise ArtCNNONNXError(
            f"Channel dimensions must be static for this node, got input={inp.shape}, output={out.shape}"
        )
    return int(in_ch), int(out_ch)


def _validate_luma(session):
    in_ch, out_ch = _model_io_channels(session)
    if in_ch != 1 or out_ch != 1:
        raise ArtCNNONNXError(
            f"Selected model is not a 1->1 luma model: input={in_ch} channels, output={out_ch} channels"
        )


def _validate_rgb(session):
    in_ch, out_ch = _model_io_channels(session)
    if in_ch != 3 or out_ch != 3:
        raise ArtCNNONNXError(
            f"Selected model is not a 3->3 RGB model: input={in_ch} channels, output={out_ch} channels"
        )

def _validate_chroma(session):
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    if len(inp.shape) != 4 or len(out.shape) != 4:
        raise ArtCNNONNXError(
            f"Chroma model must be 4D input/output, got input={inp.shape}, output={out.shape}"
        )
    if _shape_dim_is_int(inp.shape[1]) and inp.shape[1] != 3:
        raise ArtCNNONNXError(
            f"This ArtCNN chroma path expects a 3-channel Y/Cb/Cr input model, got {inp.shape}"
        )
    if _shape_dim_is_int(out.shape[1]) and out.shape[1] != 2:
        raise ArtCNNONNXError(
            f"This ArtCNN chroma path expects 2-channel Cb/Cr output, got {out.shape}"
        )


def _run_fixed_batch(session, x: np.ndarray) -> np.ndarray:
    if x.ndim != 4:
        raise ArtCNNONNXError(f"Expected NCHW input, got {x.shape}")
    inp = session.get_inputs()[0]
    static_batch = inp.shape[0]
    if _shape_dim_is_int(static_batch) and static_batch == 1 and x.shape[0] != 1:
        outs = []
        for i in range(x.shape[0]):
            outs.append(np.asarray(session.run(None, {inp.name: x[i:i + 1]})[0], dtype=np.float32))
        return np.concatenate(outs, axis=0)
    return np.asarray(session.run(None, {inp.name: x})[0], dtype=np.float32)


def _run_luma(session, y: np.ndarray) -> np.ndarray:
    out = _run_fixed_batch(session, y)
    if out.ndim != 4 or out.shape[1] != 1:
        raise ArtCNNONNXError(f"Unexpected luma output shape: {out.shape}")
    return out


def _run_chroma(session, ycbcr: np.ndarray) -> np.ndarray:
    out = _run_fixed_batch(session, ycbcr)
    if out.ndim != 4 or out.shape[1] != 2:
        raise ArtCNNONNXError(f"Unexpected chroma output shape: {out.shape}")
    return out


def _rgb_to_ycbcr(rgb: torch.Tensor):
    # Full-range BT.709 YCbCr, all channels normalized to [0, 1].
    kr, kb = 0.2126, 0.0722
    kg = 1.0 - kr - kb
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = kr * r + kg * g + kb * b
    cb = (b - y) / (2.0 * (1.0 - kb)) + 0.5
    cr = (r - y) / (2.0 * (1.0 - kr)) + 0.5
    return torch.cat((y, cb, cr), dim=1)


def _ycbcr_to_rgb(ycbcr: torch.Tensor):
    kr, kb = 0.2126, 0.0722
    kg = 1.0 - kr - kb
    y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
    r = y + 2.0 * (1.0 - kr) * (cr - 0.5)
    b = y + 2.0 * (1.0 - kb) * (cb - 0.5)
    g = (y - kr * r - kb * b) / kg
    return torch.cat((r, g, b), dim=1).clamp(0, 1)


def _filter_kernel(x, kind):
    # Kernels are defined in source-pixel coordinates.  The resize is only
    # used for upscaling here, so no anti-aliasing shrink of the kernel is needed.
    if kind == "bilinear":
        ax = abs(x)
        return max(0.0, 1.0 - ax)

    if kind == "catmull-rom":
        # Cubic convolution with a=-0.5 (Catmull-Rom). Sharper than Mitchell
        # without the broader support/ringing of Lanczos.
        ax = abs(x)
        if ax < 1.0:
            return 1.5 * ax**3 - 2.5 * ax**2 + 1.0
        if ax < 2.0:
            return -0.5 * ax**3 + 2.5 * ax**2 - 4.0 * ax + 2.0
        return 0.0

    if kind == "mitchell":
        # Mitchell-Netravali, B=C=1/3.
        B = C = 1.0 / 3.0
        ax = abs(x)
        if ax < 1.0:
            return ((12 - 9*B - 6*C)*ax**3 + (-18 + 12*B + 6*C)*ax**2 + (6 - 2*B)) / 6.0
        if ax < 2.0:
            return ((-B - 6*C)*ax**3 + (6*B + 30*C)*ax**2 + (-12*B - 48*C)*ax + (8*B + 24*C)) / 6.0
        return 0.0

    if kind == "lanczos":
        if x == 0.0:
            return 1.0
        ax = abs(x)
        if ax >= 3.0:
            return 0.0
        import math
        pix = math.pi * x
        pix3 = pix / 3.0
        return (math.sin(pix) / pix) * (math.sin(pix3) / pix3)

    raise ArtCNNONNXError(f"Unknown chroma resampler: {kind}")


def _resize_axis_np(arr, out_len, axis, kind):
    # Proper pixel-center mapping for separable reconstruction filtering.
    # Vectorized over all batch/channel/spatial dimensions other than axis.
    import math
    in_len = int(arr.shape[axis])
    out_len = int(out_len)
    if in_len == out_len:
        return arr.astype(np.float32, copy=False)

    radius = {
        "bilinear": 1.0,
        "catmull-rom": 2.0,
        "mitchell": 2.0,
        "lanczos": 3.0,
    }[kind]

    scale = out_len / in_len
    coords = (np.arange(out_len, dtype=np.float64) + 0.5) / scale - 0.5
    first = np.floor(coords - radius + 1.0).astype(np.int64)
    taps = int(math.ceil(radius * 2.0))
    raw_idx = first[:, None] + np.arange(taps, dtype=np.int64)[None, :]
    idx = np.clip(raw_idx, 0, in_len - 1)

    dist = coords[:, None] - raw_idx.astype(np.float64)
    vector_kernel = np.vectorize(lambda d: _filter_kernel(float(d), kind), otypes=[np.float64])
    weights = vector_kernel(dist)

    # Boundary clamping duplicates the edge samples. Re-normalize every output
    # position so edge pixels retain unit DC gain and do not darken/brighten.
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights = np.divide(
        weights,
        weights_sum,
        out=np.zeros_like(weights),
        where=np.abs(weights_sum) > 1e-15,
    ).astype(np.float32)

    moved = np.moveaxis(arr, axis, -1)
    flat = moved.reshape(-1, in_len)
    gathered = flat[:, idx]                  # [N, out_len, taps]
    out = np.sum(gathered * weights[None, :, :], axis=-1)
    out = out.reshape(moved.shape[:-1] + (out_len,))
    return np.moveaxis(out, -1, axis).astype(np.float32, copy=False)


def _resize_chroma(chroma: torch.Tensor, size, kind: str) -> torch.Tensor:
    kind = kind.lower()
    if kind == "bilinear":
        return F.interpolate(chroma, size=size, mode="bilinear", align_corners=False)

    # Mitchell/Catmull-Rom/Lanczos are deliberately kept as deterministic CPU
    # reconstruction filters. Chroma is only 2 channels, so this remains much
    # cheaper than filtering the full RGB image.
    arr = chroma.detach().cpu().numpy().astype(np.float32, copy=False)
    arr = _resize_axis_np(arr, size[0], axis=2, kind=kind)
    arr = _resize_axis_np(arr, size[1], axis=3, kind=kind)
    return torch.from_numpy(arr)


class ArtCNNLumaUpscaleONNX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (_list_models(),),
                "chroma_model": (_list_models(include_none=True),),
                "chroma_resampler": (["Catmull-Rom", "Mitchell", "Lanczos", "Bilinear"], {"default": "Catmull-Rom"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    DESCRIPTION = (
        "ArtCNN ONNX luma upscaler with optional ArtCNN chroma reconstruction. "
        "GPU-only DirectML; no CPU fallback."
    )

    def upscale(self, image, model, chroma_model, chroma_resampler="Mitchell"):
        if model.startswith("<no "):
            raise ArtCNNONNXError(f"No ONNX models found in {_onnx_folder()}")

        luma_path = os.path.join(_onnx_folder(), model)
        if not os.path.isfile(luma_path):
            raise ArtCNNONNXError(f"Luma ONNX model not found: {luma_path}")

        luma_session = _get_session(luma_path)
        luma_in_ch, luma_out_ch = _model_io_channels(luma_session)

        chroma_session = None
        if chroma_model != "None":
            chroma_path = os.path.join(_onnx_folder(), chroma_model)
            if not os.path.isfile(chroma_path):
                raise ArtCNNONNXError(f"Chroma ONNX model not found: {chroma_path}")
            if os.path.abspath(chroma_path) == os.path.abspath(luma_path):
                raise ArtCNNONNXError("Luma and chroma models must be different files.")
            chroma_session = _get_session(chroma_path)
            _validate_chroma(chroma_session)

        if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
            raise ArtCNNONNXError(
                f"Expected IMAGE tensor [B,H,W,1/3/4], got {tuple(image.shape)}"
            )

        x = image.detach().float().clamp(0, 1)
        channels = x.shape[-1]
        if channels == 4:
            rgb = x[..., :3]
            alpha = x[..., 3:4]
        elif channels == 3:
            rgb = x
            alpha = None
        else:
            rgb = x.repeat(1, 1, 1, 3)
            alpha = None

        rgb_nchw = rgb.permute(0, 3, 1, 2).contiguous()

        # Native RGB ArtCNN model (e.g. R16F128_RGB): run RGB directly.
        if luma_in_ch == 3 and luma_out_ch == 3:
            if chroma_session is not None:
                raise ArtCNNONNXError(
                    "A 3->3 RGB ArtCNN model already processes RGB directly; "
                    "do not select a separate chroma model."
                )
            rgb_np = rgb_nchw.cpu().numpy().astype(np.float32, copy=False)
            rgb2_np = _run_fixed_batch(luma_session, rgb_np)
            if rgb2_np.ndim != 4 or rgb2_np.shape[1] != 3:
                raise ArtCNNONNXError(f"Unexpected RGB ArtCNN output shape: {rgb2_np.shape}")
            out_h, out_w = rgb2_np.shape[-2:]
            in_h, in_w = rgb_nchw.shape[-2:]
            scale_h = out_h / max(in_h, 1)
            scale_w = out_w / max(in_w, 1)
            if abs(scale_h - scale_w) > 1e-6:
                raise ArtCNNONNXError(
                    "RGB ArtCNN model produced a non-uniform scale: "
                    f"Input={in_h}x{in_w}, output={out_h}x{out_w} "
                    f"({scale_w:.3f}x{scale_h:.3f})"
                )
            if min(abs(scale_w - 2.0), abs(scale_w - 4.0)) > 1e-6:
                raise ArtCNNONNXError(
                    "This node supports 2x and 4x RGB ArtCNN models. "
                    f"Input={in_h}x{in_w}, output={out_h}x{out_w} "
                    f"({scale_w:.3f}x{scale_h:.3f})"
                )
            out = torch.from_numpy(rgb2_np).clamp(0, 1).permute(0, 2, 3, 1).contiguous()
            if alpha is not None:
                alpha_up = F.interpolate(
                    alpha.permute(0, 3, 1, 2),
                    size=(out_h, out_w),
                    mode="nearest",
                ).permute(0, 2, 3, 1)
                out = torch.cat((out, alpha_up), dim=-1)
            return (out,)

        if luma_in_ch != 1 or luma_out_ch != 1:
            raise ArtCNNONNXError(
                f"Selected model has unsupported channel layout {luma_in_ch}->{luma_out_ch}. "
                "This node supports 1->1 luma models and 3->3 RGB models."
            )

        _validate_luma(luma_session)

        # Convert to BT.709 YCbCr in NCHW.
        ycbcr = _rgb_to_ycbcr(rgb_nchw)
        y = ycbcr[:, 0:1]
        y_np = y.cpu().numpy().astype(np.float32, copy=False)
        y2_np = _run_luma(luma_session, y_np)
        y2 = torch.from_numpy(y2_np)

        in_h, in_w = y.shape[-2:]
        out_h, out_w = y2.shape[-2:]
        scale_h = out_h / max(in_h, 1)
        scale_w = out_w / max(in_w, 1)
        if abs(scale_h - scale_w) > 1e-6:
            raise ArtCNNONNXError(
                "Luma ArtCNN model produced a non-uniform scale: "
                f"Input={in_h}x{in_w}, output={out_h}x{out_w} "
                f"({scale_w:.3f}x{scale_h:.3f})"
            )
        if min(abs(scale_w - 2.0), abs(scale_w - 4.0)) > 1e-6:
            raise ArtCNNONNXError(
                "This node supports 2x and 4x luma ArtCNN models. "
                f"Input={in_h}x{in_w}, output={out_h}x{out_w} "
                f"({scale_w:.3f}x{scale_h:.3f})"
            )

        # ArtCNN chroma models explicitly require bilinear chroma pre-upscaling.
        # When no neural chroma model is selected, allow a conventional resampler.
        if chroma_session is not None:
            cbcr_up = _resize_chroma(ycbcr[:, 1:3], (out_h, out_w), "bilinear")
            chroma_in = torch.cat((y2, cbcr_up), dim=1)
            chroma_np = chroma_in.cpu().numpy().astype(np.float32, copy=False)
            chroma2_np = _run_chroma(chroma_session, chroma_np)
            chroma2 = torch.from_numpy(chroma2_np).clamp(0, 1)
        else:
            # For ordinary RGB input there is no subsampled chroma plane to
            # independently reconstruct.  Build the output chroma from a
            # spatially upscaled RGB reference at the exact output geometry,
            # then replace only its Y plane with ArtCNN's result. This keeps
            # chroma geometrically aligned with the ArtCNN luma and avoids
            # color casts from independently resampling Cb/Cr.
            if chroma_resampler.lower() == "bilinear":
                rgb_up = F.interpolate(
                    rgb_nchw, size=(out_h, out_w), mode="bilinear", align_corners=False
                )
            else:
                # Reuse the deterministic reconstruction kernels for each RGB
                # channel. This is slower than bilinear but preserves the
                # requested filter without converting color planes separately.
                rgb_up_np = rgb_nchw.detach().cpu().numpy().astype(np.float32, copy=False)
                rgb_up_np = _resize_axis_np(rgb_up_np, out_h, axis=2, kind=chroma_resampler.lower())
                rgb_up_np = _resize_axis_np(rgb_up_np, out_w, axis=3, kind=chroma_resampler.lower())
                rgb_up = torch.from_numpy(rgb_up_np)
            chroma2 = _rgb_to_ycbcr(rgb_up)[:, 1:3].clamp(0, 1)

        out_ycbcr = torch.cat((y2.clamp(0, 1), chroma2), dim=1)
        out = _ycbcr_to_rgb(out_ycbcr).permute(0, 2, 3, 1).contiguous()

        if alpha is not None:
            alpha_up = F.interpolate(
                alpha.permute(0, 3, 1, 2),
                size=(out_h, out_w),
                mode="nearest",
            ).permute(0, 2, 3, 1)
            out = torch.cat((out, alpha_up), dim=-1)

        return (out,)


NODE_CLASS_MAPPINGS = {
    "ArtCNNLumaUpscaleONNX": ArtCNNLumaUpscaleONNX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArtCNNLumaUpscaleONNX": "ArtCNN Luma Upscale (ONNX / DirectML)",
}
