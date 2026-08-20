import os
import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort


class ArtCNNONNXError(RuntimeError):
    pass


_SESSION_CACHE: Dict[str, ort.InferenceSession] = {}
_SESSION_INFO_CACHE: Dict[str, Tuple] = {}


def _onnx_folder() -> str:
    node_dir = os.path.dirname(os.path.abspath(__file__))
    comfy_root = os.path.abspath(os.path.join(node_dir, os.pardir, os.pardir))
    return os.path.join(comfy_root, "models", "onnx", "Upscale")


def _list_models(include_none=False):
    folder = _onnx_folder()
    os.makedirs(folder, exist_ok=True)

    models = sorted(
        f
        for f in os.listdir(folder)
        if f.lower().endswith(".onnx")
        and os.path.isfile(os.path.join(folder, f))
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

    # Keep the proven original execution configuration.
    # Do not force alternate memory or graph settings here.
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

    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]

    _SESSION_CACHE[key] = session
    _SESSION_INFO_CACHE[key] = (
        inp.name,
        inp.type,
        inp.shape,
        out.name,
        out.type,
        out.shape,
    )

    return session


def _session_info(session):
    for path, cached_session in _SESSION_CACHE.items():
        if cached_session is session:
            return _SESSION_INFO_CACHE[path]

    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]

    return (
        inp.name,
        inp.type,
        inp.shape,
        out.name,
        out.type,
        out.shape,
    )


def _shape_dim_is_int(value):
    return isinstance(value, (int, np.integer))


def _model_io_channels(session):
    _, _, input_shape, _, _, output_shape = _session_info(session)

    if len(input_shape) != 4 or len(output_shape) != 4:
        raise ArtCNNONNXError(
            "ArtCNN model must have 4D input/output, "
            f"got input={input_shape}, output={output_shape}"
        )

    input_channels = (
        input_shape[1]
        if _shape_dim_is_int(input_shape[1])
        else None
    )
    output_channels = (
        output_shape[1]
        if _shape_dim_is_int(output_shape[1])
        else None
    )

    if input_channels is None or output_channels is None:
        raise ArtCNNONNXError(
            "Channel dimensions must be static for this node, "
            f"got input={input_shape}, output={output_shape}"
        )

    return int(input_channels), int(output_channels)


def _validate_luma_model(session):
    input_channels, output_channels = _model_io_channels(session)

    if input_channels != 1 or output_channels != 1:
        raise ArtCNNONNXError(
            "Selected model is not a 1->1 luma model: "
            f"input={input_channels} channels, "
            f"output={output_channels} channels"
        )


def _validate_chroma_model(session):
    _, _, input_shape, _, _, output_shape = _session_info(session)

    if len(input_shape) != 4 or len(output_shape) != 4:
        raise ArtCNNONNXError(
            "Chroma model must be 4D input/output, "
            f"got input={input_shape}, output={output_shape}"
        )

    if (
        _shape_dim_is_int(input_shape[1])
        and input_shape[1] != 3
    ):
        raise ArtCNNONNXError(
            "The chroma path expects a 3-channel Y/Cb/Cr input model, "
            f"got {input_shape}"
        )

    if (
        _shape_dim_is_int(output_shape[1])
        and output_shape[1] != 2
    ):
        raise ArtCNNONNXError(
            "The chroma path expects a 2-channel Cb/Cr output model, "
            f"got {output_shape}"
        )


def _prepare_onnx_input(session, array: np.ndarray) -> np.ndarray:
    """
    Match the ONNX model's declared floating-point input type.

    FP32 model -> FP32 input
    FP16 model -> FP16 input
    """
    _, input_type, _, _, _, _ = _session_info(session)

    if input_type == "tensor(float16)":
        return array.astype(np.float16, copy=False)

    if input_type == "tensor(float)":
        return array.astype(np.float32, copy=False)

    raise ArtCNNONNXError(
        f"Unsupported ONNX input type: {input_type}. "
        "This node supports tensor(float) and tensor(float16)."
    )


def _normalize_onnx_output(output: np.ndarray) -> np.ndarray:
    if output.dtype != np.float32:
        return output.astype(np.float32, copy=False)

    return output


def _run_fixed_batch(session, array: np.ndarray) -> np.ndarray:
    if array.ndim != 4:
        raise ArtCNNONNXError(
            f"Expected NCHW input, got {array.shape}"
        )

    input_name, _, input_shape, _, _, _ = _session_info(session)
    static_batch = input_shape[0]

    array = _prepare_onnx_input(session, array)

    if (
        _shape_dim_is_int(static_batch)
        and static_batch == 1
        and array.shape[0] != 1
    ):
        outputs = []

        for index in range(array.shape[0]):
            output = session.run(
                None,
                {input_name: array[index:index + 1]},
            )[0]

            outputs.append(
                _normalize_onnx_output(output)
            )

        return np.concatenate(outputs, axis=0)

    output = session.run(
        None,
        {input_name: array},
    )[0]

    return _normalize_onnx_output(output)


def _run_primary_model(session, array: np.ndarray) -> np.ndarray:
    output = _run_fixed_batch(session, array)

    if output.ndim != 4:
        raise ArtCNNONNXError(
            f"Unexpected primary model output shape: {output.shape}"
        )

    return output


def _run_chroma_model(session, array: np.ndarray) -> np.ndarray:
    output = _run_fixed_batch(session, array)

    if output.ndim != 4 or output.shape[1] != 2:
        raise ArtCNNONNXError(
            f"Unexpected chroma output shape: {output.shape}"
        )

    return output


# -------------------------------------------------------------------------
# BT.709 conversion used by the luma/chroma pipeline.
#
# RGB 3->3 models bypass this completely.
# -------------------------------------------------------------------------


def _rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.clamp(0, 1)

    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]

    y = (
        0.2126 * r
        + 0.7152 * g
        + 0.0722 * b
    )

    cb = (
        -0.114572 * r
        - 0.385428 * g
        + 0.5 * b
        + 0.5
    )

    cr = (
        0.5 * r
        - 0.454153 * g
        - 0.045847 * b
        + 0.5
    )

    return torch.cat(
        (y, cb, cr),
        dim=1,
    ).clamp(0, 1)


def _ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    ycbcr = ycbcr.clamp(0, 1)

    y = ycbcr[:, 0:1]
    cb = ycbcr[:, 1:2] - 0.5
    cr = ycbcr[:, 2:3] - 0.5

    r = y + 1.5748 * cr
    g = y - 0.1873 * cb - 0.4681 * cr
    b = y + 1.8556 * cb

    return torch.cat(
        (r, g, b),
        dim=1,
    ).clamp(0, 1)


# -------------------------------------------------------------------------
# Chroma resampling
# -------------------------------------------------------------------------


def _sinc(x):
    if x == 0.0:
        return 1.0

    pix = math.pi * x
    return math.sin(pix) / pix


def _filter_kernel(x, kind):
    if kind == "bilinear":
        ax = abs(x)
        return max(0.0, 1.0 - ax)

    if kind == "catmull-rom":
        ax = abs(x)

        if ax < 1.0:
            return 1.5 * ax**3 - 2.5 * ax**2 + 1.0

        if ax < 2.0:
            return -0.5 * ax**3 + 2.5 * ax**2 - 4.0 * ax + 2.0

        return 0.0

    if kind == "mitchell":
        B = C = 1.0 / 3.0
        ax = abs(x)

        if ax < 1.0:
            return (
                (12 - 9 * B - 6 * C) * ax**3
                + (-18 + 12 * B + 6 * C) * ax**2
                + (6 - 2 * B)
            ) / 6.0

        if ax < 2.0:
            return (
                (-B - 6 * C) * ax**3
                + (6 * B + 30 * C) * ax**2
                + (-12 * B - 48 * C) * ax
                + (8 * B + 24 * C)
            ) / 6.0

        return 0.0

    if kind == "lanczos":
        if abs(x) >= 3.0:
            return 0.0

        return _sinc(x) * _sinc(x / 3.0)

    if kind == "lanczos2":
        if abs(x) >= 2.0:
            return 0.0

        return _sinc(x) * _sinc(x / 2.0)

    if kind == "lanczos2-sharp":
        blur = 0.9549963639785485
        radius = 2.0 * blur

        if abs(x) >= radius:
            return 0.0

        return _sinc(x / blur) * _sinc(x / radius)

    raise ArtCNNONNXError(
        f"Unknown chroma resampler: {kind}"
    )


def _normalize_resampler(kind: str) -> str:
    return kind.strip().lower().replace(" ", "-")


def _resize_axis_np(
    array,
    output_length,
    axis,
    kind,
    anti_ring=0.0,
):
    input_length = int(array.shape[axis])
    output_length = int(output_length)

    if input_length == output_length:
        return array.astype(np.float32, copy=False)

    radius = {
        "bilinear": 1.0,
        "catmull-rom": 2.0,
        "mitchell": 2.0,
        "lanczos": 3.0,
        "lanczos2": 2.0,
        "lanczos2-sharp": 2.0 * 0.9549963639785485,
    }[kind]

    scale = output_length / input_length

    coords = (
        (np.arange(output_length, dtype=np.float64) + 0.5) / scale
        - 0.5
    )

    first = np.floor(
        coords - radius + 1.0
    ).astype(np.int64)

    taps = int(math.ceil(radius * 2.0))

    raw_indices = (
        first[:, None]
        + np.arange(taps, dtype=np.int64)[None, :]
    )

    indices = np.clip(
        raw_indices,
        0,
        input_length - 1,
    )

    distances = (
        coords[:, None]
        - raw_indices.astype(np.float64)
    )

    vector_kernel = np.vectorize(
        lambda distance: _filter_kernel(
            float(distance),
            kind,
        ),
        otypes=[np.float64],
    )

    weights = vector_kernel(distances)
    weights_sum = weights.sum(axis=1, keepdims=True)

    weights = np.divide(
        weights,
        weights_sum,
        out=np.zeros_like(weights),
        where=np.abs(weights_sum) > 1e-15,
    ).astype(np.float32)

    moved = np.moveaxis(array, axis, -1)
    flat = moved.reshape(-1, input_length)
    gathered = flat[:, indices]

    output = np.sum(
        gathered * weights[None, :, :],
        axis=-1,
    )

    if anti_ring > 0.0:
        local_min = gathered.min(axis=-1)
        local_max = gathered.max(axis=-1)
        limited = np.clip(output, local_min, local_max)
        output += (limited - output) * anti_ring

    output = output.reshape(
        moved.shape[:-1] + (output_length,)
    )

    return np.moveaxis(
        output,
        -1,
        axis,
    ).astype(
        np.float32,
        copy=False,
    )


def _resize_chroma(
    chroma: torch.Tensor,
    size,
    kind: str,
) -> torch.Tensor:
    kind = _normalize_resampler(kind)

    if kind == "bilinear":
        return F.interpolate(
            chroma,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    anti_ring_total = 0.22 if kind == "lanczos2-sharp" else 0.0

    anti_ring_pass = (
        1.0 - math.sqrt(1.0 - anti_ring_total)
        if anti_ring_total > 0.0
        else 0.0
    )

    array = (
        chroma.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    array = _resize_axis_np(
        array,
        size[0],
        axis=2,
        kind=kind,
        anti_ring=anti_ring_pass,
    )

    array = _resize_axis_np(
        array,
        size[1],
        axis=3,
        kind=kind,
        anti_ring=anti_ring_pass,
    )

    return torch.from_numpy(array)


# -------------------------------------------------------------------------
# ComfyUI node
# -------------------------------------------------------------------------


class ArtCNNUpscaleONNX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (_list_models(),),
                "chroma_model": (
                    _list_models(include_none=True),
                ),
                "chroma_resampler": (
                    [
                        "Catmull-Rom",
                        "Mitchell",
                        "Lanczos",
                        "Lanczos2",
                        "Lanczos2 Sharp",
                        "Bilinear",
                    ],
                    {"default": "Catmull-Rom"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    DESCRIPTION = (
        "ArtCNN ONNX upscaler with RGB, luma, and optional "
        "chroma reconstruction using DirectML."
    )

    def upscale(
        self,
        image,
        model,
        chroma_model,
        chroma_resampler="Catmull-Rom",
    ):
        if model.startswith("<no "):
            raise ArtCNNONNXError(
                f"No ONNX models found in {_onnx_folder()}"
            )

        model_path = os.path.join(
            _onnx_folder(),
            model,
        )

        if not os.path.isfile(model_path):
            raise ArtCNNONNXError(
                f"Model not found: {model_path}"
            )

        model_session = _get_session(model_path)
        model_input_channels, model_output_channels = _model_io_channels(
            model_session
        )

        chroma_session = None

        if chroma_model != "None":
            chroma_path = os.path.join(
                _onnx_folder(),
                chroma_model,
            )

            if not os.path.isfile(chroma_path):
                raise ArtCNNONNXError(
                    f"Chroma model not found: {chroma_path}"
                )

            if os.path.abspath(chroma_path) == os.path.abspath(model_path):
                raise ArtCNNONNXError(
                    "Primary and chroma models must be different files."
                )

            chroma_session = _get_session(chroma_path)
            _validate_chroma_model(chroma_session)

        if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
            raise ArtCNNONNXError(
                "Expected IMAGE tensor [B,H,W,1/3/4], "
                f"got {tuple(image.shape)}"
            )

        image = image.detach().float().clamp(0, 1)

        if image.shape[-1] == 4:
            rgb = image[..., :3]
            alpha = image[..., 3:4]
        elif image.shape[-1] == 3:
            rgb = image
            alpha = None
        else:
            rgb = image.repeat(1, 1, 1, 3)
            alpha = None

        rgb_nchw = rgb.permute(0, 3, 1, 2).contiguous()

        # -----------------------------------------------------------------
        # RGB 3 -> 3
        # -----------------------------------------------------------------

        if (
            model_input_channels == 3
            and model_output_channels == 3
        ):
            if chroma_session is not None:
                raise ArtCNNONNXError(
                    "A 3->3 RGB model already processes RGB directly; "
                    "do not select a separate chroma model."
                )

            rgb_np = (
                rgb_nchw.cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

            rgb_output_np = _run_primary_model(
                model_session,
                rgb_np,
            )

            if (
                rgb_output_np.ndim != 4
                or rgb_output_np.shape[1] != 3
            ):
                raise ArtCNNONNXError(
                    "Unexpected RGB model output shape: "
                    f"{rgb_output_np.shape}"
                )

            output_height, output_width = rgb_output_np.shape[-2:]
            input_height, input_width = rgb_nchw.shape[-2:]

            scale_height = output_height / max(input_height, 1)
            scale_width = output_width / max(input_width, 1)

            if abs(scale_height - scale_width) > 1e-6:
                raise ArtCNNONNXError(
                    "RGB model produced a non-uniform scale: "
                    f"input={input_height}x{input_width}, "
                    f"output={output_height}x{output_width}"
                )

            if min(
                abs(scale_width - 2.0),
                abs(scale_width - 4.0),
            ) > 1e-6:
                raise ArtCNNONNXError(
                    "This node supports 2x and 4x RGB ArtCNN models. "
                    f"Input={input_height}x{input_width}, "
                    f"output={output_height}x{output_width} "
                    f"({scale_width:.3f}x)"
                )

            output = (
                torch.from_numpy(rgb_output_np)
                .clamp(0, 1)
                .permute(0, 2, 3, 1)
                .contiguous()
            )

            if alpha is not None:
                alpha_up = (
                    F.interpolate(
                        alpha.permute(0, 3, 1, 2),
                        size=(output_height, output_width),
                        mode="nearest",
                    )
                    .permute(0, 2, 3, 1)
                )

                output = torch.cat(
                    (output, alpha_up),
                    dim=-1,
                )

            return (output,)

        # -----------------------------------------------------------------
        # Luma 1 -> 1
        # -----------------------------------------------------------------

        if (
            model_input_channels != 1
            or model_output_channels != 1
        ):
            raise ArtCNNONNXError(
                "Selected primary model has unsupported channel layout "
                f"{model_input_channels}->{model_output_channels}. "
                "The primary model must be either 1->1 luma or 3->3 RGB."
            )

        _validate_luma_model(model_session)

        ycbcr = _rgb_to_ycbcr(rgb_nchw)
        y = ycbcr[:, 0:1]

        y_np = (
            y.cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        luma_output_np = _run_primary_model(
            model_session,
            y_np,
        )

        if (
            luma_output_np.ndim != 4
            or luma_output_np.shape[1] != 1
        ):
            raise ArtCNNONNXError(
                "Unexpected luma model output shape: "
                f"{luma_output_np.shape}"
            )

        luma_output = torch.from_numpy(luma_output_np)

        input_height, input_width = y.shape[-2:]
        output_height, output_width = luma_output.shape[-2:]

        scale_height = output_height / max(input_height, 1)
        scale_width = output_width / max(input_width, 1)

        if abs(scale_height - scale_width) > 1e-6:
            raise ArtCNNONNXError(
                "Luma model produced a non-uniform scale: "
                f"input={input_height}x{input_width}, "
                f"output={output_height}x{output_width}"
            )

        if min(
            abs(scale_width - 2.0),
            abs(scale_width - 4.0),
        ) > 1e-6:
            raise ArtCNNONNXError(
                "This node supports 2x and 4x luma ArtCNN models. "
                f"Input={input_height}x{input_width}, "
                f"output={output_height}x{output_width} "
                f"({scale_width:.3f}x)"
            )

        # -----------------------------------------------------------------
        # Neural chroma reconstruction
        # -----------------------------------------------------------------

        if chroma_session is not None:
            chroma_up = _resize_chroma(
                ycbcr[:, 1:3],
                (output_height, output_width),
                "bilinear",
            )

            if chroma_up.shape[-2:] != luma_output.shape[-2:]:
                raise ArtCNNONNXError(
                    "Luma output and chroma input dimensions do not match: "
                    f"luma={tuple(luma_output.shape[-2:])}, "
                    f"chroma={tuple(chroma_up.shape[-2:])}"
                )

            chroma_input = torch.cat(
                (luma_output, chroma_up),
                dim=1,
            )

            chroma_input_np = (
                chroma_input.cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

            chroma_output_np = _run_chroma_model(
                chroma_session,
                chroma_input_np,
            )

            chroma_output = (
                torch.from_numpy(chroma_output_np)
                .clamp(0, 1)
            )

        # -----------------------------------------------------------------
        # Non-neural chroma reconstruction
        # -----------------------------------------------------------------

        else:
            chroma_output = _resize_chroma(
                ycbcr[:, 1:3],
                (output_height, output_width),
                chroma_resampler,
            ).clamp(0, 1)

        # -----------------------------------------------------------------
        # YCbCr -> RGB
        # -----------------------------------------------------------------

        output_ycbcr = torch.cat(
            (
                luma_output.clamp(0, 1),
                chroma_output,
            ),
            dim=1,
        )

        output = (
            _ycbcr_to_rgb(output_ycbcr)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        if alpha is not None:
            alpha_up = (
                F.interpolate(
                    alpha.permute(0, 3, 1, 2),
                    size=(output_height, output_width),
                    mode="nearest",
                )
                .permute(0, 2, 3, 1)
            )

            output = torch.cat(
                (output, alpha_up),
                dim=-1,
            )

        return (output,)


NODE_CLASS_MAPPINGS = {
    "ArtCNNUpscaleONNX": ArtCNNUpscaleONNX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArtCNNUpscaleONNX": "ArtCNN Upscale (ONNX / DirectML)",
}
