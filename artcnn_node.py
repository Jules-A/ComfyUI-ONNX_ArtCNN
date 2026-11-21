import gc
import os
import re
import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort


class ArtCNNONNXError(RuntimeError):
    pass


# Keyed by (absolute model path, ORT provider name, device id).
_SESSION_CACHE: Dict[Tuple, ort.InferenceSession] = {}
_SESSION_INFO_CACHE: Dict[Tuple, Tuple] = {}


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


_PROVIDER_CHOICES = [
    "DirectML",
    "ROCm",
    "MIGraphX",
    "CUDA",
    "TensorRT",
    "CPU",
    "Auto",
]

_PROVIDER_ORT_NAMES = {
    "DirectML": "DmlExecutionProvider",
    "ROCm": "ROCMExecutionProvider",
    "MIGraphX": "MIGraphXExecutionProvider",
    "CUDA": "CUDAExecutionProvider",
    "TensorRT": "TensorrtExecutionProvider",
    "CPU": "CPUExecutionProvider",
}

# "Auto" skips TensorRT: it builds an engine on first use, which is slow.
_AUTO_ORDER = ["DirectML", "MIGraphX", "ROCm", "CUDA", "CPU"]

_PROVIDER_HINTS = {
    "DirectML": (
        "install onnxruntime-directml "
        "(uninstall other onnxruntime variants first)"
    ),
    "ROCm": "install an onnxruntime build that includes the ROCm provider",
    "MIGraphX": (
        "install an onnxruntime build that includes the MIGraphX provider"
    ),
    "CUDA": "install onnxruntime-gpu",
    "TensorRT": "install onnxruntime-gpu plus the NVIDIA TensorRT libraries",
    "CPU": "",
}


def _available_provider(ort_name: str):
    # Case-insensitive: provider names are spelled slightly differently
    # between onnxruntime builds and READMEs (e.g. ROCM vs ROCm).
    for name in ort.get_available_providers():
        if name.lower() == ort_name.lower():
            return name

    return None


def _resolve_provider(choice: str) -> str:
    """Map a UI choice to the exact provider name this onnxruntime exposes."""
    if choice == "Auto":
        for label in _AUTO_ORDER:
            actual = _available_provider(_PROVIDER_ORT_NAMES[label])

            if actual is not None:
                return actual

        raise ArtCNNONNXError(
            "No usable ONNX Runtime provider found. "
            f"Available providers: {ort.get_available_providers()}"
        )

    if choice not in _PROVIDER_ORT_NAMES:
        raise ArtCNNONNXError(f"Unknown provider: {choice}")

    actual = _available_provider(_PROVIDER_ORT_NAMES[choice])

    if actual is None:
        hint = _PROVIDER_HINTS.get(choice, "")
        raise ArtCNNONNXError(
            f"The {choice} provider is not available in this onnxruntime "
            f"build. Available providers: {ort.get_available_providers()}."
            + (f" To use it, {hint}." if hint else "")
        )

    return actual


def _cache_key(path: str, provider_name: str, device_id: int):
    return (os.path.abspath(path), provider_name, int(device_id))


def _prune_cache(keep) -> None:
    """
    Drop every cached session whose key is not in `keep`, so GPU memory held
    by models that are no longer selected is released instead of piling up.
    """
    stale = [key for key in _SESSION_CACHE if key not in keep]

    if not stale:
        return

    for key in stale:
        _SESSION_CACHE.pop(key, None)
        _SESSION_INFO_CACHE.pop(key, None)

    gc.collect()


def _get_session(
    path: str,
    provider_name: str = "DmlExecutionProvider",
    device_id: int = 0,
):
    device_id = int(device_id)
    key = _cache_key(path, provider_name, device_id)

    if key in _SESSION_CACHE:
        return _SESSION_CACHE[key]

    lname = provider_name.lower()
    so = ort.SessionOptions()

    if lname == "dmlexecutionprovider":
        # Keep the proven original execution configuration.
        # Do not force alternate memory or graph settings here.
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        if device_id == 0:
            providers = [provider_name]
        else:
            providers = [(provider_name, {"device_id": device_id})]
    elif lname == "tensorrtexecutionprovider":
        cache_dir = os.path.join(_onnx_folder(), ".trt_cache")
        os.makedirs(cache_dir, exist_ok=True)

        providers = [
            (
                provider_name,
                {
                    "device_id": device_id,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": cache_dir,
                },
            )
        ]

        cuda = _available_provider("CUDAExecutionProvider")

        if cuda is not None:
            providers.append((cuda, {"device_id": device_id}))
    elif lname == "cpuexecutionprovider":
        providers = [provider_name]
    else:
        providers = [(provider_name, {"device_id": device_id})]

    session = ort.InferenceSession(
        key[0],
        sess_options=so,
        providers=providers,
    )

    active = session.get_providers()

    if not active or active[0].lower() != lname:
        del session
        gc.collect()

        raise ArtCNNONNXError(
            f"{provider_name} was requested but is not the active execution "
            f"provider. Session providers: {active}"
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


def _dim_multiple(dim) -> int:
    """A symbolic dim such as '8*hb' means the size must be a multiple of 8."""
    if not isinstance(dim, str):
        return 1

    match = re.match(r"^\s*\(?\s*(\d+)\s*\*\s*[A-Za-z_]\w*\s*\)?\s*$", dim)

    return int(match.group(1)) if match else 1


def _effective_multiple(session, pad_multiple: int = 0) -> int:
    """Size multiple the model needs (from its input dims) combined with the
    optional manual `pad_multiple` override."""
    _, _, input_shape, _, _, _ = _session_info(session)

    multiple = 1

    if len(input_shape) == 4:
        multiple = math.lcm(
            _dim_multiple(input_shape[2]),
            _dim_multiple(input_shape[3]),
        )

    if pad_multiple and int(pad_multiple) > 1:
        multiple = math.lcm(multiple, int(pad_multiple))

    return multiple


def _is_multiframe(input_channels: int, output_channels: int) -> bool:
    """N stacked RGB frames (N odd, >= 3) in, one RGB frame out."""
    return (
        output_channels == 3
        and input_channels > 3
        and input_channels % 3 == 0
        and (input_channels // 3) % 2 == 1
    )


def _run_tiled(
    session,
    array: np.ndarray,
    overlap: int = 16,
    tile_override=None,
) -> np.ndarray:
    """
    Run a model over an arbitrary image using overlapping tiles with linear
    blending. Used automatically when the model's H/W are static (e.g. a
    Graphene export with a [1,C,T,T] input), or on request for models with
    dynamic H/W via `tile_override` (square tile size, e.g. to bound memory).
    """
    input_name, input_type, input_shape, _, _, output_shape = _session_info(
        session
    )

    if input_type not in ("tensor(float)", "tensor(float16)"):
        raise ArtCNNONNXError(
            f"Unsupported ONNX input type: {input_type}. "
            "This node supports tensor(float) and tensor(float16)."
        )

    dtype = np.float16 if input_type == "tensor(float16)" else np.float32

    if tile_override is None:
        if not all(_shape_dim_is_int(v) for v in output_shape[1:]):
            raise ArtCNNONNXError(
                "Tiled models need a fully static output shape, "
                f"got {output_shape}"
            )

        tile_h, tile_w = int(input_shape[2]), int(input_shape[3])
        out_channels = int(output_shape[1])
        scale = int(output_shape[2]) // tile_h

        if (
            scale < 1
            or int(output_shape[2]) != tile_h * scale
            or int(output_shape[3]) != tile_w * scale
        ):
            raise ArtCNNONNXError(
                "Could not infer an integer scale from "
                f"input={input_shape}, output={output_shape}"
            )
    else:
        tile_h = tile_w = int(tile_override)

        if not _shape_dim_is_int(output_shape[1]):
            raise ArtCNNONNXError(
                "Tiling needs a static output channel count, "
                f"got {output_shape}"
            )

        out_channels = int(output_shape[1])

        # Learn the scale by running one tile of zeros.
        probe = np.zeros(
            (1, int(array.shape[1]), tile_h, tile_w),
            dtype=dtype,
        )
        probe_out = session.run(None, {input_name: probe})[0]
        scale = int(probe_out.shape[-2]) // tile_h

        if (
            scale < 1
            or int(probe_out.shape[-2]) != tile_h * scale
            or int(probe_out.shape[-1]) != tile_w * scale
        ):
            raise ArtCNNONNXError(
                "Could not infer an integer scale from a "
                f"{tile_h}x{tile_w} probe tile: output={probe_out.shape}"
            )

    overlap = int(overlap)

    if overlap < 0 or overlap >= min(tile_h, tile_w):
        raise ArtCNNONNXError(
            f"tile_overlap must be in [0, {min(tile_h, tile_w) - 1}] "
            f"for a {tile_h}x{tile_w} tile, got {overlap}"
        )

    dtype = np.float16 if input_type == "tensor(float16)" else np.float32
    array = np.asarray(array, dtype=np.float32)
    batch, _, height, width = array.shape

    step_h = tile_h - overlap
    step_w = tile_w - overlap

    pad_h = (-(height - tile_h) % step_h) if height > tile_h else tile_h - height
    pad_w = (-(width - tile_w) % step_w) if width > tile_w else tile_w - width

    def _ramp(length):
        r = np.minimum(
            np.arange(1, length + 1),
            np.arange(length, 0, -1),
        ).astype(np.float32)

        return np.minimum(r / (overlap * scale / 2.0 + 1.0), 1.0)

    window = (
        np.outer(_ramp(tile_h * scale), _ramp(tile_w * scale))[None]
        + 1e-3
    ).astype(np.float32)

    ys = None
    xs = None
    pbar = None
    results = []

    for index in range(batch):
        padding = ((0, 0), (0, pad_h), (0, pad_w))

        try:
            padded = np.pad(array[index], padding, mode="reflect")
        except ValueError:
            padded = np.pad(array[index], padding, mode="edge")

        padded_h, padded_w = padded.shape[-2:]

        if ys is None:
            ys = range(0, padded_h - tile_h + 1, step_h)
            xs = range(0, padded_w - tile_w + 1, step_w)

            try:
                import comfy.utils

                pbar = comfy.utils.ProgressBar(len(ys) * len(xs) * batch)
            except Exception:
                pbar = None

        out = np.zeros(
            (out_channels, padded_h * scale, padded_w * scale),
            np.float32,
        )
        weight = np.zeros(
            (1, padded_h * scale, padded_w * scale),
            np.float32,
        )

        for y in ys:
            for x in xs:
                tile = padded[None, :, y:y + tile_h, x:x + tile_w]
                tile = np.ascontiguousarray(tile).astype(dtype, copy=False)

                sr = session.run(None, {input_name: tile})[0][0]
                sr = sr.astype(np.float32, copy=False)

                oy, ox = y * scale, x * scale
                out[:, oy:oy + tile_h * scale, ox:ox + tile_w * scale] += (
                    sr * window
                )
                weight[:, oy:oy + tile_h * scale, ox:ox + tile_w * scale] += (
                    window
                )

                if pbar is not None:
                    pbar.update(1)

        results.append((out / weight)[:, :height * scale, :width * scale])

    return np.stack(results, axis=0).astype(np.float32, copy=False)


def _run_fixed_batch(
    session,
    array: np.ndarray,
    overlap: int = 16,
    pad_multiple: int = 0,
    tile_size: int = 0,
    _padded: bool = False,
) -> np.ndarray:
    if array.ndim != 4:
        raise ArtCNNONNXError(
            f"Expected NCHW input, got {array.shape}"
        )

    input_name, _, input_shape, _, _, _ = _session_info(session)
    static_batch = input_shape[0]

    # Static H/W (e.g. a fixed-tile export): process with overlapping tiles.
    if _shape_dim_is_int(input_shape[2]) and _shape_dim_is_int(input_shape[3]):
        return _run_tiled(session, array, overlap)

    multiple = _effective_multiple(session, pad_multiple)

    # Optional tiling for dynamic-size models (bounds memory on big images).
    if tile_size and int(tile_size) > 0:
        tile = int(tile_size)

        if multiple > 1:
            tile = -(-tile // multiple) * multiple

        return _run_tiled(session, array, overlap, tile_override=tile)

    # Models whose input dims are like '8*hb' need sizes that are multiples
    # of 8: pad (reflect), run, then crop the output back.
    if multiple > 1 and not _padded:
        height, width = array.shape[-2:]
        pad_h = (-height) % multiple
        pad_w = (-width) % multiple

        if pad_h or pad_w:
            padding = ((0, 0), (0, 0), (0, pad_h), (0, pad_w))

            try:
                padded = np.pad(array, padding, mode="reflect")
            except ValueError:
                padded = np.pad(array, padding, mode="edge")

            out = _run_fixed_batch(
                session,
                padded,
                overlap,
                pad_multiple,
                0,
                _padded=True,
            )
            scale = out.shape[-2] // padded.shape[-2]

            return np.ascontiguousarray(
                out[..., : height * scale, : width * scale]
            )

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


def _run_primary_model(
    session,
    array: np.ndarray,
    overlap: int = 16,
    pad_multiple: int = 0,
    tile_size: int = 0,
) -> np.ndarray:
    output = _run_fixed_batch(session, array, overlap, pad_multiple, tile_size)

    if output.ndim != 4:
        raise ArtCNNONNXError(
            f"Unexpected primary model output shape: {output.shape}"
        )

    return output


def _run_chroma_model(
    session,
    array: np.ndarray,
    overlap: int = 16,
    pad_multiple: int = 0,
    tile_size: int = 0,
) -> np.ndarray:
    output = _run_fixed_batch(session, array, overlap, pad_multiple, tile_size)

    if output.ndim != 4 or output.shape[1] != 2:
        raise ArtCNNONNXError(
            f"Unexpected chroma output shape: {output.shape}"
        )

    return output


def _run_multiframe(
    session,
    frames: np.ndarray,
    num_frames: int,
    overlap: int = 16,
    pad_multiple: int = 0,
    tile_size: int = 0,
) -> np.ndarray:
    """
    Temporal models that take `num_frames` stacked RGB frames
    (channels = frame-major RGB, window centred on the target frame) and
    return one upscaled RGB frame. `frames` is the IMAGE batch as (T,3,H,W);
    every frame is upscaled using its neighbours, repeating the first/last
    frame at the ends. A single image gives a window of identical frames.
    """
    total = int(frames.shape[0])
    half = num_frames // 2
    outputs = []

    for t in range(total):
        indices = [
            min(max(t + offset, 0), total - 1)
            for offset in range(-half, half + 1)
        ]
        window = np.concatenate([frames[i] for i in indices], axis=0)[None]
        out = _run_primary_model(
            session,
            window,
            overlap,
            pad_multiple,
            tile_size,
        )
        outputs.append(out[0])

    return np.stack(outputs, axis=0)


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
            "optional": {
                # Only used by fixed-tile models (e.g. Graphene exports).
                "tile_overlap": (
                    "INT",
                    {"default": 16, "min": 0, "max": 64, "step": 2},
                ),
                # Execution backend. DirectML is the default (original
                # behaviour); the others need a matching onnxruntime build.
                "provider": (
                    _PROVIDER_CHOICES,
                    {"default": "DirectML"},
                ),
                "device_id": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16},
                ),
                # Free the model's GPU memory as soon as the node finishes.
                "unload_after_run": (
                    "BOOLEAN",
                    {"default": False},
                ),
                # 0 = automatic (read from the model's input dims, e.g.
                # '8*hb' -> 8). Set manually for models that need padding
                # but don't declare it.
                "pad_multiple": (
                    "INT",
                    {"default": 0, "min": 0, "max": 128},
                ),
                # 0 = off. For dynamic-size models: process in square tiles
                # of this size (e.g. 256) to bound memory on large images.
                "tile_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 8},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    DESCRIPTION = (
        "ArtCNN ONNX upscaler with RGB, luma, and optional "
        "chroma reconstruction, plus RGB models up to 8x and multi-frame "
        "(N stacked frames -> 1) models. DirectML by default; ROCm, "
        "MIGraphX, CUDA, TensorRT or CPU via the provider option."
    )

    def upscale(
        self,
        image,
        model,
        chroma_model,
        chroma_resampler="Catmull-Rom",
        tile_overlap=16,
        provider="DirectML",
        device_id=0,
        unload_after_run=False,
        pad_multiple=0,
        tile_size=0,
    ):
        provider_name = _resolve_provider(provider)
        device_id = int(device_id)

        # Release sessions for models that are no longer selected *before*
        # creating a new one, so old and new models are never resident at once.
        keep = set()

        if not model.startswith("<no "):
            keep.add(
                _cache_key(
                    os.path.join(_onnx_folder(), model),
                    provider_name,
                    device_id,
                )
            )

            if chroma_model != "None":
                keep.add(
                    _cache_key(
                        os.path.join(_onnx_folder(), chroma_model),
                        provider_name,
                        device_id,
                    )
                )

        _prune_cache(keep)

        try:
            return self._upscale(
                image,
                model,
                chroma_model,
                chroma_resampler,
                tile_overlap,
                provider_name,
                device_id,
                pad_multiple,
                tile_size,
            )
        finally:
            if unload_after_run:
                _prune_cache(set())

    def _upscale(
        self,
        image,
        model,
        chroma_model,
        chroma_resampler="Catmull-Rom",
        tile_overlap=16,
        provider_name="DmlExecutionProvider",
        device_id=0,
        pad_multiple=0,
        tile_size=0,
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

        model_session = _get_session(model_path, provider_name, device_id)
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

            chroma_session = _get_session(chroma_path, provider_name, device_id)
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

        multiframe = _is_multiframe(
            model_input_channels,
            model_output_channels,
        )

        if (
            model_input_channels == 3
            and model_output_channels == 3
        ) or multiframe:
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

            if multiframe:
                rgb_output_np = _run_multiframe(
                    model_session,
                    rgb_np,
                    model_input_channels // 3,
                    tile_overlap,
                    pad_multiple,
                    tile_size,
                )
            else:
                rgb_output_np = _run_primary_model(
                    model_session,
                    rgb_np,
                    tile_overlap,
                    pad_multiple,
                    tile_size,
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

            if (
                abs(scale_width - round(scale_width)) > 1e-6
                or not 2 <= round(scale_width) <= 8
            ):
                raise ArtCNNONNXError(
                    "This node supports integer 2x-8x RGB models. "
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
            tile_overlap,
            pad_multiple,
            tile_size,
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
                tile_overlap,
                pad_multiple,
                tile_size,
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
