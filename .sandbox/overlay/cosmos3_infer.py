#!/usr/bin/env python3
"""Standalone offline Cosmos3 inference via vllm-omni — for FP8 / bf16 checkpoints.

Drives the vllm-omni ``Omni`` engine directly (no NIM container, no Triton).
Mirrors how the production serving stack builds the engine and a diffusion
request (``cosmos3/serving_stack/cosmos3_omni.py`` :: ``build_omni`` +
``generate_diffusion``), trimmed to a single text-to-video (or image-to-video)
generation that writes an MP4.

FP8 is auto-detected from the checkpoint's ``transformer/config.json``
``quantization_config`` block (``quant_method=modelopt``, ``quant_algo=FP8``),
so there is no ``--quantization`` flag — point ``--model`` at an FP8 export to
run FP8, or at the bf16 dir for the baseline. The script prints wall-clock and
peak CUDA memory so the two can be compared (the doc's success criterion).

Run from inside an environment that has vllm-omni + torch installed
(e.g. the dev venv). Example:

    python sandbox/cosmos3_fp8_infer.py \
        --model /path/to/cosmos3-nano/fp8 \
        --prompt "A robotic arm pouring liquid into a glass, bright modern kitchen." \
        --height 480 --width 720 --num-frames 29 --num-inference-steps 15 \
        --output /tmp/cosmos3_fp8.mp4

    # baseline for comparison:
    python sandbox/cosmos3_fp8_infer.py --model /path/to/cosmos3-nano/bf16 ... --output /tmp/cosmos3_bf16.mp4

    # reuse an existing request JSON (v2v-style); video conditioning fields are
    # ignored (T2V/I2V only in this script):
    python sandbox/cosmos3_fp8_infer.py --model <dir> --input examples/offline_inference/cosmos3/inputs/v2v.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Defaults aligned with cosmos3/serving_stack/cosmos3_omni.py.
_OMNI_DEFAULT_FLOW_SHIFT = 10.0
_OMNI_DEFAULT_MAX_SEQUENCE_LENGTH = 4096

_DEFAULT_MODEL = "/home/scratch.wkutak_other_1/dev/cosmos3/quantization/data/cosmos3-nano/fp8"
_DEFAULT_PROMPT = (
    "A robotic arm, primarily white with black joints, gently pours a "
    "transparent liquid into a clear glass on a clean modern tabletop. Bright "
    "soft lighting, smooth controlled motion, photorealistic."
)

# Request-JSON keys this script understands; the rest (video conditioning) are
# ignored with a warning.
_SUPPORTED_INPUT_KEYS = {
    "prompt", "negative_prompt", "height", "width", "num_frames",
    "num_inference_steps", "guidance_scale", "flow_shift", "fps", "image",
}
_IGNORED_INPUT_KEYS = {"vision_path", "condition_frame_indexes_vision", "condition_video_keep"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default=_DEFAULT_MODEL,
                   help="Checkpoint dir (fp8 or bf16). FP8 is auto-detected from config.json.")
    p.add_argument("--input", type=Path, default=None,
                   help="Optional request JSON (v2v-style). CLI flags override its values.")
    p.add_argument("--output", type=Path, default=Path("/tmp/cosmos3_out.mp4"))

    # prompt / conditioning
    p.add_argument("--prompt", default=None)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--image", default=None, help="Optional input image for image-to-video.")

    # sampling shape — default None so resolution is CLI > input JSON > hardcoded.
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--flow-shift", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-sequence-length", type=int, default=_OMNI_DEFAULT_MAX_SEQUENCE_LENGTH)
    p.add_argument("--no-system-prompt", action="store_true")
    p.add_argument("--use-duration-template", action="store_true")

    # engine
    p.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    p.add_argument("--cfg", type=int, default=1, help="cfg_parallel_size")
    p.add_argument("--ulysses", type=int, default=1, help="ulysses_degree")
    p.add_argument("--torch-compile", action="store_true",
                   help="Enable torch.compile (default: enforce_eager).")

    return p.parse_args()


def merge_input_json(args: argparse.Namespace) -> dict:
    """Load --input JSON (if any) as base values; CLI flags win when explicitly set."""
    base: dict = {}
    if args.input is not None:
        data = json.loads(Path(args.input).read_text())
        ignored = sorted(k for k in data if k in _IGNORED_INPUT_KEYS)
        if ignored:
            print(f"[warn] ignoring video-conditioning keys (T2V/I2V only): {ignored}")
        unknown = sorted(k for k in data if k not in _SUPPORTED_INPUT_KEYS | _IGNORED_INPUT_KEYS)
        if unknown:
            print(f"[warn] ignoring unrecognized input keys: {unknown}")
        base = {k: v for k, v in data.items() if k in _SUPPORTED_INPUT_KEYS}
    return base


def main() -> None:
    args = parse_args()
    base = merge_input_json(args)

    # Resolve effective values: explicit CLI > input JSON > hardcoded default.
    def pick(cli_val, key, default):
        if cli_val is not None:
            return cli_val
        return base.get(key, default)

    prompt = args.prompt or base.get("prompt") or _DEFAULT_PROMPT
    negative_prompt = args.negative_prompt or base.get("negative_prompt") or ""
    image = args.image or base.get("image")
    height = pick(args.height, "height", 480)
    width = pick(args.width, "width", 720)
    num_frames = pick(args.num_frames, "num_frames", 29)
    num_steps = pick(args.num_inference_steps, "num_inference_steps", 15)
    guidance = pick(args.guidance_scale, "guidance_scale", 6.0)
    fps = pick(args.fps, "fps", 24.0)
    flow_shift = pick(args.flow_shift, "flow_shift", _OMNI_DEFAULT_FLOW_SHIFT)

    import torch
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    print("=" * 60)
    print(f" model            = {args.model}")
    print(f" shape            = {width}x{height}, {num_frames} frames, {num_steps} steps")
    print(f" guidance/flow    = {guidance} / {flow_shift}, fps={fps}, seed={args.seed}")
    print(f" engine           = tp{args.tp} cfg{args.cfg} ulysses{args.ulysses} "
          f"{'compile' if args.torch_compile else 'eager'}")
    print(f" image (I2V)      = {image or '<none, T2V>'}")
    print("=" * 60)

    t0 = time.time()
    omni = Omni(
        model=args.model,
        model_class_name="Cosmos3OmniDiffusersPipeline",
        trust_remote_code=True,
        enforce_eager=not args.torch_compile,
        tensor_parallel_size=args.tp,
        ulysses_degree=args.ulysses,
        cfg_parallel_size=args.cfg,
        max_sequence_length=args.max_sequence_length,
        model_config={"guardrails": False},
    )
    print(f"[load] Omni engine ready in {time.time() - t0:.1f}s")

    prompt_payload: dict[str, object] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or None,
    }
    if image:
        from PIL import Image
        with Image.open(image) as f:
            prompt_payload["multi_modal_data"] = {"image": f.convert("RGB")}

    gen_params = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        seed=args.seed,
        fps=fps,
        max_sequence_length=args.max_sequence_length,
        extra_args={
            "flow_shift": flow_shift,
            "max_sequence_length": args.max_sequence_length,
            "use_duration_template": args.use_duration_template,
            "use_system_prompt": not args.no_system_prompt,
        },
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    outputs = omni.generate(prompt_payload, gen_params)
    gen_s = time.time() - t1
    if not outputs:
        raise RuntimeError("Omni returned no outputs.")

    frames = outputs[0].request_output.images[0]
    if hasattr(frames, "detach"):
        frames = frames.detach().cpu().numpy()
    frames = np.asarray(frames)
    if frames.ndim == 5:
        frames = frames[0]
    # Pipeline yields uint8 (T,H,W,C); guard against a float range just in case.
    if frames.dtype != np.uint8:
        f = frames.astype(np.float32)
        if f.min() < 0:
            f = (f + 1.0) / 2.0
        frames = np.clip(f * (255.0 if f.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)

    print(f"[gen] {frames.shape[0]} frames in {gen_s:.1f}s ({frames.shape[0] / gen_s:.2f} fps gen)")
    if torch.cuda.is_available():
        print(f"[mem] peak CUDA allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    write_mp4(frames, args.output, fps)
    print(f"[done] wrote {args.output}")


def write_mp4(frames: np.ndarray, out_path: Path, fps: float) -> None:
    """Write (T,H,W,C) uint8 frames to MP4, with graceful fallbacks."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(str(out_path), list(frames), fps=fps, codec="libvpx-vp9",
                         quality=8, macro_block_size=None)
        return
    except Exception as e:  # noqa: BLE001
        print(f"[warn] imageio mp4 write failed ({e}); trying cv2")
    try:
        import cv2
        h, w = frames.shape[1], frames.shape[2]
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for fr in frames:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release()
        return
    except Exception as e:  # noqa: BLE001
        npy = out_path.with_suffix(".npy")
        print(f"[warn] cv2 write failed ({e}); saving raw frames to {npy}")
        np.save(npy, frames)


if __name__ == "__main__":
    main()
