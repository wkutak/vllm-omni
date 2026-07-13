#!/usr/bin/env python3
"""Standalone offline Cosmos3 **image-to-video** (i2v) inference via vllm-omni.

Sibling of `cosmos3_infer.py` (T2V/I2V) and `cosmos3_infer_t2i.py` (T2I). This one
is specialised for i2v: it fetches a conditioning image (default: the Cosmos
`car_driving.jpg`), feeds it as `multi_modal_data={"image": ...}`, and generates a
video continuing from it, writing an MP4. Use it to sanity-check a super-i2v FP8
export against its bf16 baseline.

How it's i2v (vs t2i): the pipeline detects i2v from the presence of a conditioning
**image** with a **video** output modality (default) — the image is VAE-encoded as a
clean first frame and re-injected each denoise step (`_prepare_latents_i2v`). t2i
would instead pass `modalities=["image"]`; i2v does not.

FP8 is auto-detected from the checkpoint's `transformer/config.json`
`quantization_config`, so there is no `--quantization` flag — point `--model` at the
FP8 export or the bf16 dir.

Example:
    python .sandbox/overlay/cosmos3_infer_i2v.py \
        --model /home/scratch.wkutak_other_1/dev/cosmos3/quantization/data/super-i2v/fp8 \
        --output /tmp/super_i2v_fp8.mp4
    # bf16 baseline (same image/seed):
    python .sandbox/overlay/cosmos3_infer_i2v.py --model .../super-i2v/bf16 --output /tmp/super_i2v_bf16.mp4
    # your own image / prompt:
    python .sandbox/overlay/cosmos3_infer_i2v.py --image /path/to/frame0.jpg --prompt "..."
"""

from __future__ import annotations

import argparse
import io
import time
import urllib.request
from pathlib import Path

import numpy as np

# Defaults aligned with cosmos3/serving_stack/cosmos3_omni.py (video/i2v path).
_OMNI_DEFAULT_FLOW_SHIFT = 10.0
_OMNI_DEFAULT_MAX_SEQUENCE_LENGTH = 4096

_DEFAULT_MODEL = "/home/scratch.wkutak_other_1/dev/cosmos3/quantization/data/super-i2v/fp8"
_DEFAULT_IMAGE_URL = (
    "https://raw.githubusercontent.com/NVIDIA/cosmos/refs/heads/main/cookbooks/"
    "cosmos3/generator/audiovisual/assets/images/image2video/car_driving.jpg"
)
_DEFAULT_PROMPT = (
    "A car drives forward along the road, smooth continuous camera motion, "
    "consistent lighting and scenery, photorealistic."
)


def load_image(path_or_url: str):
    """Load a conditioning image from a local path or an http(s) URL as PIL RGB."""
    from PIL import Image

    if path_or_url.startswith(("http://", "https://")):
        print(f"[i2v] fetching conditioning image: {path_or_url}")
        with urllib.request.urlopen(path_or_url) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default=_DEFAULT_MODEL,
                   help="Checkpoint dir (fp8 or bf16). FP8 auto-detected from config.json.")
    p.add_argument("--image", default=_DEFAULT_IMAGE_URL,
                   help="Conditioning first-frame image (local path or http(s) URL).")
    p.add_argument("--output", type=Path, default=Path("/tmp/cosmos3_i2v.mp4"))

    p.add_argument("--prompt", default=_DEFAULT_PROMPT)
    p.add_argument("--negative-prompt", default="")

    # sampling shape (i2v = video; num_frames must satisfy (n-1) % vae_temporal == 0).
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num-frames", type=int, default=93)
    p.add_argument("--num-inference-steps", type=int, default=35)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--flow-shift", type=float, default=_OMNI_DEFAULT_FLOW_SHIFT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-sequence-length", type=int, default=_OMNI_DEFAULT_MAX_SEQUENCE_LENGTH)
    p.add_argument("--no-system-prompt", action="store_true")

    # engine
    p.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    p.add_argument("--cfg", type=int, default=1, help="cfg_parallel_size")
    p.add_argument("--ulysses", type=int, default=1, help="ulysses_degree")
    p.add_argument("--torch-compile", action="store_true",
                   help="Enable torch.compile (default: enforce_eager).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image = load_image(args.image)

    import torch
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    print("=" * 60)
    print(f" model            = {args.model}")
    print(f" mode             = I2V (multi_modal_data image, video output)")
    print(f" image            = {args.image}  ({image.width}x{image.height})")
    print(f" shape            = {args.width}x{args.height}, {args.num_frames} frames, {args.num_inference_steps} steps")
    print(f" guidance/flow    = {args.guidance_scale} / {args.flow_shift}, fps={args.fps}, seed={args.seed}")
    print(f" engine           = tp{args.tp} cfg{args.cfg} ulysses{args.ulysses} "
          f"{'compile' if args.torch_compile else 'eager'}")
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
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt or None,
        "multi_modal_data": {"image": image},
    }

    gen_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        fps=args.fps,
        max_sequence_length=args.max_sequence_length,
        extra_args={
            "flow_shift": args.flow_shift,
            "max_sequence_length": args.max_sequence_length,
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
    if frames.dtype != np.uint8:
        f = frames.astype(np.float32)
        if f.min() < 0:
            f = (f + 1.0) / 2.0
        frames = np.clip(f * (255.0 if f.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)

    print(f"[gen] {frames.shape[0]} frames in {gen_s:.1f}s ({frames.shape[0] / gen_s:.2f} fps gen)")
    if torch.cuda.is_available():
        print(f"[mem] peak CUDA allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    write_mp4(frames, args.output, args.fps)
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
