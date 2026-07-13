#!/usr/bin/env python3
"""Standalone offline Cosmos3 **text-to-image** inference via vllm-omni.

Sibling of ``cosmos3_infer.py`` (which does T2V/I2V). This one drives the
vllm-omni ``Omni`` engine for a single text-to-image generation and writes a PNG,
so you can sanity-check an FP8 super-t2i export against its bf16 baseline.

What makes it T2I (vs T2V with one frame): the pipeline detects t2i from the
**prompt modalities**, not the frame count — ``_is_t2i_request`` returns
``"image" in prompt["modalities"]`` (pipeline_cosmos3.py:740). So the prompt
payload carries ``"modalities": ["image"]`` (as in the bagel/lance examples).
Once detected, the pipeline forces the latent T-dim to 1, applies the T2I system
prompt + image-resolution template, and the T2I guidance_interval.

FP8 is auto-detected from the checkpoint's ``transformer/config.json``
``quantization_config`` (``quant_method=modelopt``), so there's no
``--quantization`` flag — point ``--model`` at the FP8 export or the bf16 dir.

Defaults mirror the super-t2i calibration shape set in
``pipeline_checkpoints/models.mk`` (num_frames=1, 720x1280, 50 steps,
guidance 6.0, flow_shift 3.0) so the check runs at the shape the FP8 scales were
calibrated for. Override any of them on the CLI. Example:

    # FP8 checkpoint:
    python .sandbox/overlay/cosmos3_infer_t2i.py \
        --model /home/scratch.wkutak_other_1/dev/cosmos3/quantization/data/super-t2i/fp8 \
        --prompt "A photorealistic red fox sitting in a snowy forest at dawn." \
        --output /tmp/super_t2i_fp8.png

    # bf16 baseline for A/B:
    python .sandbox/overlay/cosmos3_infer_t2i.py --model .../super-t2i/bf16 --output /tmp/super_t2i_bf16.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# T2I-specific defaults, aligned with super-t2i's models.mk calibration and the
# pipeline's t2i serving constants (pipeline_cosmos3.py: flow_shift=3.0, 50 steps).
_T2I_DEFAULT_FLOW_SHIFT = 3.0
_OMNI_DEFAULT_MAX_SEQUENCE_LENGTH = 4096

_DEFAULT_MODEL = "/home/scratch.wkutak_other_1/dev/cosmos3/quantization/data/super-t2i/fp8"
_DEFAULT_PROMPT = (
    "A photorealistic red fox sitting upright in a snowy pine forest at dawn, "
    "soft golden light, fine fur detail, shallow depth of field."
)

_SUPPORTED_INPUT_KEYS = {
    "prompt", "negative_prompt", "height", "width",
    "num_inference_steps", "guidance_scale", "flow_shift",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default=_DEFAULT_MODEL,
                   help="Checkpoint dir (fp8 or bf16). FP8 is auto-detected from config.json.")
    p.add_argument("--input", type=Path, default=None,
                   help="Optional request JSON. CLI flags override its values.")
    p.add_argument("--output", type=Path, default=Path("/tmp/cosmos3_t2i.png"))

    # prompt
    p.add_argument("--prompt", default=None)
    p.add_argument("--negative-prompt", default="")

    # sampling shape — default None so value is CLI > input JSON > t2i default.
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--flow-shift", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-sequence-length", type=int, default=_OMNI_DEFAULT_MAX_SEQUENCE_LENGTH)
    p.add_argument("--no-system-prompt", action="store_true",
                   help="Disable the T2I system prompt (default: enabled).")

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
        unknown = sorted(k for k in data if k not in _SUPPORTED_INPUT_KEYS)
        if unknown:
            print(f"[warn] ignoring unrecognized input keys (t2i): {unknown}")
        base = {k: v for k, v in data.items() if k in _SUPPORTED_INPUT_KEYS}
    return base


def main() -> None:
    args = parse_args()
    base = merge_input_json(args)

    def pick(cli_val, key, default):
        return cli_val if cli_val is not None else base.get(key, default)

    prompt = args.prompt or base.get("prompt") or _DEFAULT_PROMPT
    negative_prompt = args.negative_prompt or base.get("negative_prompt") or ""
    height = pick(args.height, "height", 720)
    width = pick(args.width, "width", 1280)
    num_steps = pick(args.num_inference_steps, "num_inference_steps", 50)
    guidance = pick(args.guidance_scale, "guidance_scale", 6.0)
    flow_shift = pick(args.flow_shift, "flow_shift", _T2I_DEFAULT_FLOW_SHIFT)

    import torch
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    print("=" * 60)
    print(f" model            = {args.model}")
    print(f" mode             = T2I (modalities=['image'], num_frames=1)")
    print(f" shape            = {width}x{height}, {num_steps} steps")
    print(f" guidance/flow    = {guidance} / {flow_shift}, seed={args.seed}")
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

    # "modalities": ["image"] is what flips the pipeline into T2I mode
    # (_is_t2i_request); it also routes the image-output stage in Omni.generate.
    prompt_payload: dict[str, object] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or None,
        "modalities": ["image"],
    }

    gen_params = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_frames=1,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        seed=args.seed,
        max_sequence_length=args.max_sequence_length,
        extra_args={
            "flow_shift": flow_shift,
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

    image = outputs[0].request_output.images[0]
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    # Collapse any leading singleton (batch/T) dims -> (H,W,C).
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.dtype != np.uint8:
        f = image.astype(np.float32)
        if f.min() < 0:
            f = (f + 1.0) / 2.0
        image = np.clip(f * (255.0 if f.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)

    print(f"[gen] image {image.shape} in {gen_s:.1f}s")
    if torch.cuda.is_available():
        print(f"[mem] peak CUDA allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    write_image(image, args.output)
    print(f"[done] wrote {args.output}")


def write_image(image: np.ndarray, out_path: Path) -> None:
    """Write (H,W,C) uint8 to PNG, with a PIL fallback."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(str(out_path), image)
        return
    except Exception as e:  # noqa: BLE001
        print(f"[warn] imageio write failed ({e}); trying PIL")
    try:
        from PIL import Image
        Image.fromarray(image).save(str(out_path))
    except Exception as e:  # noqa: BLE001
        npy = out_path.with_suffix(".npy")
        print(f"[warn] PIL write failed ({e}); saving raw array to {npy}")
        np.save(npy, image)


if __name__ == "__main__":
    main()
