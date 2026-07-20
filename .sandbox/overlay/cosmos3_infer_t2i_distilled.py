#!/usr/bin/env python3
"""Standalone offline Cosmos3 **distilled text-to-image** inference via vllm-omni.

Sibling of ``cosmos3_infer_t2i.py``, specialized for the 4-step DMD2 distilled
student (``nvidia/Cosmos3-Super-Text2Image-4Step`` -> ``super-t2i-distilled``).
Same T2I plumbing (``modalities=["image"]`` flips the pipeline into T2I mode,
forces latent T=1, applies the T2I system prompt); only the sampling defaults
differ, matching the distilled model's calibration in ``models.mk``:

    num_inference_steps = 4        (vs 50 for super-t2i)
    guidance_scale      = 1.0      (DMD2 is CFG-free; guidance>1 does nothing)

Use it to sanity-check the distilled bf16 checkpoint before quantizing, and for
bf16-vs-fp8 A/B after.

IMPORTANT — scheduler caveat
----------------------------
vllm-omni's ``Cosmos3OmniDiffusersPipeline`` hardcodes ``UniPCMultistepScheduler``
(``diffusion/models/cosmos3/pipeline_cosmos3.py``); it does NOT yet drive the
distilled model's ``FlowMatchEulerDiscreteScheduler`` fixed 4-step **stochastic
(SDE)** sampler (explicit ``fixed_step_sampler_config.t_list`` sigmas). So:

  * This script is valid for **bf16-vs-fp8 A/B** — both run through the same UniPC
    path, so the quantization delta is measured cleanly.
  * It will **not** reproduce the reference 4-step DMD2 image quality until
    vllm-omni gains distilled-scheduler support. For a faithful reference image,
    drive the diffusers ``Cosmos3OmniPipeline``/distilled modular pipeline directly
    (FlowMatchEuler + t_list sigmas), which is a separate script.

The cache backend does read ``is_distilled`` from ``model_index.json``
(``cache_dit_backend.py``: ``has_separate_cfg = not pipeline.is_distilled``), so
CFG handling on the cache path already reflects the distilled (single-stream) model.

FP8 is auto-detected from ``transformer/config.json`` (``quant_method=modelopt``),
so point ``--model`` at the fp8 export or the bf16 dir — no ``--quantization`` flag.

Example:

    # bf16 baseline:
    python .sandbox/overlay/cosmos3_infer_t2i_distilled.py \
        --model /home/scratch.wkutak_other_1/dev/cosmos3/quantization/data/super-t2i-distilled/bf16 \
        --prompt "A photorealistic red fox sitting in a snowy forest at dawn." \
        --output /tmp/super_t2i_distilled_bf16.png

    # fp8 A/B (after quantize-super-t2i-distilled):
    python .sandbox/overlay/cosmos3_infer_t2i_distilled.py \
        --model .../super-t2i-distilled/fp8 --output /tmp/super_t2i_distilled_fp8.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

_OMNI_DEFAULT_MAX_SEQUENCE_LENGTH = 4096

# Distilled defaults — mirror models.mk (STEPS=4, GUID=1.0).
_DEFAULT_MODEL = "/home/scratch.wkutak_other_1/dev/cosmos3/quantization/data/super-t2i-distilled/bf16"
_DEFAULT_STEPS = 4
_DEFAULT_GUIDANCE = 1.0
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
    p.add_argument("--model-class-name", default="Cosmos3OmniDiffusersPipeline",
                   help="vllm-omni pipeline class (only Cosmos3OmniDiffusersPipeline is registered).")
    p.add_argument("--input", type=Path, default=None,
                   help="Optional request JSON. CLI flags override its values.")
    p.add_argument("--output", type=Path, default=Path("/tmp/cosmos3_t2i_distilled.png"))

    # prompt
    p.add_argument("--prompt", default=None)
    p.add_argument("--negative-prompt", default="")

    # sampling shape — default None so value is CLI > input JSON > distilled default.
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--flow-shift", type=float, default=None,
                   help="UniPC flow_shift (vllm-omni path). Unset by default: the distilled "
                        "model uses FlowMatchEuler shift=1.0, so leave it off unless A/B'ing "
                        "the UniPC serving path explicitly.")
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
            print(f"[warn] ignoring unrecognized input keys (t2i-distilled): {unknown}")
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
    num_steps = pick(args.num_inference_steps, "num_inference_steps", _DEFAULT_STEPS)
    guidance = pick(args.guidance_scale, "guidance_scale", _DEFAULT_GUIDANCE)
    flow_shift = pick(args.flow_shift, "flow_shift", None)  # distilled: no flow_shift by default

    import torch
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    print("=" * 60)
    print(f" model            = {args.model}")
    print(f" mode             = T2I distilled (modalities=['image'], num_frames=1)")
    print(f" shape            = {width}x{height}, {num_steps} steps")
    print(f" guidance/flow    = {guidance} / {flow_shift}, seed={args.seed}")
    print(f" engine           = tp{args.tp} cfg{args.cfg} ulysses{args.ulysses} "
          f"{'compile' if args.torch_compile else 'eager'}")
    if guidance != 1.0:
        print(" [warn] distilled DMD2 is CFG-free; guidance_scale != 1.0 has no effect on it.")
    print(" [note] vllm-omni cosmos3 uses UniPC, not the distilled 4-step FlowMatchEuler SDE;")
    print("        valid for bf16-vs-fp8 A/B, not a faithful reference-quality image.")
    print("=" * 60)

    t0 = time.time()
    omni = Omni(
        model=args.model,
        model_class_name=args.model_class_name,
        trust_remote_code=True,
        enforce_eager=not args.torch_compile,
        tensor_parallel_size=args.tp,
        ulysses_degree=args.ulysses,
        cfg_parallel_size=args.cfg,
        max_sequence_length=args.max_sequence_length,
        model_config={"guardrails": False},
    )
    print(f"[load] Omni engine ready in {time.time() - t0:.1f}s")

    # "modalities": ["image"] flips the pipeline into T2I mode (_is_t2i_request)
    # and routes the image-output stage in Omni.generate.
    prompt_payload: dict[str, object] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or None,
        "modalities": ["image"],
    }

    extra_args: dict[str, object] = {
        "max_sequence_length": args.max_sequence_length,
        "use_system_prompt": not args.no_system_prompt,
    }
    if flow_shift is not None:
        extra_args["flow_shift"] = flow_shift

    gen_params = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_frames=1,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        seed=args.seed,
        max_sequence_length=args.max_sequence_length,
        extra_args=extra_args,
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
