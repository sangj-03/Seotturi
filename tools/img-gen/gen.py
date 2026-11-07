"""Draw the 서투리 conversation partners with Qwen-Image-Edit-2511 (Apache-2.0), each edited from one of
the app's own character images so that the art style stays the same.

    PYTHONPATH=pylib python gen.py encode [--only id,id]     # prompts -> enc.pt, then the process exits
    PYTHONPATH=pylib python gen.py draw [--only id,id] [--seeds 2]
    ... --jobs touchup.json                                  # another job list (enc-touchup.pt, out/<id>/<name>-s<seed>.png)

A job's "ref" is a file in refs/, or "out/<id>/<file>" to edit an earlier drawing again.

The GPU shares one memory with the CPU and with the site, so both stages keep their footprint small:
encoding runs in its own process (the 17 GB text encoder is gone when it exits); drawing builds the
20B transformer tensor by tensor straight on the GPU, adds the 8-step Lightning LoRA to each bf16 weight
and stores it as FP8 (diffusers layerwise casting upcasts one layer at a time to bf16 while it runs),
about 21 GB instead of 41, dropping each file from the page cache after reading it and waiting while
the system reports memory pressure. Output: out/<id>/s<seed>.png
"""
import argparse
import json
import math
import os
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HOME", "/data/huggingface")
from accelerate import init_empty_weights  # noqa: E402
from accelerate.utils import set_module_tensor_to_device  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline, QwenImageTransformer2DModel  # noqa: E402
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import CONDITION_IMAGE_SIZE, calculate_dimensions  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = snapshot_download("Qwen/Qwen-Image-Edit-2511", local_files_only=True)
LORA = hf_hub_download("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
                       local_files_only=True)
# The scheduler the Lightning LoRAs were distilled with (lightx2v/Qwen-Image-Lightning README).
LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False, "max_image_seq_len": 8192,
    "max_shift": math.log(3), "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": None,
    "stochastic_sampling": False, "time_shift_type": "exponential", "use_beta_sigmas": False,
    "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False,
}


def pressure():
    with open("/proc/pressure/memory") as f:
        return float(f.readline().split()[1].split("=")[1])  # "some avg10"


def calm(limit=12.0):
    """Wait until the system's memory pressure (avg over 10 s) is below `limit` percent."""
    t0 = time.time()
    while pressure() > limit and time.time() - t0 < 600:
        time.sleep(2)


def drop_cache(path):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def enc_path(args):
    name = os.path.splitext(os.path.basename(args.jobs))[0]
    return os.path.join(HERE, "enc.pt" if name == "jobs" else f"enc-{name}.pt")


def jobs_of(args):
    spec = json.load(open(os.path.join(HERE, args.jobs), encoding="utf-8"))
    return spec, [j for j in spec["jobs"] if not args.only or j["id"] in args.only.split(",")]


def ref_of(j):
    path = j["ref"] if j["ref"].startswith("out/") else os.path.join("refs", j["ref"])
    return Image.open(os.path.join(HERE, path)).convert("RGB")


def encode(args):
    spec, jobs = jobs_of(args)
    te = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16,
                                                            device_map="cuda")
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, transformer=None, text_encoder=te, torch_dtype=torch.bfloat16)
    ENC = enc_path(args)
    enc = torch.load(ENC) if os.path.exists(ENC) else {}
    with torch.inference_mode():
        for j in jobs:
            ref = ref_of(j)
            cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, ref.width / ref.height)
            pe, pm = pipe.encode_prompt(image=[pipe.image_processor.resize(ref, ch, cw)], prompt=f"{j['prompt']} {spec.get('style', '')}".strip(),
                                        device="cuda")
            enc[j["id"]] = {"pe": pe.cpu(), "pm": None if pm is None else pm.cpu(), "prompt": j["prompt"]}
            print("encoded", j["id"], tuple(pe.shape), flush=True)
    torch.save(enc, ENC)


def lora_deltas():
    """module name -> (up, down, scale) of the Lightning LoRA (kohya format: lora_up, lora_down, alpha)."""
    out = {}
    with safe_open(LORA, "pt") as f:
        names = {k.rsplit(".", 2)[0] for k in f.keys() if k.endswith(".lora_down.weight")}
        for n in names:
            down, up = f.get_tensor(f"{n}.lora_down.weight"), f.get_tensor(f"{n}.lora_up.weight")
            alpha = float(f.get_tensor(f"{n}.alpha")) if f"{n}.alpha" in f.keys() else down.shape[0]
            out[n] = (up, down, alpha / down.shape[0])
    return out


def fp8_transformer():
    cfg = QwenImageTransformer2DModel.load_config(MODEL, subfolder="transformer")
    with init_empty_weights():
        tr = QwenImageTransformer2DModel.from_config(cfg)
    tr.to(torch.bfloat16)
    tr.requires_grad_(False)
    # Decides per layer which weights are stored as FP8 (norms, embeddings and the in/out projections stay bf16).
    tr.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    deltas = lora_deltas()
    folder = os.path.join(MODEL, "transformer")
    index = json.load(open(os.path.join(folder, "diffusion_pytorch_model.safetensors.index.json")))
    merged = 0
    for shard in sorted(set(index["weight_map"].values())):
        calm()
        path = os.path.join(folder, shard)
        with safe_open(path, "pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k).to("cuda")
                mod = k[:-len(".weight")] if k.endswith(".weight") else None
                if mod in deltas:
                    up, down, scale = deltas[mod]
                    t = (t.float() + scale * (up.to("cuda").float() @ down.to("cuda").float())).to(torch.bfloat16)
                    merged += 1
                set_module_tensor_to_device(tr, k, "cuda", value=t)  # takes the dtype chosen above (FP8 or bf16)
                del t
        drop_cache(path)
        print(f"loaded {shard}: GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB, pressure {pressure():.0f}%", flush=True)
    assert merged == len(deltas), (merged, len(deltas))
    assert not any(p.is_meta for p in tr.parameters())
    tr.to("cuda")  # the buffers (rotary tables) built on the CPU
    return tr


def draw(args):
    spec, jobs = jobs_of(args)
    enc = torch.load(enc_path(args))
    tag = "" if args.jobs == "jobs.json" else os.path.splitext(os.path.basename(args.jobs))[0] + "-"
    t0 = time.time()
    tr = fp8_transformer()
    sched = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, transformer=tr, text_encoder=None, scheduler=sched, torch_dtype=torch.bfloat16)
    pipe.vae.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"transformer ready in {time.time() - t0:.0f} s, GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    for j in jobs:
        e = enc[j["id"]]
        if e["prompt"] != j["prompt"]:
            raise SystemExit(f"{j['id']}: prompt changed since encoding; run encode again")
        ref = ref_of(j)
        w, h = (j["w"], j["h"]) if j.get("w") else calculate_dimensions(1024 * 1024, ref.width / ref.height)
        pe = e["pe"].to("cuda")
        pm = None if e["pm"] is None else e["pm"].to("cuda")
        os.makedirs(os.path.join(HERE, "out", j["id"]), exist_ok=True)
        for s in range(args.seed0, args.seed0 + args.seeds):
            calm(25.0)
            t1 = time.time()
            with torch.inference_mode():
                im = pipe(image=[ref], prompt_embeds=pe, prompt_embeds_mask=pm, true_cfg_scale=1.0, num_inference_steps=8,
                          width=w // 16 * 16, height=h // 16 * 16, generator=torch.Generator("cuda").manual_seed(s)).images[0]
            path = os.path.join(HERE, "out", j["id"], f"{tag}s{s}.png")
            im.save(path)
            print(f"{j['id']} seed {s}: {im.size} in {time.time() - t1:.0f} s, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB "
                  f"-> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["encode", "draw"])
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", default="jobs.json")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--gpu-share", type=float, default=0.4)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.gpu_share)
    encode(a) if a.stage == "encode" else draw(a)
