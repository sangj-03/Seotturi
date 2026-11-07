import os
os.environ.setdefault("HF_HOME", "/data/huggingface")
from huggingface_hub import snapshot_download, hf_hub_download
p = hf_hub_download("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")
print("lora", p, flush=True)
p = snapshot_download("Qwen/Qwen-Image-Edit-2511", max_workers=8)
print("model", p, flush=True)
