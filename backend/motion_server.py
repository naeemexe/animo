"""
Animo motion server: text in, BVH motion out.

Loads the Kimodo text-to-motion model once and serves it to the browser app.
Runs on any NVIDIA GPU host (a RunPod pod or a Vultr Cloud GPU instance):

    cp .env.example .env          # HF_TOKEN, KIMODO_DIR, HF_HOME, MOTION_CORS_ORIGINS
    pip install -r backend/requirements-motion.txt
    uvicorn backend.motion_server:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health           -> {"status": "ok", "model": ..., "gpu": ...}
    POST /generate-motion  {"prompt": str, "duration": seconds} -> .bvh file
"""
import hashlib
import os
import sys
import tempfile

from dotenv import load_dotenv

load_dotenv()

# These must be in place before kimodo is imported.
os.environ.setdefault("HF_TOKEN", "")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
sys.path.insert(0, os.environ.get("KIMODO_DIR", "/workspace/kimodo"))

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from kimodo import load_model
from kimodo.exports.bvh import save_motion_bvh
from kimodo.skeleton import SOMASkeleton30, global_rots_to_local_rots

allowed_origins = [o.strip() for o in os.environ.get("MOTION_CORS_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(title="Animo motion server")
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["GET", "POST"], allow_headers=["*"])

# Load the model once at startup; it stays in GPU memory between requests.
print("Loading Kimodo model...")
device = "cuda:0"
model, resolved_model = load_model(
    "Kimodo-SOMA-RP-v1",
    device=device,
    default_family="Kimodo",
    return_resolved_name=True,
)
skeleton = model.skeleton
if isinstance(skeleton, SOMASkeleton30):
    skeleton = skeleton.somaskel77.to(device)
print(f"Model loaded: {resolved_model}")


@app.get("/health")
def health():
    return {"status": "ok", "model": resolved_model, "gpu": str(device)}


@app.post("/generate-motion")
def generate_motion(data: dict):
    prompt = data["prompt"]
    duration = float(data.get("duration", 5.0))
    fps = model.fps

    output = model(
        [prompt],
        [int(duration * fps)],
        constraint_lst=[],
        num_denoising_steps=100,
        num_samples=1,
        multi_prompt=True,
        num_transition_frames=10,
        post_processing=False,
        return_numpy=True,
    )

    joints_pos = torch.from_numpy(output["posed_joints"][0]).to(device)
    joints_rot = torch.from_numpy(output["global_rot_mats"][0]).to(device)
    local_rot_mats = global_rots_to_local_rots(joints_rot, skeleton)
    root_positions = joints_pos[:, skeleton.root_idx, :]

    name = hashlib.md5(f"{prompt}|{duration}".encode()).hexdigest()[:8]
    bvh_path = os.path.join(tempfile.gettempdir(), f"animo_motion_{name}.bvh")
    save_motion_bvh(bvh_path, local_rot_mats, root_positions, skeleton=skeleton, fps=fps)
    return FileResponse(bvh_path)
