"""
Animo Gemini proxy: lets the browser use Gemini through your own Google Cloud
project (Vertex AI) without any key in client-side code.

It holds your Application Default Credentials (a refreshable OAuth token,
which belongs on a machine rather than in browser JS), refreshes them
automatically, and exposes an endpoint with the same request and response
shape as the AI Studio API.

Setup (one-time):
    1. Log in with Application Default Credentials:
         bash <(curl -sSL https://storage.googleapis.com/cloud-samples-data/adc/setup_adc.sh)
    2. Enable Vertex AI on your project:
         gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT_ID
    3. Set GOOGLE_CLOUD_PROJECT (and optionally VERTEX_LOCATION / VERTEX_MODEL)
       in .env or your shell.

Run it:
    pip install -r backend/requirements-proxy.txt
    uvicorn backend.gemini_proxy:app --port 8010

Then set window.ANIMO_GEMINI_PROXY_URL = 'http://localhost:8010' in local-config.js.
"""
import os

try:  # .env is optional; plain environment variables work too
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import google.auth
import google.auth.transport.requests
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
DEFAULT_MODEL = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # local dev only — this never leaves your machine
    allow_methods=["POST"],
    allow_headers=["*"],
)

# google.auth.default() reads ~/.config/gcloud/application_default_credentials.json
# (set up by the setup_adc.sh script) and returns a Credentials object that
# knows how to refresh itself. We keep ONE instance and just refresh it before
# each call instead of re-loading from disk every time.
_credentials = None
_auth_request = google.auth.transport.requests.Request()


def get_access_token() -> str:
    global _credentials
    if _credentials is None:
        try:
            _credentials, adc_project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=(
                    "No Application Default Credentials found. Run: "
                    "bash <(curl -sSL https://storage.googleapis.com/cloud-samples-data/adc/setup_adc.sh) "
                    f"— underlying error: {e}"
                ),
            )
    _credentials.refresh(_auth_request)  # no-op if the cached token still has time left
    return _credentials.token


@app.get("/health")
def health():
    configured = bool(PROJECT_ID)
    return {"ok": configured, "project": PROJECT_ID or None, "location": LOCATION,
            "hint": None if configured else "Set GOOGLE_CLOUD_PROJECT before starting this server"}


@app.post("/generateContent")
async def generate_content(request: Request):
    if not PROJECT_ID:
        raise HTTPException(500, "GOOGLE_CLOUD_PROJECT is not set — see the top of gemini_proxy.py")

    model = request.query_params.get("model", DEFAULT_MODEL)
    body = await request.json()

    token = get_access_token()
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
        f"/locations/{LOCATION}/publishers/google/models/{model}:generateContent"
    )
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body,
        timeout=120,
    )
    # Vertex uses the same {candidates:[...]} / {error:{status,message}} shape
    # as the AI Studio API, so the frontend's existing response parsing needs
    # no changes — just forward status + body through as-is.
    return JSONResponse(status_code=resp.status_code, content=resp.json())
