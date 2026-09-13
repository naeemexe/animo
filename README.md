# Animo

Type what you want, watch it acted out in 3D — live, in the browser.

Animo has two modes that share one engine:

**Create** — describe a scene, then direct it one instruction at a time:

```
"a person walks toward the middle of the platform"   → scene + Person 1
"add another person walking in from the other side"  → Person 2 joins, walks in to meet them
"they fight"                                          → a new beat appended to both people
"make a car arrive into the scene"                    → an object beat, no person involved
```

**Chat** — a character stands on screen, answers out loud, and can *physically
demonstrate* the answer. Ask "teach me how to do a squat" and she generates the
motion and performs it; ask an ordinary question and she just talks, gesturing
while she speaks. Two characters ship with the app and you can swap between
them at any time from the top-right toggle.

---

## How a prompt becomes animation

Gemini never renders anything. It makes **decisions** — structured JSON, nothing
else — and JavaScript does all the generation and rendering in the browser.

```
                         your prompt
                              |
                              v
                   ┌────────────────────┐
                   │      Gemini        │   decisions only, strict JSON
                   └────────────────────┘
                    /         |         \
        what's in  /   what to do next   \  is this a person
        the scene /    (the director)     \  or an object?
                 v            v            v
      ┌────────────────┐ ┌──────────┐ ┌──────────────┐
      │ scene plan     │ │ add a    │ │ object path  │
      │ props + light  │ │ person / │ │ keyframes    │
      │ + motion text  │ │ motion   │ │              │
      └────────────────┘ └──────────┘ └──────────────┘
                 |            |             |
                 |            v             |
                 |     ┌─────────────┐      |
                 |     │ motion      │      |     text → .bvh
                 |     │ server (GPU)│      |     NVIDIA Kimodo
                 |     └─────────────┘      |
                 |            |             |
                 v            v             v
        ┌───────────────────────────────────────────┐
        │            JavaScript / three.js          │
        │  places props · retargets .bvh onto the   │
        │  VRM characters · one shared play clock   │
        └───────────────────────────────────────────┘
                              |
                              v
                   ┌────────────────────┐
                   │  WebM video export │   canvas capture + orbit,
                   │  (+ Gemini music)  │   optional generated soundtrack
                   └────────────────────┘
```

Two things run: a **static frontend** (this repo, in your browser — no build
step) and a **GPU backend** (`backend/motion_server.py`) that wraps NVIDIA's
Kimodo text-to-motion model. Everything else — Gemini calls, layout, skinning,
playback, video export — happens client-side.

Without a reachable motion server the app still runs; Create falls back to the
sample `.bvh` files in `assets/motions/`.

---

## The scene: more than one character

This is the part that makes it an animation tool rather than a motion viewer.
Kimodo generates one body moving in isolation — it has no concept of a second
person, or of anything else on stage. The scene layer is what turns those
isolated clips into something that reads as a single shot.

| Problem | What the scene layer does |
|---|---|
| A newcomer would walk in anywhere | `pickSpawnForNewCharacter` solves for the start point that puts their **last** frame beside the existing cast, so the walk *ends* in the right place |
| Two people ignore each other | `characterEndFacing` / `clipFacingAt` / `yawClipToFace` rotate a whole beat — travel and body — to aim at whoever they're interacting with, recomputed each beat so aim can't drift |
| A shared beat walks them through each other | `interactionTravelCap` bounds travel by the gap actually between them, not a fixed number; a punch reaches ~72 units, so they meet at 58 |
| Beats would snap at the seam | `mergeClips` shifts clip B onto clip A's last root position and crossfades the pose (slerped rotations) over 0.4 s |
| Everyone starts at once | `prependHoldToCharacter` holds a newcomer still until the beat they're reacting to has finished — someone getting out of a car waits for the car to arrive |
| Someone emerging from a prop | `pickSpawnNearObject` places them just outside that object's footprint, facing the scene |
| Props in the way | the layout engine keeps scenery clear of every character's path, scaled by each prop's own footprint |

Each character keeps its own skeleton, mixer and clip list, and gets its own
row on the timeline. One shared clock drives every character and every moving
object, so the whole scene plays once, together, and pauses together.

---

## Files

| File | What it is |
|---|---|
| `index.html` | The entire UI: top bar, mode toggle, side panel, viewport, timeline. No build step — three.js and three-vrm load from a CDN via an import map. |
| `app.js` | All application logic (see the map below). |
| `style.css` | All styling. Design tokens live in `:root` at the top. |
| `local-config.js` | **Your browser-side settings** (Gemini, motion server URL, ElevenLabs). Gitignored — copy `local-config.example.js` to create it. |
| `.env` | **Server-side secrets** for the backend. Gitignored — copy `.env.example`. |
| `backend/motion_server.py` | The GPU service. Loads Kimodo once, serves `POST /generate-motion`. |
| `backend/gemini_proxy.py` | Optional: routes Gemini through your own Google Cloud project (ADC) instead of an API key. |
| `models/` | The prop library — `tree`, `bush`, `house`, `building`, `truck`, each with a thumbnail. |
| `assets/character/` | The two VRM characters, `avatar_male.vrm` and `avatar_female.vrm`. |
| `assets/motions/` | `.bvh` motions. `persona_idle`, `persona_wave` and `talking` are the Chat character's own three (listening, the hello she gives on arrival, speaking) — generated once and committed, so entering Chat costs no GPU call. The rest are the fallback when no motion server is reachable. |

## Map of `app.js`

It's one file, in this order:

| Section | What lives there |
|---|---|
| **Config & motion fetch** | `API`, `GEMINI_URL`, `jsonGenConfig` (shared low-thinking, JSON-only generation settings), `fetchMotionBVH` (falls back to a sample `.bvh` when the server is unreachable). |
| **Create look** | The bright painted stage — sky, grade and ground treatment used in Create only; Chat keeps a neutral studio look. |
| **VRM characters** | `loadAvatarVRM`, `buildVrmSkinnedMeshes` — the characters are skinned onto the *BVH* skeleton rather than retargeting rotations onto their own rig (the two rigs' rest axes differ per bone). Their own 4-bone skin weights carry over, so joints blend as they would in any VRM viewer, with no per-frame code. Create alternates through `CREATE_AVATAR_ORDER` so a crowd isn't one face repeated; Chat uses whichever the toggle picked. |
| **Shared playback clock** | `createPlaybackClock`, `restartPlayback`, `updateObjectAnimations`. One clock for every character and object. |
| **Characters** | `characters[]` — each with its own skeleton, mixer, clip list and avatar. `createCharacter`, `appendMotionToCharacter`, and the relative-positioning helpers described above. |
| **Gemini scene agent** | `SCENE_SYSTEM_PROMPT` + `callGemini` turn the first prompt into scene JSON. `PRIMARY_TYPE_PROMPT` runs first and decides whether the prompt is even about a person — "make a car move into the scene" never touches the motion server and creates no body. |
| **Layout engine** | `computeLayout`, `buildScene` — Gemini picks *what*, code decides *where*. Large props are ranked well out from the centre so they read as backdrop, and a prop can carry a `count` when the location *is* that prop (a forest of trees, a street of buildings). |
| **Director** | `CHAT_CLASSIFY_PROMPT`, `classifyChat`, `handleAddCharacter` / `handleAddMotion` / `handleAddObject` — every follow-up instruction, routed against the current cast. |
| **Staying in frame** | `normalizeRootTravel` clamps a beat's travel; `frameCreateStage` fits the camera to the cast plus nearby props (distant scenery fills the background without deciding the shot). |
| **Filling a beat** | `actionSpan` / `sliceClip` / `fillBeat` — Kimodo pads a clip with a static pose, so the moving part is cut out and repeated to the beat's full length, identically for everyone in the beat. |
| **Timeline** | `renderTimelineClips` (one row per character, plus a row for an object-led beat), `updatePlayhead`. |
| **Editor & export** | Click-select, move/rotate/scale, the prop library, WebM video export, `.animo` scene save/load. |
| **Chat mode** | `HELP_SYSTEM_PROMPT`, `askHelp` — the Q&A side. Only a "teach me X" answer spends a motion generation, and a take you save to Learned (IndexedDB) replays whenever that move is asked for again. `personaPlayOnly` keeps her three motions mutually exclusive, so a question arriving mid-hello can't leave two stacked. |
| **Voice** | ElevenLabs when a key is present, matched per character; otherwise the best voice the browser has installed. |
| **Mode separation** | `setMode` + snapshots — Create and Chat keep entirely separate scenes, camera included. |

---

## Running it

### 1. Frontend

```bash
python3 -m http.server 8080
```

Open <http://localhost:8080>. Then copy the config file and fill in what you use:

```bash
cp local-config.example.js local-config.js
```

### 2. Gemini — pick one

**A. Your own Google Cloud project** (no key in the browser; needed if your
org disallows API keys, since Application Default Credentials are then the only
credential available):

```bash
bash <(curl -sSL https://storage.googleapis.com/cloud-samples-data/adc/setup_adc.sh)
gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT_ID

pip install -r backend/requirements-proxy.txt
GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID uvicorn backend.gemini_proxy:app --port 8010
```

Then set `window.ANIMO_GEMINI_PROXY_URL = 'http://localhost:8010'` in
`local-config.js`. The proxy holds the refreshable OAuth token — which belongs
on a machine, not in browser JS — and exposes the same request/response shape
as the AI Studio API, so the frontend needs no changes.

**B. An AI Studio API key** — simpler for a quick start. Put your key in
`window.ANIMO_GEMINI_KEY` (get one at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey)).

### 3. Motion server (optional)

Without it, motions fall back to the samples in `assets/motions/`. With it,
every beat is generated from your text.

```bash
cp .env.example .env          # HF_TOKEN, KIMODO_DIR, HF_HOME, MOTION_CORS_ORIGINS
pip install -r backend/requirements-motion.txt
uvicorn backend.motion_server:app --host 0.0.0.0 --port 8000
```

| Endpoint | |
|---|---|
| `GET /health` | `{"status": "ok", "model": ..., "gpu": ...}` |
| `POST /generate-motion` | `{"prompt": str, "duration": seconds}` → a `.bvh` file |

Point the frontend at it with `window.ANIMO_API_URL`, or tunnel it to your
machine and leave the default:

```bash
ssh -N -L 8000:localhost:8000 -p <port> -i ~/.ssh/id_ed25519 root@<host>
```

### Deploying the motion server

It needs an NVIDIA GPU. The service is identical on either host:

- **RunPod** — start a PyTorch GPU pod, expose port 8000, clone the repo and run
  the commands above. Convenient for bursty use; the pod can be stopped between
  sessions.
- **Vultr** — a Cloud GPU instance gives you a persistent box with a stable IP,
  which is easier to point a deployed frontend at. Open port 8000 in the
  firewall and run the same commands.

Set `MOTION_CORS_ORIGINS` to the origin serving the frontend (or `*` while
developing). `HF_TOKEN` needs access to `meta-llama/Meta-Llama-3-8B-Instruct`,
which Kimodo uses as its text encoder.

---

## Configuration

| Where | What goes in it |
|---|---|
| `local-config.js` (gitignored) | Read by the browser: `ANIMO_API_URL`, `ANIMO_GEMINI_PROXY_URL` **or** `ANIMO_GEMINI_KEY`, `ANIMO_ELEVENLABS_KEY`, `ANIMO_ELEVENLABS_VOICES`. |
| `.env` (gitignored) | Read by the backend: `HF_TOKEN`, `KIMODO_DIR`, `HF_HOME`, `MOTION_CORS_ORIGINS`, `GOOGLE_CLOUD_PROJECT`. |

Only put client-side keys in `local-config.js` — it is served to the browser.
Both files are gitignored; the `.example` versions document every setting.

**Voice (optional).** With an ElevenLabs key the characters speak in real
voices, one per character, switching with the toggle. Without one they still
speak, using the best voice your browser has installed — which is most of the
way there, offline and free.

---

## Exporting

| | |
|---|---|
| **Render** (top bar) | Click a point to orbit around, then it records the canvas to `animo-render.webm` (VP9), optionally with a Gemini-generated soundtrack matched to the scene. |
| **Save / import** | The whole scene — cast, beats, props — round-trips through a `.animo` file. |

## Keyboard

| Key | Action |
|---|---|
| Space | Play / pause (replays from the start once finished) |
| M / R / S | Move / rotate / scale the selected object |
| Escape | Deselect |
| Backspace | Delete the selected object, or the selected timeline beat |

---

## Known limits

- Each character's motion is generated independently — Kimodo has no concept of
  two people interacting. A shared beat turns them to face each other and caps
  travel by the real gap between them, so a fight or a handshake reads as one
  exchange, but hands don't literally connect.
- Kimodo takes prompts starting with "A person", maxes out around 10 seconds,
  and can't do full body inversions (a backflip won't actually invert). It also
  pads: a 5 s clip often performs the action in about a second and stands still
  for the rest, which is what `fillBeat` works around.
- Terrain shape is set once, from the opening prompt — a later beat is a pose
  change, not a redescription of the ground.
- Gemini's free tier allows a limited number of requests per day per model;
  each story beat is one request. The proxy above bills against Cloud credit
  instead.

## Planned

Sign-in (Auth0), persistent per-user memory for learned skills and saved scenes
(Backboard), and generation history (TigerData). `.env.example` lists the
settings each will need; none are wired up yet.

## Built with

[NVIDIA Kimodo](https://github.com/nv-tlabs/kimodo) ·
[Google Gemini](https://ai.google.dev) ·
[three.js](https://threejs.org) ·
[@pixiv/three-vrm](https://github.com/pixiv/three-vrm) ·
[ElevenLabs](https://elevenlabs.io)
