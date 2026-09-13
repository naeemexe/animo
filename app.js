import * as THREE from 'three';
import { BVHLoader } from 'three/addons/loaders/BVHLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

// ---- Configuration ---------------------------------------------------------
// Nothing environment-specific lives in the code: URLs and keys come from
// local-config.js (gitignored — copy local-config.example.js).

// Motion backend (backend/motion_server.py on a RunPod or Vultr GPU). Defaults
// to localhost:8000, e.g. an SSH tunnel to the GPU box.
const API = (window.ANIMO_API_URL || 'http://localhost:8000').replace(/\/$/, '');

// Gemini access — two ways to reach it, same request/response shape either way:
//
//   ANIMO_GEMINI_PROXY_URL: your own Google Cloud project via
//     backend/gemini_proxy.py. No key in the browser — the proxy holds your
//     Application Default Credentials and bills against your project instead
//     of the shared free tier. Used whenever it's set.
//   ANIMO_GEMINI_KEY: an AI Studio API key. Simple, but the free tier is capped.
const GEMINI_PROXY_URL = (window.ANIMO_GEMINI_PROXY_URL || '').replace(/\/$/, '');
const USE_VERTEX_PROXY = !!GEMINI_PROXY_URL;
const GEMINI_MODEL = USE_VERTEX_PROXY ? 'gemini-2.5-flash' : 'gemini-3.8-flash';
const GEMINI_KEY = window.ANIMO_GEMINI_KEY || '';
if (!USE_VERTEX_PROXY && !GEMINI_KEY) console.warn('No Gemini access configured — set ANIMO_GEMINI_PROXY_URL or ANIMO_GEMINI_KEY in local-config.js');
const GEMINI_URL = USE_VERTEX_PROXY
    ? `${GEMINI_PROXY_URL}/generateContent?model=${GEMINI_MODEL}`
    : `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_KEY}`;

// POSTs to Gemini, retrying with backoff on transient errors (503 "high
// demand", 429 quota) instead of failing the whole generation outright —
// larger prompts (like scene planning) seem to get shed under load more
// than tiny ones, and a short retry usually clears it.
async function fetchGeminiWithRetry(url, body, maxRetries = 3) {
    let lastData = null;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        let res;
        try {
            res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
        } catch (err) {
            // fetch() rejects outright for a dead server, a refused connection
            // or a missing CORS header — there's no response to read. Name the
            // endpoint: a bare "Failed to fetch" sends you hunting in the
            // wrong place (it reads like a Gemini problem, not a local one).
            return { error: { status: 'UNREACHABLE', message: `Can't reach Gemini at ${url} — ${err.message}. Is the proxy running?` } };
        }
        // Read as text first: an error page (or a bare "Internal Server Error")
        // isn't JSON, and res.json() would throw with nothing left to inspect.
        const raw = await res.text();
        let data;
        try {
            data = JSON.parse(raw);
        } catch {
            return { error: { status: 'BAD_RESPONSE', message: `Gemini endpoint returned ${res.status} ${res.statusText || ''} — ${raw.slice(0, 200) || '(empty body)'}` } };
        }
        const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
        if (text) return data;
        lastData = data;
        const status = data?.error?.status;
        // Only retry transient overload. A quota error (RESOURCE_EXHAUSTED)
        // means the daily allowance is gone — retrying just burns more of it.
        const retriable = status === 'UNAVAILABLE' || res.status === 503;
        if (!retriable || attempt === maxRetries) break;
        const backoffMs = 1500 * Math.pow(2, attempt); // 1.5s, 3s, 6s
        log(`Gemini busy, retrying in ${(backoffMs / 1000).toFixed(1)}s... (${attempt + 1}/${maxRetries})`, 'system');
        await new Promise(r => setTimeout(r, backoffMs));
    }
    return lastData;
}

// Every call here wants one small JSON object back, and none of them need the
// model to deliberate about it. Left at its defaults, gemini flash models spent
// ~570 thinking tokens to produce an ~80 token answer — measured 5s on a
// trivial question and 49s on a bad one, which is the "stuck thinking" pause.
// thinkingLevel 'low' cuts that to ~46 tokens (thinkingBudget is rejected by
// this model). responseMimeType makes it emit bare JSON rather than a
// markdown-fenced block.
function jsonGenConfig(temperature, maxOutputTokens = 1200) {
    return {
        temperature,
        maxOutputTokens,
        responseMimeType: 'application/json',
        // Same intent (near-zero thinking), different field per API: AI Studio's
        // gemini-3.8-flash accepts thinkingLevel but 400s on thinkingBudget:0;
        // Vertex's gemini-2.5-flash is the other way around — 400s on
        // thinkingLevel ("not supported by this model") but accepts
        // thinkingBudget:0 fine. Both measured directly, not assumed.
        thinkingConfig: USE_VERTEX_PROXY ? { thinkingBudget: 0 } : { thinkingLevel: 'low' },
    };
}

// Surfaces the real reason a Gemini call failed (rate limit, bad key, etc.)
// instead of a generic "empty response" — data is the parsed response body.
function geminiErrorMessage(data) {
    const err = data?.error;
    if (!err) return 'Gemini returned empty response';
    if (err.status === 'RESOURCE_EXHAUSTED') {
        // Never retried automatically (see fetchGeminiWithRetry) — retrying a
        // quota error just spends more of the allowance. Say what to do instead.
        return 'Gemini quota exceeded — swap the key in local-config.js and reload';
    }
    return err.message || `Gemini error (${err.status || 'unknown'})`;
}
const viewport = document.getElementById('viewport');

// ========================================================================
// MOTION FETCH — talks to the motion server at API; falls back to a bundled
// sample BVH when there's no backend yet (network error, CORS, or non-2xx),
// so the rest of the app is usable/demoable before a GPU backend is up. Set
// ANIMO_API_URL in local-config.js once a motion server is running.
// ========================================================================
const SAMPLE_MOTIONS = ['backflip.bvh', 'drunk_dance.bvh', 'silly_dance.bvh', 'sneak_forward.bvh', 'sneak_jump.bvh', 'spinning_kick.bvh'];
let _warnedNoBackend = false;

async function fetchMotionBVH(prompt, duration = 5) {
    try {
        const res = await fetch(`${API}/generate-motion`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, duration })
        });
        if (!res.ok) throw new Error(`Motion backend responded ${res.status}`);
        return { bvhText: await res.text(), isFallback: false };
    } catch (err) {
        if (!_warnedNoBackend) {
            log('No motion backend reachable — using a sample animation instead. Set ANIMO_API_URL in local-config.js once your motion server is running.', 'system');
            _warnedNoBackend = true;
        }
        const file = SAMPLE_MOTIONS[Math.floor(Math.random() * SAMPLE_MOTIONS.length)];
        const sampleRes = await fetch(`assets/motions/${file}`);
        if (!sampleRes.ok) throw err; // truly nothing we can do
        return { bvhText: await sampleRes.text(), isFallback: true };
    }
}

// Three.js setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xd8d8d0);

const camera = new THREE.PerspectiveCamera(60, viewport.clientWidth / viewport.clientHeight, 1, 10000);
camera.position.set(0, 150, 400);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(viewport.clientWidth, viewport.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
// OrbitControls scales each wheel event by its size (|deltaY| / 100). A mouse
// notch is ~100, but a trackpad pinch or two-finger scroll sends a stream of
// 1-10s, so the same speed barely moved the camera. Just before the controls
// see each event (capture phase), pick the speed by the kind of input.
controls.zoomSpeed = 2.5;
renderer.domElement.addEventListener('wheel', (e) => {
    const trackpad = e.ctrlKey || Math.abs(e.deltaY) < 40; // pinch arrives as ctrl+wheel
    controls.zoomSpeed = trackpad ? 18 : 2.5;
}, { capture: true, passive: true });
controls.target.set(0, 100, 0);
controls.update();
const HOME_VIEW = { position: camera.position.clone(), target: controls.target.clone() }; // where Clear Scene puts the camera back

// Grid
const grid = new THREE.GridHelper(500, 20, 0xc0c0b8, 0xccccc4);
scene.add(grid);

// Lights — soft key + fill + rim for nice form
scene.add(new THREE.AmbientLight(0x303050, 1.5));
const keyLight = new THREE.DirectionalLight(0xffffff, 2);
keyLight.position.set(100, 200, 150);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0x8888ff, 0.8);
fillLight.position.set(-100, 100, -50);
scene.add(fillLight);
const rimLight = new THREE.DirectionalLight(0xa78bfa, 1);
rimLight.position.set(0, 50, -200);
scene.add(rimLight);

// ========================================================================
// CREATE LOOK — a bright, painted open-world stage, the same for every
// scene: a clear azure sky fading to a pale horizon, white clouds shaded
// cool underneath, hazy blue mountains, fresh grass underfoot, crisp sun
// and light aerial haze — then a colour grade (saturation, vibrance, gentle
// contrast, cool shadows, warm highlights, soft bloom) over the whole frame.
// Chat keeps its own neutral studio, so setCreateLook() swaps everything in
// and out with the mode.
// ========================================================================
// Create frames go through the grade (see renderFrame), which renders in
// linear light and converts to screen colour at the end — so colours that
// bypass lighting (sky, mountains) are given in linear here too.
const linVec = (hex) => { const c = new THREE.Color(hex); return new THREE.Vector3(c.r, c.g, c.b); };
// Ahead and a little left of the default camera, low enough that its warm
// glow hazes the horizon in shot, like a late-morning sun over a meadow.
const SKY_SUN_DIR = new THREE.Vector3(-0.35, 0.2, -0.92).normalize();
const skyGroup = new THREE.Group();
skyGroup.visible = false;
{
    const dome = new THREE.Mesh(
        new THREE.SphereGeometry(6000, 48, 24),
        new THREE.ShaderMaterial({
            side: THREE.BackSide, depthWrite: false, fog: false,
            uniforms: {
                uTop: { value: linVec(0x2e82de) },
                uMid: { value: linVec(0x6db3ec) },
                uLow: { value: linVec(0xb9dcef) },
                uHorizon: { value: linVec(0xf6efd6) }, // sunny haze, not a cold blue-white
                uSunDir: { value: SKY_SUN_DIR },
                uSun: { value: linVec(0xfff0c4) },
            },
            vertexShader: `varying vec3 vDir;
                void main() { vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
            fragmentShader: `uniform vec3 uTop, uMid, uLow, uHorizon, uSunDir, uSun; varying vec3 vDir;
                void main() {
                    vec3 d = normalize(vDir); float h = d.y;
                    vec3 col = mix(uHorizon, uLow, smoothstep(-0.02, 0.14, h));
                    col = mix(col, uMid, smoothstep(0.12, 0.38, h));
                    col = mix(col, uTop, smoothstep(0.34, 0.9, h));
                    float s = max(dot(d, uSunDir), 0.0);
                    // Broad warm haze around the sun, a brighter halo, and the disc itself.
                    col += uSun * (pow(s, 4.0) * 0.32 + pow(s, 48.0) * 0.55 + smoothstep(0.9982, 0.9994, s) * 1.2);
                    if (h < 0.0) col = uHorizon; // below the horizon is haze, never open sky
                    gl_FragColor = vec4(min(col, vec3(1.0)), 1.0);
                }`,
        })
    );
    skyGroup.add(dome);

    // Distant mountain ranges, layered like a painted landscape: each range is
    // one band all the way around the stage whose skyline is a run of peaks —
    // triangular, but with rough, uneven slopes rather than straight sides —
    // fading from a soft blue crest down into the pale horizon. The back range
    // is taller and paler, the front one lower and bluer. Seeded, so it's the
    // same skyline every time. Colours are painted per vertex and shown unlit,
    // like a backdrop.
    const horizonCol = new THREE.Color(0xeef0dc);
    const mountainMat = new THREE.MeshBasicMaterial({ vertexColors: true, fog: false, toneMapped: false, side: THREE.DoubleSide });
    const seededRandom = (seed) => () => { // mulberry32
        seed = (seed + 0x6D2B79F5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const mountainRange = ({ seed, radius, peaks, minH, maxH, base, color }) => {
        const rand = seededRandom(seed);
        const summits = Array.from({ length: peaks }, (_, i) => ({
            at: ((i + rand() * 0.8) / peaks) * Math.PI * 2,
            h: minH + rand() * (maxH - minH),
            halfWidth: 0.16 + rand() * 0.22,   // radians
            lean: 0.7 + rand() * 0.6,          // one slope steeper than the other
        }));
        const ph = [rand(), rand(), rand()].map(p => p * Math.PI * 2);
        const skyline = (ang) => {
            let h = base;
            for (const s of summits) {
                let d = Math.atan2(Math.sin(ang - s.at), Math.cos(ang - s.at)); // wrapped to -π..π
                d *= d < 0 ? s.lean : 1 / s.lean;
                const f = 1 - Math.abs(d) / s.halfWidth;
                if (f > 0) h = Math.max(h, s.h * f);
            }
            // Rough slopes: small bumps along the ridge, scaled with height.
            const rough = Math.sin(ang * 23 + ph[0]) * 0.5 + Math.sin(ang * 57 + ph[1]) * 0.3 + Math.sin(ang * 131 + ph[2]) * 0.2;
            return h * (1 + 0.06 * rough);
        };
        const SEG = 720, ROWS = 4, perColumn = ROWS + 1;
        const positions = [], colors = [], index = [];
        const peakCol = new THREE.Color(color), c = new THREE.Color();
        for (let i = 0; i <= SEG; i++) {
            const ang = (i / SEG) * Math.PI * 2;
            const top = skyline(ang);
            const x = Math.cos(ang) * radius, z = Math.sin(ang) * radius;
            for (let r = 0; r <= ROWS; r++) {
                const t = r / ROWS;
                positions.push(x, -60 + (top + 60) * t, z);
                c.copy(horizonCol).lerp(peakCol, Math.pow(t * Math.min(1, top / maxH), 0.6));
                colors.push(c.r, c.g, c.b);
            }
        }
        for (let i = 0; i < SEG; i++) {
            for (let r = 0; r < ROWS; r++) {
                const a = i * perColumn + r, b = a + perColumn;
                index.push(a, b, a + 1, b, b + 1, a + 1);
            }
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geo.setIndex(index);
        skyGroup.add(new THREE.Mesh(geo, mountainMat));
    };
    mountainRange({ seed: 7, radius: 5600, peaks: 14, minH: 520, maxH: 950, base: 180, color: 0xa9c6e6 });
    mountainRange({ seed: 23, radius: 4700, peaks: 11, minH: 300, maxH: 620, base: 90, color: 0x7ea6d1 });

    // Clouds: soft round puffs painted on a canvas, bright white on top with
    // a cool blue-grey underside, as billboards drifting above the
    // mountains — more of them gathered toward the sun.
    const cloudTexture = (() => {
        const c = document.createElement('canvas');
        c.width = 512; c.height = 256;
        const ctx = c.getContext('2d');
        for (let i = 0; i < 22; i++) {
            const x = 70 + Math.random() * 372, y = 110 + Math.random() * 70 - Math.abs(x - 256) * 0.18;
            const r = 40 + Math.random() * 55;
            const g = ctx.createRadialGradient(x, y, 0, x, y, r);
            g.addColorStop(0, 'rgba(255,255,255,0.95)');
            g.addColorStop(0.6, 'rgba(255,255,255,0.7)');
            g.addColorStop(1, 'rgba(255,255,255,0)');
            ctx.fillStyle = g;
            ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalCompositeOperation = 'source-atop';
        const tint = ctx.createLinearGradient(0, 40, 0, 230);
        tint.addColorStop(0, 'rgba(255,255,255,0)');
        tint.addColorStop(1, 'rgba(232,216,190,0.5)');
        ctx.fillStyle = tint;
        ctx.fillRect(0, 0, 512, 256);
        const t = new THREE.CanvasTexture(c);
        t.colorSpace = THREE.SRGBColorSpace;
        return t;
    })();
    const sunAngle = Math.atan2(SKY_SUN_DIR.z, SKY_SUN_DIR.x);
    for (let i = 0; i < 18; i++) {
        const a = i < 8 ? sunAngle + (Math.random() - 0.5) * 1.6 : Math.random() * Math.PI * 2;
        const dist = 3400 + Math.random() * 1500;
        const cloud = new THREE.Sprite(new THREE.SpriteMaterial({
            map: cloudTexture, transparent: true, depthWrite: false, fog: false, toneMapped: false,
            opacity: 0.75 + Math.random() * 0.25,
        }));
        const w = 1300 + Math.random() * 1500;
        cloud.scale.set(w, w * 0.5, 1);
        cloud.position.set(Math.cos(a) * dist, 900 + Math.random() * 1300, Math.sin(a) * dist);
        skyGroup.add(cloud);
    }
}
scene.add(skyGroup);

// The grade. Rendering goes to a linear HDR target (multisampled, so edges
// stay as smooth as the plain canvas), soft bloom lets bright sky and
// sunlit surfaces glow a little, OutputPass converts to screen colour, and
// the grade pass finishes in screen space where its numbers read naturally:
// saturation plus vibrance (muted colours lifted more than already-vivid
// ones), gentle contrast around mid-grey, cool lift in the shadows and a
// warm gain in the highlights.
const composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 }));
composer.addPass(new RenderPass(scene, camera));
// Threshold above 1: only the sky's sun glow and truly hot highlights bloom —
// at 0.9 the character's toon skin crossed it and hands and face glowed white.
composer.addPass(new UnrealBloomPass(new THREE.Vector2(viewport.clientWidth, viewport.clientHeight), 0.15, 0.5, 1.05));
composer.addPass(new OutputPass());
composer.addPass(new ShaderPass({
    uniforms: {
        tDiffuse: { value: null },
        uSat: { value: 1.12 }, uVibrance: { value: 0.16 }, uContrast: { value: 1.07 }, uBright: { value: 1.03 },
        uLift: { value: new THREE.Vector3(0.015, 0.025, 0.05) }, uGain: { value: new THREE.Vector3(1.03, 1.01, 0.97) },
    },
    vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `uniform sampler2D tDiffuse; uniform float uSat, uVibrance, uContrast, uBright; uniform vec3 uLift, uGain; varying vec2 vUv;
        void main() {
            vec4 src = texture2D(tDiffuse, vUv);
            vec3 col = src.rgb;
            float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
            float chroma = max(col.r, max(col.g, col.b)) - min(col.r, min(col.g, col.b));
            col = mix(vec3(luma), col, uSat + uVibrance * (1.0 - chroma));
            col = (col - 0.5) * uContrast + 0.5;
            col = col * uGain + uLift * (1.0 - col);
            gl_FragColor = vec4(clamp(col * uBright, 0.0, 1.0), src.a);
        }`,
}));
composer.setPixelRatio(window.devicePixelRatio);
composer.setSize(viewport.clientWidth, viewport.clientHeight);
let useGrade = false;
// Every frame of the main loop, the orbit render and the recorder comes
// through here, so a Create video carries the same grade as the viewport.
function renderFrame() {
    if (useGrade) {
        fadeOccluders(); // right before drawing, so no frame (viewport or video) ever shows the character behind a prop
        return composer.render();
    }
    return renderer.render(scene, camera)
}

// Before a new scene is revealed: upload every texture and compile every
// shader, then draw one full frame. Without this the GPU did that work on the
// first visible frames, so props popped in just after playback had started.
async function warmUpScene() {
    const textures = new Set();
    scene.traverse(o => {
        if (!o.material) return;
        for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
            if (m.map) textures.add(m.map);
            if (m.emissiveMap) textures.add(m.emissiveMap);
        }
    });
    textures.forEach(t => renderer.initTexture(t));
    try { await renderer.compileAsync(scene, camera); } catch { renderer.compile(scene, camera); }
    renderFrame();
}

const studioLook = {
    background: scene.background.clone(),
    ambient: { color: sceneAmbientColor(), intensity: 1.5 },
    key: { color: keyLight.color.clone(), intensity: keyLight.intensity, position: keyLight.position.clone() },
    fill: { color: fillLight.color.clone(), intensity: fillLight.intensity },
    rim: { color: rimLight.color.clone(), intensity: rimLight.intensity },
};
function sceneAmbientColor() {
    const amb = scene.children.find(o => o.isAmbientLight);
    return amb ? amb.color.clone() : new THREE.Color(0x303050);
}
let createFog = null; // this scene's haze, kept so it comes back on returning from Chat

// The meadow runs on past the platform to the haze, so the stage sits in a
// landscape instead of floating as a square island in the sky. It sits just
// below the platform's ground and stays put (unlike the sky, it has texture
// that would visibly slide). Built on first use, once the ground helpers
// it relies on exist.
let farMeadow = null;

function setCreateLook(on) {
    const amb = scene.children.find(o => o.isAmbientLight);
    if (on && !farMeadow) {
        farMeadow = createTexturedGround(16000, CREATE_GROUND_COLOR);
        farMeadow.position.y = -0.5;
        scene.add(farMeadow);
    }
    if (farMeadow) farMeadow.visible = on;
    skyGroup.visible = on;
    useGrade = on;
    if (on) {
        // Kept close to Chat's overall light level on purpose: brighter
        // fill flattened the character's toon shading to white skin and a
        // grey hoodie. Props stay bright through their own soft glow (see
        // matteGLBMaterials) and the grade instead.
        if (amb) { amb.color.set(0xe8f0ff); amb.intensity = 0.35; }
        keyLight.color.set(0xfff4e2); keyLight.intensity = 2.0; keyLight.position.set(250, 420, 320);
        fillLight.color.set(0xcfe0ff); fillLight.intensity = 0.45;
        rimLight.color.set(0xffffff); rimLight.intensity = 0.5;
        scene.fog = createFog;
    } else {
        if (amb) { amb.color.copy(studioLook.ambient.color); amb.intensity = studioLook.ambient.intensity; }
        keyLight.color.copy(studioLook.key.color); keyLight.intensity = studioLook.key.intensity; keyLight.position.copy(studioLook.key.position);
        fillLight.color.copy(studioLook.fill.color); fillLight.intensity = studioLook.fill.intensity;
        rimLight.color.copy(studioLook.rim.color); rimLight.intensity = studioLook.rim.intensity;
        scene.fog = null;
        scene.background = studioLook.background.clone();
    }
}

// Aerial haze scaled to the platform: props at the rim soften a little, the
// mountains and sky beyond melt into the pale blue horizon.
function setCreateFog(groundSize) {
    createFog = new THREE.Fog(0xe9efe0, groundSize * 1.0, groundSize * 4 + 2000);
    if (skyGroup.visible) scene.fog = createFog;
}

// Create scenes stand on fresh grass whatever ground colour Gemini picked —
// except a city, which gets trodden town ground: less green, warmer, with
// cobbles (see createTexturedGround's 'town' style).
const CREATE_GROUND_COLOR = '#8dbf68'; // the grade adds saturation on top; brighter than this turned lime
const TOWN_GROUND_COLOR = '#b5b08f';
function stylizeGroundColor(style) {
    return style === 'town' ? TOWN_GROUND_COLOR : CREATE_GROUND_COLOR;
}

// Keep the character in sight: any backdrop prop whose actual 3D extent
// (crown and roof included) cuts a sight line from the camera to the
// character's head, middle or feet — or that the camera is standing in — is
// hidden at once, and fades back in once the view clears. A trunk-distance
// test missed wide crowns overhead, which is exactly what hid the character
// when orbiting. Runs every frame in Create: a box and three ray tests per prop.
const _occluderTarget = new THREE.Vector3();
const _occluderBox = new THREE.Box3(), _occluderRay = new THREE.Ray(), _occluderHit = new THREE.Vector3();
const _occluderPoints = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
let _occluderHips = null, _occluderHipsOf = null;
function fadeOccluders() {
    if (!skyGroup.visible || !currentBones) return;
    // Aim at the hips — the part that actually travels — not the skeleton's
    // root, which stays at the spawn point while the character walks away.
    if (_occluderHipsOf !== currentBones) {
        _occluderHipsOf = currentBones;
        _occluderHips = null;
        currentBones.traverse(n => { if (!_occluderHips && /hips/i.test(n.name)) _occluderHips = n; });
    }
    (_occluderHips || currentBones).getWorldPosition(_occluderTarget);
    const cam = camera.position;
    for (let i = 0; i < 3; i++) _occluderPoints[i].copy(_occluderTarget).y += [70, 0, -80][i]; // head, hips, feet
    for (const o of sceneObjects) {
        if (!o.userData._scenery) continue;
        // Padded a little so the prop's edge clears the character's outline.
        _occluderBox.setFromObject(o).expandByScalar(12);
        // Standing in (or right against) a prop, the lens sees only its inside.
        const aroundCamera = _occluderBox.distanceToPoint(cam) < 40;
        let blocking = false;
        for (let i = 0; i < 3 && !blocking && !aroundCamera; i++) {
            _occluderRay.origin.copy(cam);
            _occluderRay.direction.subVectors(_occluderPoints[i], cam);
            const len = _occluderRay.direction.length() || 1;
            _occluderRay.direction.divideScalar(len);
            const hit = _occluderRay.intersectBox(_occluderBox, _occluderHit);
            blocking = hit !== null && cam.distanceTo(hit) < len;
        }
        // Hidden outright rather than ghosted: a see-through crown in front
        // of the lens still smothered the frame.
        const target = aroundCamera || blocking ? 0 : 1;
        const current = o.userData._opacity ?? 1;
        if (Math.abs(current - target) < 0.005) continue;
        // Get out of the way at once (easing down left the character hidden
        // for a split second); come back gently once the view is clear.
        const next = target < current || Math.abs(target - current) < 0.02 ? target : current + (target - current) * 0.18;
        o.userData._opacity = next;
        o.visible = next > 0.02;
        o.traverse(child => {
            if (!child.isMesh || !child.material) return;
            for (const mat of Array.isArray(child.material) ? child.material : [child.material]) {
                mat.transparent = next < 1;
                mat.opacity = next;
                mat.depthWrite = next > 0.5;
            }
        });
    }
}

setCreateLook(document.body.dataset.mode !== 'help');

let mixer = null;
let currentBones = null;
let characterGroup = null;
let bodyMeshes = [];
// Selection & animation control state
let selectedObject = null, selectedType = null;
let selectionBox = null; // purple wireframe box around selected object

function showSelectionBox(obj) {
    removeSelectionBox();
    const box = new THREE.Box3().setFromObject(obj);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    // Add a little padding
    size.multiplyScalar(1.08);
    const geo = new THREE.BoxGeometry(size.x, size.y, size.z);
    const edges = new THREE.EdgesGeometry(geo);
    const mat = new THREE.LineBasicMaterial({ color: 0x9b7fd4, transparent: true, opacity: 1 });
    selectionBox = new THREE.LineSegments(edges, mat);
    selectionBox.position.copy(center);
    selectionBox.userData._isSelectionBox = true;
    scene.add(selectionBox);
}

function removeSelectionBox() {
    if (selectionBox) {
        scene.remove(selectionBox);
        selectionBox.geometry.dispose();
        selectionBox.material.dispose();
        selectionBox = null;
    }
}
// ========================================================================
// VRM AVATAR — a real character (see AVATAR_FILES) instead of
// the capsule mannequin, in Chat mode. Its meshes are skinned onto the
// character's live BVH skeleton (see buildVrmSkinnedMeshes), so the same
// mixer-driven bones that move the mannequin move her — no retargeting of
// rotations onto the VRM's own rig, which never lined up.
// ========================================================================
// Per character: the file, any of its own meshes to leave out, and a garment
// borrowed from the other model. Both are VRoid exports, so their meshes are
// split one-per-material with predictable names (Tops / Bottoms / Shoes /
// Accessory_*), which is what makes hiding and borrowing by material work.
const AVATARS = {
    male: {
        url: 'assets/character/avatar_male.vrm',
        // System voices are tried in order; the first one this browser has wins.
        // These are the good macOS/Windows ones — the default voice is the
        // robotic-sounding one, and it's only the default because nobody picked.
        systemVoices: ['Daniel', 'Alex', 'Tom', 'Google UK English Male', 'Microsoft David'],
        voiceGender: 'male',
    },
    female: {
        url: 'assets/character/avatar_female.vrm',
        hide: [/FoxTail/i, /Tops/i],                  // drop the tail, and her own sleeveless top
        borrow: { from: 'male', match: /Tops/i },     // wear his hoodie instead
        systemVoices: ['Samantha', 'Ava', 'Karen', 'Google UK English Female', 'Microsoft Zira'],
        voiceGender: 'female',
    },
};
const matchesMaterial = (obj, patterns) => {
    const names = (Array.isArray(obj.material) ? obj.material : [obj.material]).map(m => (m && m.name) || '');
    return patterns.some(re => names.some(n => re.test(n)));
};
let avatarKey = 'male';       // which character is on stage
const avatarCache = {};       // key -> { vrm, scale }; a switched-away model stays loaded, so switching back is instant
let avatarVRM = null;         // the active VRM — buildVrmSkinnedMeshes reads this and avatarScale
let avatarScale = 1;

// Load + cache a character without disturbing which one is on stage — the
// borrowed-garment donor has to be loaded this way.
async function avatarEntry(key) {
    if (avatarCache[key]) return avatarCache[key];
    const url = AVATARS[key].url;
    const gltf = await gltfLoader.loadAsync(url);
    const vrm = gltf.userData.vrm;
    if (!vrm) throw new Error(`${url} has no VRM data`);
    // Deliberately NOT calling VRMUtils.removeUnnecessaryJoints here: it rebuilds
    // each mesh's skeleton and rewrites skinIndex to match, and on geometry shared
    // between meshes that left vertices pointing at the wrong bone — chunks ended
    // up 0.5-1.5m from their joint. It's only a render optimization anyway, and
    // buildVrmSkinnedMeshes never renders the original SkinnedMeshes.
    if (vrm.meta?.metaVersion === '0') VRMUtils.rotateVRM0(vrm); // old VRM0 files face -Z; rotate to the +Z-forward convention everything else here assumes
    vrm.scene.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(vrm.scene);
    // Normalize whatever real-world scale the file was authored at to this
    // app's unit convention, per character — the two models aren't the same
    // height, and the skinning bakes this in.
    avatarCache[key] = { vrm, scale: CHAR_HEIGHT / Math.max(box.max.y - box.min.y, 0.01) };
    applySkinTone(document.body.dataset.mode === 'create');
    return avatarCache[key];
}

// VRoid marks its two skin materials with a _SKIN suffix (Face and Body);
// the mouth, brows and lashes are separate _FACE overlays and keep their own
// colour, so tinting those would just muddy the makeup. The models ship pale:
// under Chat's studio lights they read pink, and under Create's brighter,
// graded look they wash out almost white. Hence a multiplier per look rather
// than one flat colour.
// Given as sRGB hex rather than bare floats: THREE.Color reads raw components
// as linear-light, so an obvious-looking (0.95, 0.82, 0.70) encodes to #f9eada
// — a 2% dip in red, 14% in blue — which on screen is still simply pink.
// Multiplying two colours in linear space is the same as multiplying their
// sRGB values, so a hex here means what it looks like: keep this much of each
// channel of the texture, which averages a pale pink (219,178,167).
const SKIN_TINT = {
    chat: new THREE.Color('#e9deb7'),
    create: new THREE.Color('#d9c99e'),
};
const isSkinMaterial = (m) => /_SKIN/i.test((m && m.name) || '');

// Multiplied over the file's own factors, never assigned outright: the
// original is stashed on first touch so switching modes back and forth
// re-tints from the same base instead of compounding into mud.
function applySkinTone(createLook) {
    const tint = createLook ? SKIN_TINT.create : SKIN_TINT.chat;
    for (const entry of Object.values(avatarCache)) {
        entry?.vrm?.scene?.traverse(o => {
            if (!o.material) return;
            for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
                if (!isSkinMaterial(m)) continue;
                if (!m.userData._skinBase) {
                    m.userData._skinBase = {
                        color: m.color?.clone(),
                        shade: m.shadeColorFactor?.clone(), // MToon only
                    };
                }
                const base = m.userData._skinBase;
                if (base.color && m.color) m.color.copy(base.color).multiply(tint);
                if (base.shade && m.shadeColorFactor) m.shadeColorFactor.copy(base.shade).multiply(tint);
                m.needsUpdate = true;
            }
        });
    }
}

// Load a character and anything it borrows, so createCharacter (which is
// synchronous) can just read them out of the cache.
async function ensureAvatar(key) {
    const entry = await avatarEntry(key);
    const borrow = AVATARS[key].borrow;
    if (borrow) await avatarEntry(borrow.from);
    return entry;
}

// Chat's persona: same thing, but it also becomes the active one.
async function loadAvatarVRM(key = avatarKey) {
    const entry = await ensureAvatar(key);
    avatarVRM = entry.vrm;
    avatarScale = entry.scale;
    return avatarVRM;
}

// Create's cast alternates through the models, so a scene with several people
// isn't the same face repeated.
const CREATE_AVATAR_ORDER = ['male', 'female'];
const avatarForIndex = (i) => CREATE_AVATAR_ORDER[i % CREATE_AVATAR_ORDER.length];

// Swap the character on stage. Rebuilds the persona, because the VRM is
// skinned onto the BVH skeleton at build time — there's no live model to
// re-point. She greets again on the way in, same as any arrival.
async function setAvatarKey(key) {
    if (key === avatarKey || !AVATARS[key]) return;
    avatarKey = key;
    for (const b of document.querySelectorAll('.avatar-btn')) b.classList.toggle('active', b.dataset.avatar === key);
    const buttons = [...document.querySelectorAll('.avatar-btn')];
    buttons.forEach(b => { b.disabled = true; });
    try {
        await loadAvatarVRM(key); // first switch pays the download; cached after that
        await showChatPersona({ greet: true });
    } catch (err) {
        console.warn('Could not switch character:', err);
    }
    buttons.forEach(b => { b.disabled = false; });
}

// Which BVH bone each VRM humanoid role should rigidly follow. Bones the
// humanoid spec doesn't name (fingers, hair joints, eyes) fall back to the
// nearest mapped ancestor in resolveBvhName below, rather than being
// dropped — a strand of hair follows the head, a finger follows the hand.
const BVH_TO_VRM_BONE = {
    Hips: 'hips', Spine1: 'spine', Spine2: 'chest', Chest: 'upperChest',
    Neck1: 'neck', Head: 'head',
    LeftShoulder: 'leftShoulder', LeftArm: 'leftUpperArm', LeftForeArm: 'leftLowerArm', LeftHand: 'leftHand',
    RightShoulder: 'rightShoulder', RightArm: 'rightUpperArm', RightForeArm: 'rightLowerArm', RightHand: 'rightHand',
    LeftLeg: 'leftUpperLeg', LeftShin: 'leftLowerLeg', LeftFoot: 'leftFoot', LeftToeBase: 'leftToes',
    RightLeg: 'rightUpperLeg', RightShin: 'rightLowerLeg', RightFoot: 'rightFoot', RightToeBase: 'rightToes',
};

// Skins the VRM's meshes onto this character's live BVH skeleton and
// returns the SkinnedMeshes (added under the character's group, so
// clearAllCharacters's scene.remove(group) takes them along).
//
// The VRM's own skeleton is never posed. Each of its meshes gets a new
// skeleton made of the BVH bones, with inverse-bind matrices computed so the
// mesh's bind pose lands on the BVH rest pose bone by bone. Per bone that's
// ONE constant transform: express bind-pose geometry in a frame aligned to
// the VRM bone's direction (joint -> child joint), stretch it to the BVH
// segment length, and re-express it in the BVH bone's direction frame under
// that bone. Roll (twist about the bone) is pinned to something anatomical
// both rigs have — the thumb for hands, the shin for feet, the rig's own
// facing (read off its toes) for everything else — never a bare world axis,
// which is what flipped a shoe sole-up and a palm skyward. Skin weights come
// straight from the VRM (up to four bones per vertex, each VRM bone folded
// onto the BVH bone it maps to, fingers/hair joints onto their nearest
// mapped ancestor), so three.js blends across joints exactly as it would for
// the original: no seams at shoulders or hips, and no per-frame code — the
// mixer moves the bones, the GPU does the rest. (Retargeting BVH rotations
// onto the VRM's own skeleton was tried first and abandoned: the rigs' rest
// axes differ per bone. Cutting the mesh into rigid per-bone chunks worked
// but tore at every joint; this is that same alignment, smoothly skinned.)
function buildVrmSkinnedMeshes(vrm, bones, scale = avatarScale, includeMesh = null) {
    const rawToBvh = new Map(); // raw (bind-skeleton) bone Object3D -> BVH bone name
    for (const [bvhName, vrmName] of Object.entries(BVH_TO_VRM_BONE)) {
        const raw = vrm.humanoid.getRawBoneNode(vrmName);
        if (raw) rawToBvh.set(raw, bvhName);
    }
    const resolveCache = new Map();
    function resolveBvhName(rawBone) {
        if (resolveCache.has(rawBone)) return resolveCache.get(rawBone);
        let n = rawBone, name = null;
        while (n) { if (rawToBvh.has(n)) { name = rawToBvh.get(n); break; } n = n.parent; }
        resolveCache.set(rawBone, name);
        return name;
    }

    vrm.scene.updateMatrixWorld(true);
    const group = bones.parent || bones;
    group.updateMatrixWorld(true);

    // Joint world positions on both sides at their rest poses, plus the
    // roll-reference joints (thumbs) that aren't skinning targets themselves.
    const bvhBone = {}, bvhPos = {}, vrmPos = {};
    for (const [bvhName, vrmName] of Object.entries(BVH_TO_VRM_BONE)) {
        const b = findBone(bones, bvhName), r = vrm.humanoid.getRawBoneNode(vrmName);
        if (!b || !r) continue; // optional VRM bones (upperChest, toes) may simply not exist
        bvhBone[bvhName] = b;
        bvhPos[bvhName] = b.getWorldPosition(new THREE.Vector3());
        vrmPos[bvhName] = r.getWorldPosition(new THREE.Vector3());
    }
    const REF_JOINTS = { LeftThumb: ['LeftHandThumb1', ['leftThumbMetacarpal', 'leftThumbProximal']], RightThumb: ['RightHandThumb1', ['rightThumbMetacarpal', 'rightThumbProximal']] };
    for (const [key, [bvhName, vrmNames]] of Object.entries(REF_JOINTS)) {
        const b = findBone(bones, bvhName);
        const r = vrmNames.map(n => vrm.humanoid.getRawBoneNode(n)).find(Boolean);
        if (!b || !r) continue;
        bvhPos[key] = b.getWorldPosition(new THREE.Vector3());
        vrmPos[key] = r.getWorldPosition(new THREE.Vector3());
    }
    // A bone points at its child joint; end bones (head, hands, toes) continue
    // their parent's direction. The child chain skips joints one rig lacks.
    const DIR_CHILD = {
        Hips: 'Spine1', Spine1: 'Spine2', Spine2: 'Chest', Chest: 'Neck1', Neck1: 'Head',
        LeftShoulder: 'LeftArm', LeftArm: 'LeftForeArm', LeftForeArm: 'LeftHand',
        RightShoulder: 'RightArm', RightArm: 'RightForeArm', RightForeArm: 'RightHand',
        LeftLeg: 'LeftShin', LeftShin: 'LeftFoot', LeftFoot: 'LeftToeBase',
        RightLeg: 'RightShin', RightShin: 'RightFoot', RightFoot: 'RightToeBase',
    };
    const DIR_PARENT = { Head: 'Neck1', LeftHand: 'LeftForeArm', RightHand: 'RightForeArm', LeftToeBase: 'LeftFoot', RightToeBase: 'RightFoot' };
    // Roll references: the thumb for hands (palm orientation), the shin for
    // feet and toes (the top of the foot faces up the leg).
    const ROLL_REF = { LeftHand: 'LeftThumb', RightHand: 'RightThumb', LeftFoot: 'LeftShin', LeftToeBase: 'LeftShin', RightFoot: 'RightShin', RightToeBase: 'RightShin' };
    function childFor(name, pos) {
        let child = DIR_CHILD[name];
        while (child && !pos[child]) child = DIR_CHILD[child];
        return child || null;
    }
    function dirFor(name, pos) {
        const child = childFor(name, pos);
        if (child) return pos[child].clone().sub(pos[name]).normalize();
        const parent = DIR_PARENT[name];
        if (parent && pos[parent]) return pos[name].clone().sub(pos[parent]).normalize();
        return new THREE.Vector3(0, 1, 0);
    }
    // Each rig's own facing, read off its feet (toes are in front of ankles)
    // rather than assumed: the raw VRM0 skeleton natively faces -Z while the
    // BVH persona faces +Z.
    function forwardFrom(pos) {
        const f = new THREE.Vector3();
        for (const [foot, toe] of [['LeftFoot', 'LeftToeBase'], ['RightFoot', 'RightToeBase']]) {
            if (pos[foot] && pos[toe]) f.add(pos[toe].clone().sub(pos[foot]));
        }
        f.y = 0;
        return f.lengthSq() > 1e-8 ? f.normalize() : null;
    }
    const fwdBvh = forwardFrom(bvhPos) || new THREE.Vector3(0, 0, 1);
    const fwdVrm = forwardFrom(vrmPos) || new THREE.Vector3(0, 0, vrm.meta?.metaVersion === '0' ? -1 : 1);
    const UP = new THREE.Vector3(0, 1, 0);
    // Orthonormal frame: x along the bone, roll pinned to `ref`.
    function basisFor(d, ref) {
        const b2 = ref.clone().sub(d.clone().multiplyScalar(ref.dot(d))).normalize();
        const b3 = new THREE.Vector3().crossVectors(d, b2);
        return new THREE.Matrix4().makeBasis(d, b2, b3);
    }
    // The SAME kind of reference on both rigs: an anatomical joint where one
    // exists, else the rig's facing — or world up when the bone itself points
    // forward, decided once from the VRM side so the two rigs can't disagree
    // (a threshold flipping on one side only is what inverted one shoe).
    function rollRefs(name, dV) {
        const rn = ROLL_REF[name];
        if (rn && vrmPos[rn] && bvhPos[rn]) return [vrmPos[rn].clone().sub(vrmPos[name]), bvhPos[rn].clone().sub(bvhPos[name])];
        return Math.abs(dV.dot(fwdVrm)) > 0.9 ? [UP, UP] : [fwdVrm, fwdBvh];
    }

    // Where each bone's geometry is anchored in BVH space. For limbs that's
    // simply the matching BVH joint. The spine is different: the two rigs
    // space their torso joints very differently (measured on both characters
    // - the VRM puts `chest` at 41% of the hips->neck span and `upperChest`
    // at 67%, where the BVH rig puts Spine2 at 26% and Chest at 43%).
    // Anchoring VRM chest geometry on BVH Spine2 therefore drags the whole
    // upper torso down a seventh of its height and bunches the surface at the
    // waist, which shows from the side as a crease in the belly. So torso
    // geometry is anchored at the same PROPORTION along the BVH spine that it
    // occupies along the VRM's own; the BVH bone still drives its rotation.
    const TORSO_CHAIN = ['Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1'];
    const anchor = {};
    for (const name of Object.keys(bvhBone)) anchor[name] = bvhPos[name];
    const torso = TORSO_CHAIN.filter(n => bvhPos[n] && vrmPos[n]);
    if (torso.length >= 3) {
        const arc = (pts) => { const a = [0]; for (let i = 1; i < pts.length; i++) a.push(a[i - 1] + pts[i - 1].distanceTo(pts[i])); return a; };
        const bPts = torso.map(n => bvhPos[n]), vPts = torso.map(n => vrmPos[n]);
        const bAcc = arc(bPts), vAcc = arc(vPts);
        const bTot = bAcc[bAcc.length - 1], vTot = vAcc[vAcc.length - 1];
        const along = (pts, target) => {
            let acc = 0;
            for (let i = 0; i < pts.length - 1; i++) {
                const seg = pts[i].distanceTo(pts[i + 1]);
                if (acc + seg >= target || i === pts.length - 2) {
                    return pts[i].clone().lerp(pts[i + 1], seg > 1e-6 ? THREE.MathUtils.clamp((target - acc) / seg, 0, 1) : 0);
                }
                acc += seg;
            }
            return pts[pts.length - 1].clone();
        };
        // Endpoints land on themselves (fraction 0 and 1), so only the
        // intermediate spine joints actually move.
        if (bTot > 1e-6 && vTot > 1e-6) torso.forEach((n, i) => { anchor[n] = along(bPts, (vAcc[i] / vTot) * bTot); });
    }

    // One BVH-bone skeleton shared by every mesh. Per bone, the inverse-bind
    // matrix is inverse(BVH rest world) * M, where M takes a bind-pose world
    // point to where it should sit at BVH rest:
    //   translate to the anchor <- BVH bone frame <- stretch/scale <- VRM bone frame^T <- translate from VRM joint
    const boneList = [], boneInverses = [], boneIndexOf = {};
    const _t = new THREE.Matrix4();
    for (const name of Object.keys(bvhBone)) {
        const dV = dirFor(name, vrmPos), dB = dirFor(name, bvhPos);
        const [refV, refB] = rollRefs(name, dV);
        const Bv = basisFor(dV, refV), Bb = basisFor(dB, refB);
        const cb = childFor(name, bvhPos), cv = childFor(name, vrmPos);
        let along = scale;
        if (cb && cv) {
            const lenB = bvhPos[cb].distanceTo(bvhPos[name]), lenV = vrmPos[cv].distanceTo(vrmPos[name]);
            if (lenV > 1e-4) along = THREE.MathUtils.clamp(lenB / lenV, scale * 0.6, scale * 1.6);
        }
        const M = new THREE.Matrix4().makeTranslation(anchor[name].x, anchor[name].y, anchor[name].z)
            .multiply(Bb)
            .multiply(_t.makeScale(along, scale, scale))
            .multiply(Bv.clone().transpose())
            .multiply(_t.makeTranslation(-vrmPos[name].x, -vrmPos[name].y, -vrmPos[name].z));
        boneIndexOf[name] = boneList.length;
        boneList.push(bvhBone[name]);
        boneInverses.push(new THREE.Matrix4().copy(bvhBone[name].matrixWorld).invert().multiply(M));
    }
    const skeleton = new THREE.Skeleton(boneList, boneInverses);
    const fallbackIdx = boneIndexOf.Hips ?? 0;

    const meshes = [];
    const _p = new THREE.Vector3(), _n = new THREE.Vector3();
    vrm.scene.traverse(obj => {
        if (!obj.isSkinnedMesh) return;
        if (includeMesh && !includeMesh(obj)) return;
        const geo = obj.geometry;
        const posAttr = geo.attributes.position, normAttr = geo.attributes.normal;
        const skinIdxAttr = geo.attributes.skinIndex, skinWeightAttr = geo.attributes.skinWeight;
        const src = obj.skeleton;
        if (!skinIdxAttr || !skinWeightAttr) return;

        // The skinning shader's per-bone matrix for this mesh's bind pose —
        // bone.matrixWorld * inverseBind * bindMatrix takes a mesh-local
        // vertex to its bind-pose WORLD position (the VRM is untouched, so
        // its bones ARE at bind). Cached per bone index on first use.
        const boneMats = [], normalMats = [];
        const matsFor = (bi) => {
            if (!boneMats[bi]) {
                boneMats[bi] = new THREE.Matrix4().multiplyMatrices(src.bones[bi].matrixWorld, src.boneInverses[bi]).multiply(obj.bindMatrix);
                normalMats[bi] = new THREE.Matrix3().setFromMatrix4(boneMats[bi]);
            }
            return bi;
        };
        const remap = src.bones.map(b => { const n = resolveBvhName(b); return n && boneIndexOf[n] !== undefined ? boneIndexOf[n] : fallbackIdx; });

        const count = posAttr.count;
        const newPos = new Float32Array(count * 3);
        const newNorm = normAttr ? new Float32Array(count * 3) : null;
        const newIdx = new Uint16Array(count * 4);
        const newW = new Float32Array(count * 4);
        for (let v = 0; v < count; v++) {
            let bestW = -1, bestI = 0;
            for (let k = 0; k < 4; k++) {
                const w = skinWeightAttr.getComponent(v, k);
                if (w > bestW) { bestW = w; bestI = skinIdxAttr.getComponent(v, k); }
            }
            const bi = matsFor(bestI);
            _p.fromBufferAttribute(posAttr, v).applyMatrix4(boneMats[bi]);
            newPos[v * 3] = _p.x; newPos[v * 3 + 1] = _p.y; newPos[v * 3 + 2] = _p.z;
            if (normAttr) {
                _n.fromBufferAttribute(normAttr, v).applyMatrix3(normalMats[bi]).normalize();
                newNorm[v * 3] = _n.x; newNorm[v * 3 + 1] = _n.y; newNorm[v * 3 + 2] = _n.z;
            }
            for (let k = 0; k < 4; k++) {
                const ri = remap[skinIdxAttr.getComponent(v, k)];
                newIdx[v * 4 + k] = ri === undefined ? fallbackIdx : ri;
                newW[v * 4 + k] = skinWeightAttr.getComponent(v, k);
            }
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(newPos, 3));
        if (newNorm) g.setAttribute('normal', new THREE.BufferAttribute(newNorm, 3));
        if (geo.attributes.uv) g.setAttribute('uv', geo.attributes.uv.clone());
        g.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(newIdx, 4));
        g.setAttribute('skinWeight', new THREE.BufferAttribute(newW, 4));
        if (geo.index) g.setIndex(geo.index.clone());
        for (const grp of geo.groups) g.addGroup(grp.start, grp.count, grp.materialIndex); // multi-material VRoid meshes keep their material ranges

        const mesh = new THREE.SkinnedMesh(g, obj.material);
        mesh.bind(skeleton, new THREE.Matrix4()); // explicit identity bind matrix — bind() must not recompute the inverses from the live pose
        mesh.castShadow = true;
        mesh.frustumCulled = false; // bounds are the bind pose, not the animated one
        group.add(mesh);
        meshes.push(mesh);
    });
    return meshes;
}
let currentClip = null, currentAction = null;
let isPlaying = true, isScrubbing = false;
let playbackFinished = false; // true once the current playthrough has run to completion
let lastBvhText = null;
let timelineClips = [];
let totalDuration = 0;
let groundMesh = null; // Reference to ground plane for dynamic repositioning
const clock = new THREE.Clock();

// ========================================================================
// SHARED PLAYBACK CLOCK — drives the character AND any object-path
// animations (e.g. a car driving) off one clock, so everything plays once
// and pauses together instead of looping forever. See updateObjectAnimations().
// ========================================================================
let activeObjectAnimations = []; // [{ object3D, path, duration, basePos, baseYaw }]
// Object-driven beats from the opening prompt ("a car arrives") — parallel
// to characters[].clips, but for the timeline row an object-primary scene
// has no character to hang one off of. [{ keyword, prompt, duration }]
let objectBeats = [];
let objectMixer = null;   // dummy AnimationMixer used purely as a shared clock + 'finished' source
let objectClockAction = null;
let chatPersonaIdle = false;   // true while the Chat persona is on stage rather than demoing
let personaTalking = false;    // she only gestures while an answer is being delivered
let personaIdleAction = null;  // her standing-and-breathing loop — always running while chatPersonaIdle
let personaTalkAction = null;  // her talking-and-gesturing loop — crossfaded in over the idle one
let personaWaveAction = null;  // her one-shot hello, played on arriving in Chat
let personaGreetToken = 0;     // invalidates a pending wave->idle hand-off if anything supersedes the greeting
let helpDemoObject = null;     // the standalone object spawned for a Help-mode object_path demo

function createPlaybackClock(duration) {
    if (objectMixer) objectMixer.stopAllAction();
    const dummy = new THREE.Object3D();
    const track = new THREE.NumberKeyframeTrack('.rotation[x]', [0, Math.max(duration, 0.1)], [0, 0]);
    const clip = new THREE.AnimationClip('clock', Math.max(duration, 0.1), [track]);
    objectMixer = new THREE.AnimationMixer(dummy);
    objectClockAction = objectMixer.clipAction(clip);
    objectClockAction.setLoop(THREE.LoopOnce, 1);
    objectClockAction.clampWhenFinished = true;
    objectClockAction.play();
    objectMixer.addEventListener('finished', onPlaybackFinished);
    totalDuration = duration;
}

function onPlaybackFinished() {
    // The persona's own actions loop on their own (LoopRepeat, crossfaded —
    // see setPersonaTalking); this "story clock" finishing just means her
    // first cycle elapsed, nothing needs restarting. Doing that here (as a
    // hard action.reset()+setTime(0)) is exactly what used to make her visibly
    // snap back to frame 0 every few seconds.
    // ...but only while Chat is actually on screen. chatPersonaIdle stays true
    // after switching to Create, and an unqualified return here meant Create's
    // own clips never registered as finished — the play button kept showing
    // "playing" and Replay wouldn't restart them.
    if (chatPersonaIdle && document.body.dataset.mode === 'help') return;
    isPlaying = false;
    playbackFinished = true;
    updatePlayPauseIcon();
}

function showGenLoadingOverlay(text) {
    const el = document.getElementById('gen-loading-overlay');
    document.getElementById('gen-loading-text').textContent = text || 'Generating...';
    el.classList.add('visible');
}
function hideGenLoadingOverlay() {
    document.getElementById('gen-loading-overlay').classList.remove('visible');
}

function updatePlayPauseIcon() {
    const btn = document.getElementById('tl-playpause');
    if (!btn) return;
    btn.innerHTML = isPlaying
        ? '<svg width="10" height="12" viewBox="0 0 10 12"><rect x="1" y="0" width="2.5" height="12" rx="0.5" fill="currentColor"/><rect x="6.5" y="0" width="2.5" height="12" rx="0.5" fill="currentColor"/></svg>'
        : '<svg width="10" height="12" viewBox="0 0 10 12"><path d="M1 0.5v11l8.5-5.5z" fill="currentColor"/></svg>';
}

// Restart the current playthrough from the beginning (used by the Play
// button after a clip has finished, and when a fresh clip starts).
function restartPlayback() {
    playbackFinished = false;
    isPlaying = true;
    // Rewind every character so the whole cast replays in sync.
    for (const c of characters) {
        c.action.reset();
        c.action.play();
        c.mixer.setTime(0);
    }
    if (characters.length === 0) {
        if (currentAction) { currentAction.reset(); currentAction.play(); }
        if (mixer) mixer.setTime(0);
    }
    if (objectClockAction) { objectClockAction.reset(); objectClockAction.play(); }
    // mixer.time/objectMixer.time are running totals that action.reset() does NOT
    // rewind — setTime(0) is what actually zeroes the clock updatePlayhead() reads.
    if (objectMixer) objectMixer.setTime(0);
    updateObjectAnimations(0);
    updatePlayPauseIcon();
}

// Play from `wallSeconds` into the story instead of the top — used right after
// a beat is added, so you see the new part straight away (Replay still plays
// the whole thing). Character mixers run at CHAR_TIME_SCALE, and setTime goes
// through that scale, so the same wall-clock time lines every clock up.
function playFrom(wallSeconds) {
    restartPlayback();
    const t = Math.max(0, Math.min(wallSeconds, totalDuration - 0.05));
    if (t <= 0) return;
    for (const c of characters) c.mixer.setTime(t);
    if (characters.length === 0 && mixer) mixer.setTime(t);
    if (objectMixer) objectMixer.setTime(t);
    updateObjectAnimations(t);
}

// A timeline seek: the whole scene to `wallSeconds`, still playing or paused
// as it was. Every clock moves together, so nobody is left out of step.
function seekPlayback(wallSeconds) {
    const wasPlaying = isPlaying && !playbackFinished;
    playFrom(wallSeconds);
    isPlaying = wasPlaying;
    updatePlayPauseIcon();
}

// Interpolate every tracked object's position/rotation at time t (seconds)
// along its authored path. Objects hold their final pose once t exceeds
// their own path duration.
function lerpPath(path, t) {
    if (!path || path.length === 0) return null;
    if (t <= path[0].t) return path[0];
    if (t >= path[path.length - 1].t) return path[path.length - 1];
    for (let i = 0; i < path.length - 1; i++) {
        const a = path[i], b = path[i + 1];
        if (t >= a.t && t <= b.t) {
            const span = (b.t - a.t) || 1;
            const f = (t - a.t) / span;
            return {
                pos: [
                    a.pos[0] + (b.pos[0] - a.pos[0]) * f,
                    a.pos[1] + (b.pos[1] - a.pos[1]) * f,
                    a.pos[2] + (b.pos[2] - a.pos[2]) * f,
                ],
                yaw: a.yaw + (b.yaw - a.yaw) * f,
            };
        }
    }
    return path[path.length - 1];
}

function updateObjectAnimations(t) {
    for (const anim of activeObjectAnimations) {
        const frame = lerpPath(anim.path, Math.min(t, anim.duration));
        if (!frame) continue;
        anim.object3D.position.set(
            anim.basePos.x + frame.pos[0],
            anim.basePos.y + frame.pos[1],
            anim.basePos.z + frame.pos[2]
        );
        anim.object3D.rotation.y = anim.baseYaw + THREE.MathUtils.degToRad(frame.yaw);
    }
}

// Body segments: [fromBone, toBone, radiusTop, radiusBottom]
// Modeled after a wooden drawing mannequin
const BODY_SEGMENTS = [
    // Full torso — one smooth piece, broader at top
    ['Hips', 'Neck1', 10, 7],
    // Neck
    ['Neck1', 'Head', 3, 3],
    // Shoulders
    ['Chest', 'LeftShoulder', 5, 4],
    ['Chest', 'RightShoulder', 5, 4],
    // Left arm
    ['LeftShoulder', 'LeftArm', 4, 3.5],
    ['LeftArm', 'LeftForeArm', 3.5, 3],
    ['LeftForeArm', 'LeftHand', 3, 2],
    // Right arm
    ['RightShoulder', 'RightArm', 4, 3.5],
    ['RightArm', 'RightForeArm', 3.5, 3],
    ['RightForeArm', 'RightHand', 3, 2],
    // Left leg
    ['LeftLeg', 'LeftShin', 5.5, 4],
    ['LeftShin', 'LeftFoot', 4, 3],
    ['LeftFoot', 'LeftToeBase', 3, 2],
    // Right leg
    ['RightLeg', 'RightShin', 5.5, 4],
    ['RightShin', 'RightFoot', 4, 3],
    ['RightFoot', 'RightToeBase', 3, 2],
    // Hip to leg — thin peg connectors
    ['Hips', 'LeftLeg', 3.5, 3.5],
    ['Hips', 'RightLeg', 3.5, 3.5],
];

const bodyColor = 0xd4b896; // wooden mannequin color

function findBone(root, name) {
    if (root.name === name) return root;
    for (const child of root.children) {
        const found = findBone(child, name);
        if (found) return found;
    }
    return null;
}

// Builds capsule/head/hand meshes for a skeleton, added to `scene` and
// returned as a fresh array. Every character gets its own independent
// array so they don't stomp on each other.
function buildBodyMeshesFor(rootBone) {
    const arr = [];
    const mat = new THREE.MeshStandardMaterial({
        color: bodyColor, roughness: 0.75, metalness: 0.0,
    });

    // Head — elongated sphere
    const headBone = findBone(rootBone, 'HeadEnd');
    if (headBone) {
        const geo = new THREE.SphereGeometry(1, 24, 20);
        const mesh = new THREE.Mesh(geo, mat.clone());
        mesh.userData.type = 'head';
        mesh.userData.bone = headBone;
        mesh.userData.baseBone = findBone(rootBone, 'Head');
        scene.add(mesh);
        arr.push(mesh);
    }

    // Smooth tapered cylinders between bone pairs
    const tv1 = new THREE.Vector3();
    const tv2 = new THREE.Vector3();

    for (const [fromName, toName, rTop, rBot] of BODY_SEGMENTS) {
        const fromBone = findBone(rootBone, fromName);
        const toBone = findBone(rootBone, toName);
        if (!fromBone || !toBone) continue;

        fromBone.getWorldPosition(tv1);
        toBone.getWorldPosition(tv2);
        const dist = tv1.distanceTo(tv2);

        // Tapered cylinder with hemisphere caps via LatheGeometry
        const height = Math.max(dist, 1);
        const segments = 20; // smoother capsules
        // Create smooth profile: bottom cap → cylinder → top cap
        const points = [];
        const capSteps = 8;
        // Bottom hemisphere cap
        for (let i = 0; i <= capSteps; i++) {
            const angle = (Math.PI / 2) * (i / capSteps);
            points.push(new THREE.Vector2(
                Math.sin(angle) * rBot,
                -height / 2 - Math.cos(angle) * rBot + rBot
            ));
        }
        // Tapered body
        points.push(new THREE.Vector2(rBot, -height / 2 + rBot));
        points.push(new THREE.Vector2(rTop, height / 2 - rTop));
        // Top hemisphere cap
        for (let i = 0; i <= capSteps; i++) {
            const angle = (Math.PI / 2) * (i / capSteps);
            points.push(new THREE.Vector2(
                Math.cos(angle) * rTop,
                height / 2 + Math.sin(angle) * rTop - rTop
            ));
        }

        const geo = new THREE.LatheGeometry(points, segments);
        const mesh = new THREE.Mesh(geo, mat.clone());
        mesh.userData.type = 'capsule';
        mesh.userData.fromBone = fromBone;
        mesh.userData.toBone = toBone;
        mesh.castShadow = true;
        scene.add(mesh);
        arr.push(mesh);
    }

    // Hands — slightly flattened spheres
    for (const handName of ['LeftHand', 'RightHand']) {
        const bone = findBone(rootBone, handName);
        if (bone) {
            const geo = new THREE.SphereGeometry(4, 12, 10);
            const mesh = new THREE.Mesh(geo, mat.clone());
            mesh.userData.type = 'joint';
            mesh.userData.bone = bone;
            scene.add(mesh);
            arr.push(mesh);
        }
    }
    return arr;
}


const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _mid = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
const _quat = new THREE.Quaternion();
const _dir = new THREE.Vector3();

function updateBodyMeshes() { updateBodyMeshesIn(bodyMeshes); }

function updateBodyMeshesIn(meshArray) {
    for (const mesh of meshArray) {
        if (mesh.userData.type === 'head') {
            // Position between Head and HeadEnd, scale as ellipsoid
            const base = mesh.userData.baseBone;
            const top = mesh.userData.bone;
            base.getWorldPosition(_v1);
            top.getWorldPosition(_v2);
            _mid.lerpVectors(_v1, _v2, 0.5);
            mesh.position.copy(_mid);
            const h = _v1.distanceTo(_v2);
            mesh.scale.set(8, h * 0.6, 8.5);
            _dir.subVectors(_v2, _v1).normalize();
            if (_dir.lengthSq() > 0.0001) {
                _quat.setFromUnitVectors(_up, _dir);
                mesh.quaternion.copy(_quat);
            }
            continue;
        }

        if (mesh.userData.type === 'joint') {
            mesh.userData.bone.getWorldPosition(_v1);
            mesh.position.copy(_v1);
            continue;
        }

        // Capsule — position at midpoint, orient along bone axis
        const from = mesh.userData.fromBone;
        const to = mesh.userData.toBone;
        from.getWorldPosition(_v1);
        to.getWorldPosition(_v2);

        _mid.lerpVectors(_v1, _v2, 0.5);
        mesh.position.copy(_mid);

        _dir.subVectors(_v2, _v1).normalize();
        if (_dir.lengthSq() > 0.0001) {
            _quat.setFromUnitVectors(_up, _dir);
            mesh.quaternion.copy(_quat);
        }
    }
}

// Extract the character's root path from a BVH clip (sample every N frames)
// Silently rewrite the user's prompt to produce better, more dynamic motion.
// Always biases toward locomotion (walking/moving forward) and exaggerated body movement.
function enhanceMotionPrompt(userPrompt) {
    let p = userPrompt;

    // Replace common verbs with more dynamic versions
    // Convert terrain-related prompts to actions Kimodo handles better
    p = p.replace(/\bclimb(?:s|ing)?\s+(?:a\s+)?(?:hill|mountain|slope|incline|ridge)\b/gi, 'climbs up a long staircase steadily, one step at a time');
    p = p.replace(/\b(?:go|goes|going|walk|walks|walking)\s+up(?:hill| a hill| the hill| a slope)\b/gi, 'walks up stairs steadily');
    p = p.replace(/\b(?:go|goes|going|walk|walks|walking)\s+down(?:hill| a hill| the hill| a slope)\b/gi, 'walks down stairs carefully');
    p = p.replace(/\bhike(?:s|ing)?\b/gi, 'walks up and down stairs while moving forward');

    const replacements = {
        'walks': 'walks forward confidently with long strides and arm swings',
        'walk': 'walk forward with long exaggerated strides, swinging arms',
        'runs': 'runs forward fast with high knees, pumping arms, covering ground',
        'run': 'run forward fast with high knees, pumping arms, covering distance',
        'jogs': 'jogs forward energetically with bouncy steps, moving across the space',
        'jog': 'jog forward energetically with bouncy steps covering ground',
        'dances': 'dances with large expressive full-body movements, stepping side to side',
        'dance': 'dance with big expressive full-body movements, stepping around the space',
        'stands': 'shifts weight and moves around slowly while standing',
        'stand': 'shift weight and sway while moving slightly forward',
        'sits': 'sits down then gets up and moves around',
        'sit': 'sit down briefly then stand and walk forward',
        'does karate': 'performs karate kicks and punches while stepping forward aggressively',
        'does kung fu': 'performs kung fu strikes and kicks moving across the floor',
        'fights': 'fights with punches and kicks, advancing forward aggressively',
        'trips': 'trips and stumbles forward dramatically, arms flailing',
        'falls': 'falls forward dramatically with arms reaching out',
        'sneaks': 'sneaks forward in a low crouch, moving carefully across the space',
        'sneak': 'sneak forward in a low crouch, tiptoeing across the room',
        'explores': 'walks around exploring, looking in different directions while moving',
        'relaxes': 'stretches and moves around lazily, shifting positions',
        'celebrates': 'jumps and pumps fists while moving around excitedly',
        'shops': 'walks forward browsing, stopping briefly then moving on',
    };

    // Apply replacements (case-insensitive, whole word)
    for (const [from, to] of Object.entries(replacements)) {
        const regex = new RegExp('\\b' + from + '\\b', 'gi');
        p = p.replace(regex, to);
    }

    // If no movement verb was found, append locomotion bias
    const hasMovement = /walk|run|jog|step|move|kick|punch|jump|dance|sneak|strid|crawl/i.test(p);
    if (!hasMovement) {
        p += '. The person should walk forward while doing this, covering distance across the space';
    }

    // Always append quality suffix
    p += '. Make all movements large, exaggerated, and continuous. The person should travel forward through space, not stay in place.';

    return p;
}

function extractPathFromBVH(text) {
    const loader = new BVHLoader();
    const result = loader.parse(text);
    const clip = result.clip;
    const root = result.skeleton.bones[0];

    // Find the Hips position track (Root stays at 0,0,0 — Hips has the actual movement)
    const posTrack = clip.tracks.find(t => t.name.includes('Hips') && t.name.endsWith('.position'))
        || clip.tracks.find(t => t.name.endsWith('.position'));
    if (!posTrack) return { path: [[0, 0]], result };

    const sample = () => {
        const values = posTrack.values;
        const totalFrames = values.length / 3;
        const out = [];
        const numSamples = Math.min(30, totalFrames); // More samples for smoother path
        const step = Math.max(1, Math.floor(totalFrames / numSamples));
        for (let i = 0; i < totalFrames; i += step) {
            out.push([Math.round(values[i * 3]), Math.round(values[i * 3 + 1]), Math.round(values[i * 3 + 2])]);
        }
        if (totalFrames > 0) {
            const l = (totalFrames - 1) * 3;
            out.push([Math.round(values[l]), Math.round(values[l + 1]), Math.round(values[l + 2])]);
        }
        return out;
    };
    // rawPath is the motion exactly as generated — the generator's own
    // "did it actually travel?" retry check reads that. `path` is where the
    // character really goes on stage: createCharacter runs every clip
    // through normalizeRootTravel (travel clamped, walk centred on the
    // spawn), so the layout, the path line and the terrain must see the same.
    // Using the raw one had the layout clearing a middle over twice as wide
    // as the walk and pushing every tree out past it.
    const rawPath = sample();
    if (posTrack.name.includes('Hips')) normalizeRootTravel(clip);
    const path = sample();
    return { path, rawPath, result };
}


// Load the scene's primary character from BVH, replacing any existing cast.
// Additional people are added later via createCharacter() (see the director
// in handleChat) — this is just "start over with one person".
function loadBVH(text, prompt = '', avatar = null) {
    clearAllCharacters();

    const entry = createCharacter(text, { label: 'Person 1', prompt, avatar });
    syncPrimaryGlobals();

    isPlaying = true;
    playbackFinished = false;
    clock.getDelta(); // drain clock so first frame gets a small delta
    recomputePlaybackDuration();
    updatePlayPauseIcon();

    document.getElementById('gen-info').textContent =
        `${entry.mergedClip.duration.toFixed(1)}s @ 30fps`;
}

// Recalculate how long the current playthrough is (character clip, adjusted
// for mixer.timeScale, vs. any object-path animations) and (re)arm the
// shared clock/finished-event off that. Call after loadBVH(), after changing
// mixer.timeScale, or after populating activeObjectAnimations.
function recomputePlaybackDuration() {
    const charDurations = characters.map(c => c.mergedClip.duration / (c.mixer.timeScale || 1));
    const objDurations = activeObjectAnimations.map(a => a.duration);
    const duration = Math.max(...charDurations, ...objDurations, 0.1);
    createPlaybackClock(duration);
}

// ========================================================================
// CHARACTERS — every person in the scene, each with its own skeleton,
// mixer and motion timeline, so a scene can be built up prompt by prompt
// ("a person walks in" → "add another person" → "they fight").
// characters[0] is mirrored into the legacy globals (characterGroup,
// currentBones, mixer, currentAction, currentClip, bodyMeshes,
// timelineClips) so camera-follow, terrain, path viz and export keep
// working against the primary character unchanged.
//
// Each character's motion comes from its own Kimodo call — Kimodo has no
// notion of two people interacting, so they won't make literal contact;
// they're positioned facing each other so it reads as a shared scene.
// ========================================================================
// Every Create character's mixer runs at this rate: playFrom, groundCharacter,
// prependHoldToCharacter and the timeline all convert between clip time and
// the shared wall clock with it, so a mixer left at any other rate plays out
// of step with the rest of the cast.
const CHAR_TIME_SCALE = 1.5;
let characters = []; // [{ label, group, bones, bodyMeshesArr, mixer, action, clips, mergedClip, spawn }]

// Sample the feet across a character's clip and drop it so the lowest foot
// rests on the ground plane. Keeps x/z, only adjusts y.
function groundCharacter(entry) {
    const frameLows = []; // lowest foot height of each sampled frame
    const bonePos = new THREE.Vector3();
    // Enough samples to actually catch the lowest frame of a long timeline, and
    // setTime rather than update(dt) so the sweep is exact: update() multiplies
    // by mixer.timeScale, so 20 steps of duration/20 really swept 1.5x the clip
    // and wrapped around.
    const samples = Math.max(24, Math.ceil(entry.mergedClip.duration * 8));
    const footNames = ['LeftFoot', 'RightFoot', 'LeftToeBase', 'RightToeBase'];
    for (let s = 0; s <= samples; s++) {
        entry.mixer.setTime((s / samples) * entry.mergedClip.duration / CHAR_TIME_SCALE);
        entry.group.updateMatrixWorld(true);
        let frameMin = Infinity;
        for (const name of footNames) {
            const bone = findBone(entry.bones, name);
            if (bone) {
                bone.getWorldPosition(bonePos);
                if (bonePos.y < frameMin) frameMin = bonePos.y;
            }
        }
        if (frameMin < Infinity) frameLows.push(frameMin);
    }
    if (!frameLows.length) return;
    // Ground on the feet's usual lowest height, not the single lowest frame:
    // a generated fall or kick often dips a foot below the floor for a moment,
    // and grounding on that lifted the whole character, so they hovered
    // through every normal standing frame. The 10th percentile ignores those
    // dips while still catching a clip that's mostly jumping.
    frameLows.sort((a, b) => a - b);
    const minY = frameLows[Math.floor(frameLows.length * 0.1)];
    const footCapsuleRadius = 2;
    // ADJUST, don't assign. minY is a world height, so it already includes
    // whatever offset the group is carrying. Assigning -(minY - r) threw that
    // offset away every time a character was re-grounded: the first call worked
    // because the group started at y=0, but appending a beat re-grounded from
    // an already-lowered pose and snapped the whole body back up to y=0 — a
    // character left floating ~100 units over the floor, above anyone who
    // hadn't just gained a beat.
    entry.group.position.y -= (minY - footCapsuleRadius);
}

// Arm a character's action for one-shot playback. Order matters: the loop
// mode is only switched to LoopOnce AFTER any sampling passes, otherwise
// the sampling run finishes the action and clampWhenFinished leaves it
// paused — which is exactly why an earlier secondary character stood still.
function armCharacterAction(entry) {
    entry.action.setLoop(THREE.LoopOnce, 1);
    entry.action.clampWhenFinished = true;
    entry.action.reset();
    entry.action.play();
    entry.mixer.setTime(0);
}

function createCharacter(bvhText, { position = [0, 0, 0], yaw = 0, label = 'Person', prompt = '', avatar = null } = {}) {
    const loader = new BVHLoader();
    const result = loader.parse(bvhText);
    normalizeRootTravel(result.clip);

    const group = new THREE.Group();
    const bones = result.skeleton.bones[0];
    group.add(bones);
    scene.add(group);

    const mixer = new THREE.AnimationMixer(bones);
    mixer.timeScale = CHAR_TIME_SCALE;
    const action = mixer.clipAction(result.clip);
    action.play(); // default looping while we sample for grounding

    // `avatar` names which model this person wears (see AVATARS); it has to be
    // loaded already, since this function is synchronous — callers go through
    // ensureAvatar first. Anyone without one falls back to the capsule
    // mannequin, which is also what happens if a VRM failed to load.
    let vrmMeshes = null;
    let bodyMeshesArr = [];
    const avatarModel = avatar && avatarCache[avatar];
    if (avatarModel) {
        // Pose the skeleton at the clip's first frame before aligning the VRM
        // to it. Fresh from BVHLoader, every bone sits at identity rotation
        // with its offset along +X — spine, arms and legs all pointing
        // sideways — which is no pose at all to align limb directions against;
        // frame 0 is an actual standing figure. armCharacterAction rewinds to
        // time 0 afterwards anyway, so nothing else sees this.
        mixer.setTime(0);
        group.updateMatrixWorld(true);
        const spec = AVATARS[avatar];
        const hide = spec.hide || [];
        vrmMeshes = buildVrmSkinnedMeshes(avatarModel.vrm, bones, avatarModel.scale, o => !matchesMaterial(o, hide));
        // A borrowed garment is skinned from ITS OWN model, so the alignment is
        // computed against the rest pose it was authored for; both end up on the
        // same BVH skeleton, so they move together.
        const donor = spec.borrow && avatarCache[spec.borrow.from];
        if (donor) vrmMeshes.push(...buildVrmSkinnedMeshes(donor.vrm, bones, donor.scale, o => matchesMaterial(o, [spec.borrow.match])));
    } else {
        bodyMeshesArr = buildBodyMeshesFor(bones);
    }

    const entry = {
        label,
        group, bones,
        bodyMeshesArr, vrmMeshes, avatar,
        mixer, action,
        clips: [{ prompt, duration: result.clip.duration, clip: result.clip }],
        mergedClip: result.clip,
        spawn: { position, yaw },
    };

    group.position.set(position[0], 0, position[2]);
    group.rotation.y = yaw;
    groundCharacter(entry);
    armCharacterAction(entry);

    characters.push(entry);
    return entry;
}

// Append another motion onto a character's own timeline, blended onto
// whatever it was already doing.
// A trivial clip that holds `sourceClip`'s very first pose for `duration`
// (clip-time) seconds — two keyframes, both identical. Used to make a
// character wait rather than start moving the instant the shared clock hits
// zero (see prependHoldToCharacter).
function holdPoseClip(sourceClip, duration) {
    const d = Math.max(duration, 0.05);
    const tracks = sourceClip.tracks.map(t => {
        const size = t.getValueSize();
        const v0 = Array.from(t.values.slice(0, size));
        return new t.constructor(t.name, new Float32Array([0, d]), new Float32Array([...v0, ...v0]));
    });
    return new THREE.AnimationClip('hold', d, tracks);
}

// Delays a fresh character's clip by `delayWallSeconds` — real seconds, same
// units activeObjectAnimations durations use — before it starts moving.
// "A person gets out of the car" was starting to walk at t=0 of the shared
// clock, the same moment the car itself starts its own arrival, because
// each character's clip has always run independently from t=0 with no
// notion of anything else on stage. Prepending a held pose (via the same
// mergeClips crossfade every other beat boundary uses) makes them stand
// still until the car's own beat has actually finished.
function prependHoldToCharacter(entry, delayWallSeconds) {
    if (delayWallSeconds < 0.15) return;
    const delayClipSeconds = delayWallSeconds * CHAR_TIME_SCALE; // mergeClips/entry.clips work in clip-time, not wall-clock
    const hold = holdPoseClip(entry.mergedClip, delayClipSeconds);
    const merged = mergeClips(hold, entry.mergedClip, 0.3);
    entry.mixer.stopAllAction();
    entry.mixer.uncacheRoot(entry.bones);
    entry.mixer = new THREE.AnimationMixer(entry.bones);
    entry.mixer.timeScale = CHAR_TIME_SCALE;
    entry.mergedClip = merged;
    entry.action = entry.mixer.clipAction(merged);
    entry.action.play();
    entry.clips.unshift({ prompt: '(waiting)', duration: hold.duration, clip: hold });
    groundCharacter(entry);
    armCharacterAction(entry);
    syncPrimaryGlobals();
}

function appendMotionToCharacter(entry, newClip, prompt, maxTravel, pinInPlace = false) {
    // A shared beat is held in place regardless of how much travel the raw
    // motion carries: fillBeat may have stacked four step-ins into ~90 units,
    // and the usual 2.5x compression floor would still leave ~36 of it —
    // enough for two people to close a 58-unit gap to 23 and end up inside
    // each other. Skating feet are the lesser evil here.
    normalizeRootTravel(newClip, maxTravel, pinInPlace ? Infinity : MAX_COMPRESSION);
    const merged = mergeClips(entry.mergedClip, newClip, 0.4);
    entry.mixer.stopAllAction();
    entry.mixer.uncacheRoot(entry.bones);
    entry.mixer = new THREE.AnimationMixer(entry.bones);
    entry.mixer.timeScale = CHAR_TIME_SCALE;
    entry.mergedClip = merged;
    entry.action = entry.mixer.clipAction(merged);
    entry.action.play();
    entry.clips.push({ prompt, duration: newClip.duration, clip: newClip });
    groundCharacter(entry);
    armCharacterAction(entry);
    syncPrimaryGlobals();
}

// Rebuild a character's merged clip from its remaining timeline segments
// (used after deleting a segment).
function rebuildCharacterClip(entry) {
    if (entry.clips.length === 0) return;
    let merged = entry.clips[0].clip;
    for (let i = 1; i < entry.clips.length; i++) {
        merged = mergeClips(merged, entry.clips[i].clip, 0.4);
    }
    entry.mixer.stopAllAction();
    entry.mixer.uncacheRoot(entry.bones);
    entry.mixer = new THREE.AnimationMixer(entry.bones);
    entry.mixer.timeScale = CHAR_TIME_SCALE;
    entry.mergedClip = merged;
    entry.action = entry.mixer.clipAction(merged);
    entry.action.play();
    groundCharacter(entry);
    armCharacterAction(entry);
    syncPrimaryGlobals();
}

// Mirror characters[0] into the legacy single-character globals.
function syncPrimaryGlobals() {
    const p = characters[0];
    if (!p) {
        characterGroup = null; currentBones = null; bodyMeshes = [];
        mixer = null; currentAction = null; currentClip = null; timelineClips = [];
        return;
    }
    characterGroup = p.group; currentBones = p.bones; bodyMeshes = p.bodyMeshesArr;
    mixer = p.mixer; currentAction = p.action; currentClip = p.mergedClip;
    timelineClips = p.clips; // same array reference — edits flow both ways
}

function clearAllCharacters() {
    for (const c of characters) {
        scene.remove(c.group);
        c.bodyMeshesArr.forEach(m => scene.remove(m));
    }
    characters = [];
    syncPrimaryGlobals();
}

// Where a character finishes up, and which way they were travelling — both
// in world space. Root motion lives in the clip's Hips track, so the group's
// own rotation/position has to be applied on top of it.
function characterEndState(entry) {
    const pos = entry.group.position.clone();
    const yaw = entry.group.rotation.y;
    const track = entry.mergedClip.tracks.find(t => t.name.includes('Hips') && t.name.endsWith('.position'));
    if (!track || track.values.length < 6) return { point: pos, heading: yaw };

    const v = track.values;
    const sx = v[0], sz = v[2];
    const ex = v[v.length - 3], ez = v[v.length - 1];
    const cos = Math.cos(yaw), sin = Math.sin(yaw);
    // Rotate the local end offset into world space and offset by the group.
    const point = pos.clone();
    point.x += ex * cos + ez * sin;
    point.z += -ex * sin + ez * cos;

    const dx = ex - sx, dz = ez - sz;
    const heading = (Math.abs(dx) < 1 && Math.abs(dz) < 1)
        ? yaw                              // barely travelled — keep facing
        : yaw + Math.atan2(dx, dz);
    return { point, heading };
}

const WORLD_UP = new THREE.Vector3(0, 1, 0);

// ---- Making a beat last as long as it is supposed to ---------------------
//
// Kimodo returns the full duration you ask for, but usually performs the
// action in the first second or so and then stands perfectly still for the
// rest. Measured off the live backend at duration 5:
//
//   "A person throws a punch and steps forward"  moving 0.80s - 1.93s  (23%)
//   "A person blocks and staggers backward"      moving 0.70s - 1.20s  (10%)
//
// So "they fight" was one punch followed by four seconds of standing around.
// Find the part that actually moves, cut the padding, then repeat and stretch
// it so the beat is full of action for its whole length.
const BEAT_ENERGY_FLOOR = 0.12;  // fraction of peak movement that counts as moving
const BEAT_SMOOTH_WINDOW = 0.25; // seconds; smooths out single-frame twitches
const BEAT_MARGIN = 0.15;        // seconds of wind-up/follow-through to keep
const MAX_BEAT_STRETCH = 1.8;    // never slow a motion down more than this
const BEAT_SYNC_TOLERANCE = 0.05; // clip seconds people sharing a beat may be out of step before one waits

// Where in the clip the body is actually moving. Rotation only — root travel
// is handled separately, and a slide with a still body isn't an action.
function actionSpan(clip) {
    const rot = clip.tracks.filter(t => t.name.endsWith('.quaternion'));
    if (rot.length === 0) return null;
    const frames = rot[0].times.length;
    if (frames < 4) return null;

    const energy = new Float64Array(frames);
    for (const t of rot) {
        const v = t.values, size = t.getValueSize();
        if (t.times.length !== frames) continue;
        for (let i = 1; i < frames; i++) {
            let d = 0;
            for (let k = 0; k < size; k++) d += Math.abs(v[i * size + k] - v[(i - 1) * size + k]);
            energy[i] += d;
        }
    }
    // Smooth before thresholding. Raw per-frame energy is spiky, and a single
    // stray twitch four seconds in is enough to make the span look like it
    // covers the whole clip — which is how a beat that is genuinely dead for
    // its last 70% survived the first version of this trim.
    const times = rot[0].times;
    const fps = frames / Math.max(clip.duration, 1e-3);
    const half = Math.max(1, Math.round(fps * BEAT_SMOOTH_WINDOW / 2));
    const smooth = new Float64Array(frames);
    for (let i = 0; i < frames; i++) {
        let sum = 0, n = 0;
        for (let k = Math.max(0, i - half); k <= Math.min(frames - 1, i + half); k++) { sum += energy[k]; n++; }
        smooth[i] = sum / n;
    }

    let peak = 0;
    for (const e of smooth) if (e > peak) peak = e;
    if (peak <= 0) return null;

    const floor = peak * BEAT_ENERGY_FLOOR;
    let first = -1, last = -1;
    for (let i = 0; i < frames; i++) {
        if (smooth[i] > floor) { if (first < 0) first = i; last = i; }
    }
    if (first < 0 || last <= first) return null;

    const start = Math.max(0, times[first] - BEAT_MARGIN);
    const end = Math.min(clip.duration, times[last] + BEAT_MARGIN);
    return (end - start) < 0.2 ? null : { start, end };
}

// Keep only [start, end], rebased to zero.
function sliceClip(clip, start, end) {
    const tracks = [];
    for (const t of clip.tracks) {
        const size = t.getValueSize();
        const times = [], values = [];
        for (let i = 0; i < t.times.length; i++) {
            if (t.times[i] < start || t.times[i] > end) continue;
            times.push(t.times[i] - start);
            for (let k = 0; k < size; k++) values.push(t.values[i * size + k]);
        }
        if (times.length < 2) continue;   // nothing left to animate — drop it
        tracks.push(new t.constructor(t.name, new Float32Array(times), new Float32Array(values)));
    }
    return tracks.length ? new THREE.AnimationClip(clip.name, end - start, tracks) : clip;
}

function scaleClipTime(clip, factor) {
    for (const t of clip.tracks) {
        for (let i = 0; i < t.times.length; i++) t.times[i] *= factor;
    }
    clip.duration *= factor;
    return clip;
}

// Trim the padding, stretch the action toward `target` seconds, and if it's
// still short, hold its final pose for the rest. Every beat in the same
// instruction is fitted to the same target, so a two person exchange stays
// in step instead of one of them finishing early. (It used to repeat the
// action to fill the time instead — a kick-and-fall then fell twice.)
function fillBeat(clip, target) {
    const span = actionSpan(clip);
    if (!span) return clip;

    let out = sliceClip(clip, span.start, span.end);
    const factor = THREE.MathUtils.clamp(target / out.duration, 1 / MAX_BEAT_STRETCH, MAX_BEAT_STRETCH);
    out = scaleClipTime(out, factor);
    if (out.duration < target - 0.05) out = mergeClips(out, holdEndPoseClip(out, target - out.duration), 0.15);
    return out;
}

// A clip that holds `sourceClip`'s LAST pose for `duration` seconds — how a
// beat that ends lying on the ground (or mid-bow) waits out the rest of it.
function holdEndPoseClip(sourceClip, duration) {
    const d = Math.max(duration, 0.05);
    const tracks = sourceClip.tracks.map(t => {
        const size = t.getValueSize();
        const vEnd = Array.from(t.values.slice(t.values.length - size));
        return new t.constructor(t.name, new Float32Array([0, d]), new Float32Array([...vEnd, ...vEnd]));
    });
    return new THREE.AnimationClip('hold-end', d, tracks);
}

// Kimodo generates a motion, not a loop — its last frame rarely matches its
// first, so playing it on THREE.LoopRepeat snaps visibly every time it wraps
// (measured on the Chat persona's clips: a ~110-270 unit jump in combined
// joint values between first and last frame). Ease the final `blendSeconds`
// of the clip toward the starting pose (quaternions via slerp, positions via
// lerp) so the wrap lands on a matching pose instead of a pop. Mutates and
// returns the clip, same convention as normalizeRootTravel/fillBeat.
function loopifyClip(clip, blendSeconds = 0.4) {
    const q1 = new THREE.Quaternion(), q2 = new THREE.Quaternion();
    for (const t of clip.tracks) {
        const size = t.getValueSize();
        const v = t.values, times = t.times;
        const n = times.length;
        if (n < 3) continue;
        const startVal = v.slice(0, size);
        const isQuat = t.name.endsWith('.quaternion');
        for (let i = 0; i < n; i++) {
            const fromEnd = clip.duration - times[i];
            if (fromEnd > blendSeconds) continue;
            const w = 1 - fromEnd / blendSeconds;          // 0 at the blend point -> 1 at the last frame
            const eased = w * w * (3 - 2 * w);              // smoothstep
            const base = i * size;
            if (isQuat) {
                q1.set(v[base], v[base + 1], v[base + 2], v[base + 3]);
                q2.set(startVal[0], startVal[1], startVal[2], startVal[3]);
                q1.slerp(q2, eased);
                v[base] = q1.x; v[base + 1] = q1.y; v[base + 2] = q1.z; v[base + 3] = q1.w;
            } else {
                for (let k = 0; k < size; k++) {
                    v[base + k] = v[base + k] * (1 - eased) + startVal[k] * eased;
                }
            }
        }
    }
    return clip;
}

// Rotate a whole beat about its own start point: the Hips travel AND the body
// orientation. rotateBVHToward() only ever turned the travel, which is why a
// character could slide sideways while still facing the way the raw motion was
// baked. All rotations here are about Y, so they commute with the group's own
// yaw — adding delta to the clip adds delta to the world facing.
function yawClip(clip, deltaYaw) {
    if (!deltaYaw) return clip;
    const q = new THREE.Quaternion().setFromAxisAngle(WORLD_UP, deltaYaw);
    const p = new THREE.Vector3();
    const r = new THREE.Quaternion();
    for (const t of clip.tracks) {
        if (!t.name.includes('Hips')) continue;
        const v = t.values;
        if (t.name.endsWith('.position')) {
            const sx = v[0], sz = v[2];
            for (let i = 0; i < v.length; i += 3) {
                p.set(v[i] - sx, 0, v[i + 2] - sz).applyAxisAngle(WORLD_UP, deltaYaw);
                v[i] = sx + p.x;
                v[i + 2] = sz + p.z;
            }
        } else if (t.name.endsWith('.quaternion')) {
            for (let i = 0; i < v.length; i += 4) {
                r.set(v[i], v[i + 1], v[i + 2], v[i + 3]).premultiply(q);
                v[i] = r.x; v[i + 1] = r.y; v[i + 2] = r.z; v[i + 3] = r.w;
            }
        }
    }
    return clip;
}

// Which way a character's chest actually points when their timeline ends.
// Measured from the shoulders rather than read off the root rotation, because
// the skeleton's rest orientation differs between BVH sources and guessing it
// is how you end up with two people fighting back to back.
function characterEndFacing(entry) {
    const l = findBone(entry.bones, 'LeftShoulder');
    const r = findBone(entry.bones, 'RightShoulder');
    if (!l || !r) return entry.group.rotation.y;

    const was = entry.mixer.time;
    entry.mixer.setTime(entry.mergedClip.duration / CHAR_TIME_SCALE);
    entry.group.updateMatrixWorld(true);
    const L = new THREE.Vector3(), R = new THREE.Vector3();
    l.getWorldPosition(L); r.getWorldPosition(R);
    entry.mixer.setTime(was);

    const fwd = L.sub(R).setY(0).cross(WORLD_UP).normalize();
    return Math.atan2(fwd.x, fwd.z);
}

const shortestAngle = a => Math.atan2(Math.sin(a), Math.cos(a));

// Where a CANDIDATE clip (not yet merged onto the character) makes the
// skeleton face at a given point in its OWN local time — measured the exact
// same way as characterEndFacing (shoulder bones, not the root quaternion),
// so the two numbers are directly comparable. Temporarily plays the clip on
// the character's own mixer to get a real sampled pose rather than trying to
// read facing out of the raw quaternion track (rest-pose offsets vary by
// BVH source, which is the same reason characterEndFacing doesn't do that).
// Fully synchronous — nothing repaints between the temporary play and the
// cleanup, so this never flashes on screen.
function clipFacingAt(entry, clip, localTime) {
    const action = entry.mixer.clipAction(clip);
    action.reset(); action.setEffectiveWeight(1); action.play();
    const was = entry.mixer.time;
    entry.mixer.setTime(localTime);
    entry.group.updateMatrixWorld(true);

    const l = findBone(entry.bones, 'LeftShoulder');
    const r = findBone(entry.bones, 'RightShoulder');
    let angle = entry.group.rotation.y;
    if (l && r) {
        const L = new THREE.Vector3(), R = new THREE.Vector3();
        l.getWorldPosition(L); r.getWorldPosition(R);
        const fwd = L.sub(R).setY(0).cross(WORLD_UP).normalize();
        angle = Math.atan2(fwd.x, fwd.z);
    }

    action.stop();
    entry.mixer.uncacheAction(clip, entry.bones);
    entry.mixer.setTime(was);
    return angle;
}

// Rotate `clip` (both its travel AND its body orientation — see yawClip) so
// it STARTS facing `desiredWorldFacing`, instead of whatever direction
// Kimodo happened to bake this fresh generation in. Used for both plain
// continuity (desiredWorldFacing = wherever the character already is) and
// aiming a shared beat at another character (desiredWorldFacing = toward
// them) — same operation either way, just a different target angle.
function yawClipToFace(entry, clip, desiredWorldFacing) {
    const startFacing = clipFacingAt(entry, clip, 0);
    return yawClip(clip, shortestAngle(desiredWorldFacing - startFacing));
}

// Kimodo bakes a lot of ground travel into its motions — a "walk" can carry
// someone 800+ units, straight out of frame. Compress the horizontal root
// motion so a single beat stays near the middle of the scene. Height is left
// alone so jumps still read, and compression is capped so feet don't skate
// too badly. Deterministic: the same BVH always yields the same result, which
// pickSpawnForNewCharacter relies on when it previews a clip.
const MAX_BEAT_TRAVEL = 220;    // scene units a single motion may cover
const MAX_COMPRESSION = 2.5;    // never squash more than this
const INTERACTION_TRAVEL = 12;  // a shared beat is fought/shaken in place — a lean or a step-in, not a walk
const MIN_INTERACTION_GAP = 38; // stay at least this far apart (hips), even after closing in — at 10-34 a handshake pressed bodies together instead of meeting hands
const MIN_INTERACTION_TRAVEL = 4; // the least a shared beat may move someone, even at arm's length (was 10, which on its own closed the gap)

// How far can THIS beat let someone move, given they're currently `gap`
// units from whoever they're interacting with? A fixed cap on each side
// isn't enough on its own — two people each allowed the full 55 units can
// still close 110 units of combined distance, more than the ~70 units a
// "walk in and meet" beat leaves between them, so they'd walk through each
// other. Bounding each side to half the CURRENT gap (minus a safety margin)
// guarantees the beat can never close it entirely, however many interaction
// beats happen back to back — it's recomputed fresh from the live gap every
// time, not a running total.
function interactionTravelCap(gap) {
    return Math.max(MIN_INTERACTION_TRAVEL, Math.min(INTERACTION_TRAVEL, gap / 2 - MIN_INTERACTION_GAP));
}

function normalizeRootTravel(clip, maxTravel = MAX_BEAT_TRAVEL, maxCompression = MAX_COMPRESSION) {
    const track = clip.tracks.find(t => t.name.includes('Hips') && t.name.endsWith('.position'));
    if (!track || track.values.length < 6) return clip;
    const v = track.values;
    const sx = v[0], sz = v[2];

    let maxDist = 0;
    for (let i = 0; i < v.length; i += 3) {
        const d = Math.hypot(v[i] - sx, v[i + 2] - sz);
        if (d > maxDist) maxDist = d;
    }
    if (maxDist > maxTravel) {
        const scale = Math.max(maxTravel / maxDist, 1 / maxCompression);
        for (let i = 0; i < v.length; i += 3) {
            v[i] = sx + (v[i] - sx) * scale;
            v[i + 2] = sz + (v[i + 2] - sz) * scale;
        }
    }

    // Then centre the path on the spawn point. Clamping alone isn't enough:
    // a walk that starts at the origin and travels 220 units forward ends 220
    // units from the middle. Centring makes it pass *through* the middle
    // instead, so the interesting part of the beat is where the camera is
    // pointed. Beats after the first get re-based onto the previous beat's
    // end by mergeClips, so this only actually moves the opening beat.
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < v.length; i += 3) {
        if (v[i] < minX) minX = v[i];
        if (v[i] > maxX) maxX = v[i];
        if (v[i + 2] < minZ) minZ = v[i + 2];
        if (v[i + 2] > maxZ) maxZ = v[i + 2];
    }
    const cx = (minX + maxX) / 2, cz = (minZ + maxZ) / 2;
    for (let i = 0; i < v.length; i += 3) {
        v[i] -= cx;
        v[i + 2] -= cz;
    }
    return clip;
}

// Fit the Create camera around everything that is going to happen — the full
// XZ extent of every character's root path, not just where they stand at this
// instant. A character can only walk "out of frame" if the camera is ignoring
// where they are headed, so the frame is recomputed after every beat.
function castPathBoundsXZ() {
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity, any = false;
    const p = new THREE.Vector3();
    for (const c of characters) {
        const track = c.mergedClip && c.mergedClip.tracks.find(
            t => t.name.includes('Hips') && t.name.endsWith('.position'));
        if (!track) continue;
        const v = track.values;
        for (let i = 0; i < v.length; i += 3) {
            p.set(v[i], 0, v[i + 2]);
            p.applyAxisAngle(WORLD_UP, c.group.rotation.y).add(c.group.position);
            if (p.x < minX) minX = p.x;
            if (p.x > maxX) maxX = p.x;
            if (p.z < minZ) minZ = p.z;
            if (p.z > maxZ) maxZ = p.z;
            any = true;
        }
    }
    return any ? { minX, maxX, minZ, maxZ } : null;
}

const CHAR_HEIGHT = 180;   // roughly how tall the SOMA character stands
const FRAME_MARGIN = 90;   // breathing room around the action

function computeCreateFraming() {
    let b = castPathBoundsXZ(); // null when there's no character at all — see grow() below

    // Reset View used to frame the cast alone, so adding buildings or trees
    // (which don't move a character) left the camera exactly where it was —
    // "Reset" after "make a city" walked the camera straight into a
    // building it didn't know was there. And a scene whose main subject is
    // an object, not a person ("make a car move into the scene"), has NO
    // character path to seed the box from in the first place — b started
    // null and stayed null, so the camera was never placed for that kind of
    // scene at all. grow() both widens an existing box and, if there wasn't
    // one, starts one.
    const grow = (x, z, r) => {
        if (!b) { b = { minX: x - r, maxX: x + r, minZ: z - r, maxZ: z + r }; return; }
        if (x - r < b.minX) b.minX = x - r;
        if (x + r > b.maxX) b.maxX = x + r;
        if (z - r < b.minZ) b.minZ = z - r;
        if (z + r > b.maxZ) b.maxZ = z + r;
    };
    // ...but only for props near the action. A "city" or "forest" ranks its
    // buildings and trees in a wide ring around the cast (deliberately — they
    // are backdrop, see the layout engine), and letting a ring 800 units out
    // set the frame pulled the camera so far back the person was a speck
    // behind the near side of it. Distant scenery fills the background on its
    // own; it shouldn't decide the shot. With no cast at all there's nothing
    // else to frame, so everything counts.
    const cast = b && { ...b };
    const SCENERY_FRAME_REACH = 260;
    const framesTheAction = (x, z, r) => {
        if (!cast) return true;
        const dx = Math.max(cast.minX - x, 0, x - cast.maxX);
        const dz = Math.max(cast.minZ - z, 0, z - cast.maxZ);
        return Math.hypot(dx, dz) - r <= SCENERY_FRAME_REACH;
    };
    for (const o of sceneObjects) {
        // An outdoor scene's props line the platform's edge as backdrop; once
        // a forest packed in close, counting them pulled the camera out
        // beyond the treeline, where the trees hid the person entirely.
        // They still count when there's no cast to frame.
        if (o.userData._isGround || (cast && o.userData._scenery)) continue;
        const r = o.userData._footprint || 40;
        if (!framesTheAction(o.position.x, o.position.z, r)) continue;
        grow(o.position.x, o.position.z, r);
    }
    // An animated object's ENTRANCE point (its keyframes, offset from
    // basePos) too, not just wherever it currently rests — so a car's whole
    // arrival is in frame from the first rendered second, not just its
    // parked spot.
    for (const a of activeObjectAnimations) {
        const r = (a.object3D.userData && a.object3D.userData._footprint) || 40;
        for (const kf of a.path) {
            const x = a.basePos.x + kf.pos[0], z = a.basePos.z + kf.pos[2];
            if (framesTheAction(x, z, r)) grow(x, z, r);
        }
    }
    if (!b) return null;

    const cx = (b.minX + b.maxX) / 2, cz = (b.minZ + b.maxZ) / 2;
    const halfW = (b.maxX - b.minX) / 2 + FRAME_MARGIN;
    const halfD = (b.maxZ - b.minZ) / 2;
    const halfV = CHAR_HEIGHT / 2 + FRAME_MARGIN;

    // Distance needed so both the widest and the tallest extent fit the frustum.
    const tanHalfFov = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    const dist = Math.max(halfW / (tanHalfFov * camera.aspect), halfV / tanHalfFov, 320) + halfD;

    // With two or more people, look at them side-on: the default +Z camera
    // sits on the very line they face each other along, so one hides behind
    // the other and their spacing is invisible — a correctly-spaced fight
    // still looked wrong. Put the camera perpendicular to the line between
    // the first two characters' end points instead.
    let dx = 0, dz = 1;
    if (characters.length >= 2) {
        const a = characterEndState(characters[0]).point, b = characterEndState(characters[1]).point;
        const lx = b.x - a.x, lz = b.z - a.z, len = Math.hypot(lx, lz);
        if (len > 1) { dx = -lz / len; dz = lx / len; if (dz < 0) { dx = -dx; dz = -dz; } }
    }
    return { pos: new THREE.Vector3(cx + dx * dist, dist * 0.42, cz + dz * dist), target: new THREE.Vector3(cx, CHAR_HEIGHT * 0.5, cz) };
}

function frameCreateStage() {
    const f = computeCreateFraming();
    if (!f) return;
    camera.position.copy(f.pos);
    controls.target.copy(f.target);
    controls.update();
}

// Start/end/travel of a clip's root motion, in the clip's own local space.
// endX/endZ matter as much as the delta: normalizeRootTravel centres a path on
// the origin, so a clip does not start at (0,0) and the spawn solve has to
// subtract where the clip *ends*, not merely how far it moved.
function clipTravel(clip) {
    const track = clip.tracks.find(t => t.name.includes('Hips') && t.name.endsWith('.position'));
    if (!track || track.values.length < 6) return { dx: 0, dz: 0, endX: 0, endZ: 0 };
    const v = track.values;
    return {
        dx: v[v.length - 3] - v[0],
        dz: v[v.length - 1] - v[2],
        endX: v[v.length - 3],
        endZ: v[v.length - 1],
    };
}

// Where should a newly added character start?
//
// Kimodo's walk motions travel a long way (often 800+ units), so offsetting
// the newcomer's START by a fixed distance just makes two people march off in
// parallel. Instead we solve backwards: pick where they should FINISH (a
// short gap from where the existing character finishes, on the requested
// side), then subtract their own baked-in travel to get the start point, and
// rotate them so they walk in along that approach. Both people then converge
// on roughly the same spot, which is what makes "then they fight" read.
function pickSpawnForNewCharacter(spawnHint, newClip) {
    if (characters.length === 0) return { position: [0, 0, 0], yaw: 0 };

    const { point: meetPoint } = characterEndState(characters[0]);
    const travel = clipTravel(newClip);
    const localHeading = (Math.abs(travel.dx) < 1 && Math.abs(travel.dz) < 1)
        ? 0 : Math.atan2(travel.dx, travel.dz);

    const sides = { opposite: 0, left: -Math.PI / 2, right: Math.PI / 2, behind: Math.PI };
    const approach = sides[spawnHint] ?? 0;   // which side they come in from
    const MEET_GAP = 78;                       // arm's length hips-to-hips: a handshake meets hands, a punch or kick still lands (66 left bodies touching)

    const endX = meetPoint.x + Math.sin(approach) * MEET_GAP;
    const endZ = meetPoint.z + Math.cos(approach) * MEET_GAP;

    // Face them back toward the meeting point, then back out the clip's own end
    // offset so the group origin lands wherever puts their LAST frame on the
    // meeting point.
    const yaw = (approach + Math.PI) - localHeading;
    const cos = Math.cos(yaw), sin = Math.sin(yaw);
    const worldEndX = travel.endX * cos + travel.endZ * sin;
    const worldEndZ = -travel.endX * sin + travel.endZ * cos;

    return { position: [endX - worldEndX, 0, endZ - worldEndZ], yaw };
}

// Where a fresh character starts when the instruction describes them coming
// FROM a specific scene object ("a person gets out of the car") rather than
// meeting the existing cast — there IS no cast yet in that scenario, since
// an object-primary opening scene ("make a car move into the scene") never
// creates a person at all. Starts just outside the object's own footprint
// and faces toward the middle of the scene, so an ordinary forward walk
// carries them away from it rather than needing a second instruction to
// point them anywhere.
function pickSpawnNearObject(objectKeyword) {
    const target = findNearestObject(objectKeyword);
    if (!target) return null;
    // The object's animation may still be mid-arrival by the time this
    // runs — _restPosition is where it's actually parked, not wherever its
    // live, animation-driven .position happens to read right now.
    const rest = target.userData._restPosition || target.position;
    const footprint = target.userData._footprint || 60;

    const toOrigin = new THREE.Vector2(-rest.x, -rest.z);
    const dir = toOrigin.lengthSq() > 1 ? toOrigin.normalize() : new THREE.Vector2(0, 1);
    const clearance = footprint + 40; // clears the object's body before they're walking freely

    return {
        position: [rest.x + dir.x * clearance, 0, rest.z + dir.y * clearance],
        yaw: Math.atan2(dir.x, dir.y),
    };
}

// ========== GEMINI SCENE AGENT ==========

// Gemini only picks WHAT objects — client code handles WHERE to place them
const SCENE_SYSTEM_PROMPT = `You are a creative 3D scene designer. Given a user prompt, pick objects that belong in the scene. Output ONLY valid JSON (no markdown, no backticks).

You do NOT need to specify positions — the engine handles placement automatically. Just pick the right objects.

MOST IMPORTANT RULE: only include props the user actually asked for. An empty models list is the correct, expected answer for a prompt that is purely about a person doing something. Do not decorate.

JSON structure:
{"scene":{"type":"outdoor"|"indoor","models":[{"keyword":"search term","category":number,"size":"large"|"medium"|"small","count":number}],"ground":{"color":"#hex"},"lights":[{"type":"ambient"|"directional"|"point","intensity":0-3,"color":"#hex","position":[x,y,z]}],"animations":[{"keyword":"matches a models[] keyword","path":[{"t":0,"pos":[0,0,0],"yaw":0}]}]},"motion_prompt":"A person ..."}

POLY PIZZA CATEGORIES: 0=Food, 1=Clutter, 3=Transport, 4=Furniture, 5=Objects, 6=Nature, 7=Animals, 8=Buildings, 11=Other

CRITICAL — THEMATIC RELEVANCE:
Every model MUST belong in the scene. "Would this object exist in this real-world location?"
- Living room: sofa, table, lamp, bookshelf, TV, plant. NOT: car, building, tree, hydrant
- City street: building, apartment, car, lamp, bench. NOT: sofa, bed, campfire
- Forest: tree, rock, log, mushroom, campfire. NOT: skyscraper, car, desk
- Beach: palm, umbrella, boat, rock. NOT: building, bookshelf
Think carefully. Every object must make sense for the specific scene.

BANNED KEYWORDS (NEVER use): "fence", "gate", "wall", "barrier", "shelter", "bus stop", "bus shelter", "canopy", "awning", "stop sign", "road barrier", "barricade"

SIZE GUIDE:
- "large": buildings, houses, large trees, skyscrapers (background structures)
- "medium": cars, street lamps, small trees, sofas, bookshelves (mid-sized objects)
- "small": bench, chair, hydrant, trash can, flower, cone, barrel, crate (small props)

RULES:
- models: ONLY objects the user explicitly named or that are unavoidably implied by a named place ("in a kitchen" implies a counter; "a person walks" implies nothing). If they named nothing, return "models": [].
- Never pad the scene. No background filler, no scenery, no "atmosphere" objects. The user adds more later, one instruction at a time.
- 4 models is already a lot. Most prompts need 0-2.
- Each keyword must be UNIQUE — no repeats
- "count" (optional, default 1) is how many copies of that prop to scatter around. Use it ONLY when the place itself is made of that prop — a forest IS many trees, a city street IS many buildings — never to decorate. 6-12 copies reads as a place; anything above 14 is ignored. An ordinary prop stays at 1.
- Be CREATIVE with keywords! Don't use the same objects every time. Examples:
  Buildings: apartment, church, castle, tower, warehouse, factory, hotel, restaurant, bakery, cinema, museum, cottage, cabin
  Nature: oak, pine, palm, willow, cactus, bush, boulder, stump, mushroom
  Vehicles: sedan, truck, motorcycle, bicycle, taxi, ambulance, van, boat, scooter
  Props: lamppost, hydrant, mailbox, barrel, crate, statue, fountain, well, windmill, flag, phone booth, umbrella, trashcan, planter
  Furniture: sofa, armchair, bookshelf, desk, bed, dresser, TV, piano, rug, clock
- motion_prompt MUST start with "A person" and describe expressive, continuous motion
- 2-3 lights, always include ambient (0.8-1.5). No fog.

OBJECT ANIMATIONS (optional, only when it genuinely fits):
- "animations" is OPTIONAL — most scenes need none at all. Only add one when an object in models[] would obviously be moving in this scene (a car driving down a street, a boat on water, a door swinging open, a ball rolling). Do NOT animate static things like buildings, furniture, or trees.
- At most 1-2 animated objects per scene.
- keyword must exactly match one of the models[] keywords.
- path: 3-5 keyframes, {"t": seconds from 0 (max ~5), "pos": [x,y,z] offset in scene units FROM THAT OBJECT'S OWN PLACED POSITION (not world-absolute — roughly -60..60 per axis), "yaw": additional facing rotation in degrees, 0 = however it was placed, positive = turning left}.
- Keep the motion plausible for that object type and modest in scale — this is a background detail, not the main subject.
`;

// Cheap classification the opening prompt goes through before anything
// else: is this scene ABOUT a person, or about a non-human thing moving on
// its own ("a car arrives", "a ball rolls in")? Everything downstream —
// whether Kimodo runs at all, whether a human character exists — depends
// on this, so it has to happen before the (expensive, several-second)
// motion generation call, not after.
const PRIMARY_TYPE_PROMPT = `Decide the MAIN SUBJECT of this 3D-scene instruction: a person performing an action, or a non-human object/vehicle/animal that moves or arrives with no person involved.
Output ONLY JSON, no markdown.
Person: {"primary":"person"}
Object: {"primary":"object","keyword":"short search term","category":number,"size":"large"|"medium"|"small"}
CATEGORIES: 0=Food, 1=Clutter, 3=Transport, 4=Furniture, 5=Objects, 6=Nature, 7=Animals, 8=Buildings, 11=Other
"size" is real-world scale: a car or animal is "medium", a building is "large", a ball or small prop is "small".
Examples:
"a person walks toward the middle of the platform" -> {"primary":"person"}
"make a car move into the scene" -> {"primary":"object","keyword":"car","category":3,"size":"medium"}
"a ball rolls across the floor" -> {"primary":"object","keyword":"ball","category":5,"size":"small"}
If genuinely unclear, answer {"primary":"person"}.`;

async function classifyPrimaryType(userPrompt) {
    try {
        const data = await fetchGeminiWithRetry(GEMINI_URL, {
            contents: [{ role: 'user', parts: [{ text: PRIMARY_TYPE_PROMPT + '\n\nInstruction: ' + userPrompt }] }],
            generationConfig: jsonGenConfig(0.1, 200),
        });
        const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
        if (!text) return { primary: 'person' }; // fail open to the well-tested path
        return JSON.parse(text.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim());
    } catch {
        return { primary: 'person' };
    }
}

async function callGemini(userPrompt, characterPath = null, primaryObject = null) {
    let fullPrompt = SCENE_SYSTEM_PROMPT + '\n\nUser prompt: ' + userPrompt;
    if (characterPath && characterPath.length > 0) {
        // Send only x,z to Gemini (it doesn't need Y)
        const flatPath = characterPath.map(p => [p[0], p[2] || p[1]]);
        fullPrompt += `\n\nCHARACTER PATH (the character moves through these [x,z] points — DO NOT place any object within 150 units of this path):\n${JSON.stringify(flatPath)}`;
    }
    if (primaryObject) {
        // A separate classification step already named this scene's main
        // subject — without pinning the keyword, this call was free to pick
        // its own synonym ("sedan" for a prompt that said "car"), and the
        // code looking for "car" afterward to give it its entrance simply
        // never found it.
        fullPrompt += `\n\nThe main subject of this scene is already decided: keyword "${primaryObject.keyword}", category ${primaryObject.category}, size "${primaryObject.size}". It MUST appear in models[] using EXACTLY that keyword string, not a synonym — other code matches on it verbatim. Do not add an "animations" entry for it; its motion is a full entrance handled separately, not the small in-place wiggle animations[] is for.`;
    }
    fullPrompt += await getReferenceStyleContext();
    const data = await fetchGeminiWithRetry(GEMINI_URL, {
        contents: [{ role: 'user', parts: [{ text: fullPrompt }] }],
        generationConfig: jsonGenConfig(0.7)
    });
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!text) throw new Error(geminiErrorMessage(data));
    // Strip markdown code fences if present
    const clean = text.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
    return JSON.parse(clean);
}

const gltfLoader = new GLTFLoader();
gltfLoader.register((parser) => new VRMLoaderPlugin(parser)); // no-op for plain .glb props — only kicks in on files carrying VRM extensions
let sceneObjects = [];
let pathLine = null;
let pathVisible = false;

// ========== PROCEDURAL GROUND TEXTURE ==========
// A painted meadow rather than a flat colour: soft sunlit and shaded
// patches, thousands of short upward grass strokes in four greens, and a few
// flower specks. Everything is drawn wrap-aware so the texture tiles without
// seams, and a gentle large-scale colour drift is painted into the vertices
// on top so the tiling never reads as a repeating pattern.
function createTexturedGround(size, baseColor, style = 'meadow') {
    const S = 1024;
    const canvas = document.createElement('canvas');
    canvas.width = S; canvas.height = S;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = baseColor;
    ctx.fillRect(0, 0, S, S);

    const wrapped = (x, y, r, draw) => {
        for (const ox of [-S, 0, S]) for (const oy of [-S, 0, S]) {
            if (x + ox + r < 0 || x + ox - r > S || y + oy + r < 0 || y + oy - r > S) continue;
            draw(x + ox, y + oy);
        }
    };
    // A town ('town' style) is trodden ground — warm dust, worn grass and
    // cobbles — where a meadow is a lush carpet of grass.
    const town = style === 'town';
    // Light and shade patches: sunlit yellow-green and deep green on a
    // meadow; warm dust and worn grass in town.
    for (let i = 0; i < 70; i++) {
        const x = Math.random() * S, y = Math.random() * S, r = 60 + Math.random() * 170;
        const sunny = Math.random() < 0.55;
        const rgb = town ? (sunny ? '222,206,170' : '138,160,102') : (sunny ? '198,234,112' : '62,128,46');
        const a = sunny ? 0.22 + Math.random() * 0.18 : 0.14 + Math.random() * 0.14;
        wrapped(x, y, r, (cx, cy) => {
            const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
            g.addColorStop(0, `rgba(${rgb},${a})`);
            g.addColorStop(1, `rgba(${rgb},0)`);
            ctx.fillStyle = g;
            ctx.fillRect(cx - r, cy - r, r * 2, r * 2);
        });
    }
    // Grass strokes, mostly upright, dark to highlight — sparse, muted tufts
    // between the stones in town.
    const greens = town
        ? ['rgba(104,134,76,0.35)', 'rgba(134,162,96,0.35)', 'rgba(168,184,120,0.3)']
        : ['rgba(64,138,48,0.55)', 'rgba(112,184,70,0.55)', 'rgba(164,218,96,0.6)', 'rgba(210,238,128,0.5)'];
    ctx.lineCap = 'round';
    for (let i = 0; i < (town ? 2600 : 11000); i++) {
        const x = Math.random() * S, y = Math.random() * S;
        const len = 6 + Math.random() * 14, ang = -Math.PI / 2 + (Math.random() - 0.5) * 0.9;
        ctx.strokeStyle = greens[(Math.random() * greens.length) | 0];
        ctx.lineWidth = 1.2 + Math.random() * 1.8;
        wrapped(x, y, len, (cx, cy) => {
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(ang) * len, cy + Math.sin(ang) * len); ctx.stroke();
        });
    }
    if (town) {
        // Cobblestone clusters: rounded stones with a soft dark edge.
        const stones = ['#d3cab8', '#c2b8a4', '#ddd5c5', '#aea38f'];
        for (let c = 0; c < 45; c++) {
            const x0 = Math.random() * S, y0 = Math.random() * S, spread = 50 + Math.random() * 90;
            for (let k = 0; k < 60; k++) {
                const a = Math.random() * Math.PI * 2, d = Math.sqrt(Math.random()) * spread;
                const rx = 5 + Math.random() * 6, ry = rx * (0.6 + Math.random() * 0.3), rot = Math.random() * Math.PI;
                const fill = stones[(Math.random() * stones.length) | 0];
                wrapped(x0 + Math.cos(a) * d, y0 + Math.sin(a) * d, rx, (px, py) => {
                    ctx.beginPath(); ctx.ellipse(px, py, rx, ry, rot, 0, Math.PI * 2);
                    ctx.fillStyle = fill; ctx.fill();
                    ctx.strokeStyle = 'rgba(96,84,68,0.35)'; ctx.lineWidth = 1; ctx.stroke();
                });
            }
        }
        // Pebbles and grit.
        for (let i = 0; i < 1600; i++) {
            const x = Math.random() * S, y = Math.random() * S, r = 0.8 + Math.random() * 1.4;
            ctx.fillStyle = Math.random() < 0.5 ? 'rgba(120,108,92,0.5)' : 'rgba(236,226,206,0.55)';
            wrapped(x, y, r, (px, py) => { ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2); ctx.fill(); });
        }
    } else {
        // Flower specks.
        const petals = ['rgba(255,252,236,0.9)', 'rgba(255,226,102,0.9)', 'rgba(255,196,214,0.8)'];
        for (let i = 0; i < 140; i++) {
            const x = Math.random() * S, y = Math.random() * S, r = 1.4 + Math.random() * 1.4;
            ctx.fillStyle = petals[(Math.random() * petals.length) | 0];
            wrapped(x, y, r, (cx, cy) => { ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill(); });
        }
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(size / 320, size / 320);
    texture.anisotropy = renderer.capabilities.getMaxAnisotropy();

    const geo = new THREE.PlaneGeometry(size, size, 64, 64); // 64x64 subdivisions for terrain
    // Large, slow light/shade drift (warmer where it's lighter) across the
    // whole platform, multiplied over the texture.
    const pos = geo.getAttribute('position');
    const cols = new Float32Array(pos.count * 3);
    const p1 = Math.random() * 10, p2 = Math.random() * 10, p3 = Math.random() * 10;
    for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i), z = pos.getY(i);
        const n = Math.sin(x * 0.0042 + p1) * Math.sin(z * 0.0051 + p2) * 0.6
                + Math.sin((x + z) * 0.0093 + p3) * 0.4;
        cols[i * 3] = 1 + n * 0.14;
        cols[i * 3 + 1] = 1 + n * 0.1;
        cols[i * 3 + 2] = 1 + n * 0.02;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
    const mat = new THREE.MeshStandardMaterial({ map: texture, vertexColors: true, roughness: 1 });
    const plane = new THREE.Mesh(geo, mat);
    plane.rotation.x = -Math.PI / 2;
    return plane;
}

// Ground cover for the meadow (the grass itself is painted into the ground
// texture): round leafy clumps and small flowers scattered across the
// platform, and a sunny patch of worn earth along the character's walk. Cover
// stays clear of the walk so feet stay visible. It all counts as ground
// (userData._isGround), so selection, framing, placement and saving skip it.
function addGroundCover(size, charPath, style = 'meadow') {
    const town = style === 'town'; // small rocks instead of bushes and flowers; a stone path
    // Keep chunky clutter out of the side facing the default camera (+Z), as
    // the props are (see computeLayout), so nothing sits right in front of the lens.
    const inCameraWedge = (x, z) => z > 0 && Math.abs(x) < z * 0.5 + 100;
    const half = size / 2 - 15;
    const walk = (charPath || []).filter(p => p.length >= 3).map(p => [p[0], p[2]]);
    const walkDist = (x, z) => {
        let d = Infinity;
        for (const [px, pz] of walk) d = Math.min(d, Math.hypot(x - px, z - pz));
        return d;
    };
    const addCover = (obj) => {
        obj.userData._isGround = true;
        obj.raycast = () => {}; // never a click target
        scene.add(obj);
        sceneObjects.push(obj);
    };
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3(), at = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);

    // Leafy clumps: a few low-poly blobs merged into one soft bush shape.
    if (!town) {
        const pos = [];
        for (const [x, y, z, r] of [[0, 0.55, 0, 0.6], [0.45, 0.4, 0.15, 0.45], [-0.4, 0.38, -0.2, 0.48], [0.05, 0.35, -0.45, 0.42]]) {
            const g = new THREE.IcosahedronGeometry(r, 0).toNonIndexed();
            const p = g.getAttribute('position');
            for (let i = 0; i < p.count; i++) pos.push(p.getX(i) + x, p.getY(i) * 0.85 + y, p.getZ(i) + z);
            g.dispose();
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        geo.computeVertexNormals();
        const max = Math.min(450, Math.round(size * size / 3400));
        const clumps = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({
            flatShading: true, roughness: 0.9, emissive: 0x3a6a14, emissiveIntensity: 0.35,
        }), max);
        const leafy = [0x5fb842, 0x78c850, 0x94d65c, 0x4fa83a, 0xa6de66].map(h => new THREE.Color(h));
        let n = 0;
        for (let tries = 0; n < max && tries < max * 4; tries++) {
            const x = (Math.random() * 2 - 1) * half, z = (Math.random() * 2 - 1) * half;
            if (walkDist(x, z) < 80 || inCameraWedge(x, z)) continue;
            const w = 12 + Math.random() * 26;
            m4.compose(at.set(x, -1, z), q.setFromAxisAngle(up, Math.random() * Math.PI * 2), sc.set(w * 1.2, w * (0.7 + Math.random() * 0.4), w * 1.2));
            clumps.setMatrixAt(n, m4);
            clumps.setColorAt(n, leafy[(Math.random() * leafy.length) | 0]);
            n++;
        }
        clumps.count = n;
        addCover(clumps);
    }

    // Flowers peeking out of the grass.
    if (!town) {
        const max = Math.min(900, Math.round(size * size / 1600));
        const flowers = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1.6, 0), new THREE.MeshBasicMaterial(), max);
        const petals = [0xfffbe8, 0xfffbe8, 0xffe066, 0xffc2d6].map(h => new THREE.Color(h));
        let n = 0;
        for (let tries = 0; n < max && tries < max * 3; tries++) {
            const x = (Math.random() * 2 - 1) * half, z = (Math.random() * 2 - 1) * half;
            if (walkDist(x, z) < 50) continue;
            m4.compose(at.set(x, 6 + Math.random() * 12, z), q.identity(), sc.set(1, 1, 1));
            flowers.setMatrixAt(n, m4);
            flowers.setColorAt(n, petals[(Math.random() * petals.length) | 0]);
            n++;
        }
        flowers.count = n;
        addCover(flowers);
    }

    // Town clutter: small weathered rocks.
    if (town) {
        const rockMax = Math.min(260, Math.round(size * size / 9000));
        const rocks = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 0), new THREE.MeshStandardMaterial({
            flatShading: true, roughness: 1, emissive: 0x6a6252, emissiveIntensity: 0.25,
        }), rockMax);
        const greys = [0xc9c1b1, 0xb3aa98, 0xd8d0c0, 0xa0978a].map(h => new THREE.Color(h));
        let n = 0;
        for (let tries = 0; n < rockMax && tries < rockMax * 4; tries++) {
            const x = (Math.random() * 2 - 1) * half, z = (Math.random() * 2 - 1) * half;
            if (walkDist(x, z) < 70 || inCameraWedge(x, z)) continue;
            const w = 4 + Math.random() * 10;
            m4.compose(at.set(x, w * 0.2, z), q.setFromAxisAngle(up, Math.random() * Math.PI * 2), sc.set(w, w * (0.45 + Math.random() * 0.3), w * (0.8 + Math.random() * 0.4)));
            rocks.setMatrixAt(n, m4);
            rocks.setColorAt(n, greys[(Math.random() * greys.length) | 0]);
            n++;
        }
        rocks.count = n;
        addCover(rocks);
    }

    // Sunny worn earth along the walk (pale stone in town): soft,
    // ragged-edged decals just above the ground texture.
    if (walk.length) {
        const c = document.createElement('canvas');
        c.width = c.height = 256;
        const ctx = c.getContext('2d');
        for (let i = 0; i < 26; i++) {
            const a = Math.random() * Math.PI * 2, rr = Math.random() * 55;
            const x = 128 + Math.cos(a) * rr, y = 128 + Math.sin(a) * rr, r = 40 + Math.random() * 45;
            const g = ctx.createRadialGradient(x, y, 0, x, y, r);
            const earth = town ? '214,206,190' : '226,202,146';
            g.addColorStop(0, `rgba(${earth},0.55)`);
            g.addColorStop(1, `rgba(${earth},0)`);
            ctx.fillStyle = g;
            ctx.fillRect(0, 0, 256, 256);
        }
        for (let i = 0; i < 260; i++) {
            ctx.fillStyle = Math.random() < 0.5 ? 'rgba(186,158,104,0.35)' : 'rgba(246,230,186,0.4)';
            ctx.fillRect(40 + Math.random() * 176, 40 + Math.random() * 176, 2, 2);
        }
        const tex = new THREE.CanvasTexture(c);
        tex.colorSpace = THREE.SRGBColorSpace;
        const mat = new THREE.MeshStandardMaterial({
            map: tex, transparent: true, depthWrite: false, roughness: 1,
            polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
        });
        const plane = new THREE.PlaneGeometry(1, 1);
        let last = null;
        for (const [px, pz] of walk) {
            if (last && Math.hypot(px - last[0], pz - last[1]) < 35) continue;
            last = [px, pz];
            const decal = new THREE.Mesh(plane, mat);
            const s = 150 + Math.random() * 60;
            decal.scale.set(s, s, 1);
            decal.rotation.set(-Math.PI / 2, 0, Math.random() * Math.PI * 2);
            decal.position.set(px + (Math.random() - 0.5) * 20, 0.4, pz + (Math.random() - 0.5) * 20);
            addCover(decal);
        }
    }
}

// Displace terrain vertices based on path Y values
function applyTerrainFromPath(terrainMesh, pathPoints) {
    if (!pathPoints || pathPoints.length < 2) return;
    const baseY = pathPoints[0][1] || 94;

    // Detect sustained elevation: look at the trend, not spikes (jumps)
    // Smooth the Y values first to filter out jumps
    const smoothedY = pathPoints.map((p, i) => {
        const y = p[1] || 94;
        // Average with neighbors (3-point moving average)
        const prev = (pathPoints[i - 1]?.[1] || y);
        const next = (pathPoints[i + 1]?.[1] || y);
        return (prev + y + next) / 3;
    });

    // Check if there's a sustained trend (not just bouncing)
    const maxSmoothedDelta = Math.max(...smoothedY.map(y => Math.abs(y - baseY)));
    // No meaningful elevation. A walk's hips bob several units on their own;
    // at a 3-unit threshold that bob dented the ground along every path —
    // invisible on plain grass, but it sank the stage below the meadow around
    // it, which showed through as a green disc in town.
    if (maxSmoothedDelta < 12) return;

    // Real terrain changes elevation gradually while the character covers
    // ground (walking up/down a slope); a pose change — sitting, kneeling,
    // collapsing — drops the Hips height sharply with almost no horizontal
    // travel. "One dies" used to read as a steep hill and carve a crater
    // into the ground right under wherever they fell. If any consecutive
    // pair of waypoints is steeper than a generous walkable incline, treat
    // the whole path as "not a slope" and leave the terrain flat.
    const MAX_WALKABLE_SLOPE = 1.2; // vertical units per horizontal unit
    for (let i = 1; i < pathPoints.length; i++) {
        const dx = pathPoints[i][0] - pathPoints[i - 1][0];
        const dz = (pathPoints[i][2] || 0) - (pathPoints[i - 1][2] || 0);
        const horiz = Math.hypot(dx, dz);
        const dy = Math.abs(smoothedY[i] - smoothedY[i - 1]);
        if (dy > 2 && dy / (horiz + 1) > MAX_WALKABLE_SLOPE) return;
    }

    const posAttr = terrainMesh.geometry.getAttribute('position');

    for (let i = 0; i < posAttr.count; i++) {
        // PlaneGeometry is created in XY plane, then rotated -90° on X.
        // Before rotation: local X = world X, local Y = world -Z
        const vx = posAttr.getX(i);   // world X
        const vz = -posAttr.getY(i);  // world Z (negated!)

        // Find two closest path points and interpolate — allows both incline AND decline
        let best1 = { d: Infinity, h: 0 }, best2 = { d: Infinity, h: 0 };

        for (let j = 0; j < pathPoints.length; j++) {
            const p = pathPoints[j];
            const px = p[0], pz = p[2] || 0;
            const d = Math.sqrt((vx - px) ** 2 + (vz - pz) ** 2);
            const h = smoothedY[j] - baseY;
            if (d < best1.d) {
                best2 = { ...best1 };
                best1 = { d, h };
            } else if (d < best2.d) {
                best2 = { d, h };
            }
        }

        // Interpolate between two nearest for smooth slopes
        const totalD = best1.d + best2.d;
        const height = totalD > 0.1
            ? (best1.h * (1 - best1.d / totalD) + best2.h * (1 - best2.d / totalD))
            : best1.h;

        // Wider falloff near the path, tight enough to look like terrain
        const influence = Math.exp(-(best1.d * best1.d) / (120 * 120));
        // Rise for a slope, but never sink below the stage: a dip opened a
        // hole onto the meadow underneath, right where the character stands.
        posAttr.setZ(i, Math.max(0, height * influence * 1.2));
    }
    posAttr.needsUpdate = true;
    terrainMesh.geometry.computeVertexNormals();
}

// ========== PATH VISUALIZATION ==========
// Extract path from an AnimationClip directly (not BVH text)
function extractPathFromClip(clip) {
    const posTrack = clip.tracks.find(t => t.name.includes('Hips') && t.name.endsWith('.position'));
    if (!posTrack) return [[0, 0, 0]];
    const values = posTrack.values;
    const totalFrames = values.length / 3;
    const path = [];
    const numSamples = Math.min(40, totalFrames);
    const step = Math.max(1, Math.floor(totalFrames / numSamples));
    for (let i = 0; i < totalFrames; i += step) {
        path.push([Math.round(values[i * 3]), Math.round(values[i * 3 + 1]), Math.round(values[i * 3 + 2])]);
    }
    if (totalFrames > 0) {
        path.push([Math.round(values[(totalFrames - 1) * 3]), Math.round(values[(totalFrames - 1) * 3 + 1]), Math.round(values[(totalFrames - 1) * 3 + 2])]);
    }
    return path;
}

// Spawn fill objects around new path areas that don't have coverage yet
function extendEnvironmentAlongPath(newPath) {
    const MIN_DIST = 60;
    const existingPositions = [];
    sceneObjects.forEach(obj => {
        if (obj.userData._isGround) return;
        existingPositions.push([obj.position.x, obj.position.z]);
    });

    const fillKeywords = ['tree', 'pine', 'bush', 'rock'];
    let added = 0;

    for (const [px, , pz] of newPath) {
        // Spawn objects on both sides of the path at this waypoint
        for (let attempt = 0; attempt < 8; attempt++) {
            const side = attempt % 2 === 0 ? -1 : 1;
            const offsetX = side * (120 + Math.random() * 250);
            const offsetZ = (Math.random() - 0.5) * 200;
            const fx = px + offsetX;
            const fz = (pz || 0) + offsetZ;

            // Check distance from existing objects
            let tooClose = false;
            for (const [ex, ez] of existingPositions) {
                if (Math.sqrt((fx - ex) ** 2 + (fz - ez) ** 2) < MIN_DIST) {
                    tooClose = true; break;
                }
            }
            // Check distance from path
            for (const [ppx, , ppz] of newPath) {
                if (Math.sqrt((fx - ppx) ** 2 + (fz - (ppz||0)) ** 2) < 100) {
                    tooClose = true; break;
                }
            }
            if (tooClose) continue;

            existingPositions.push([fx, fz]);
            const kw = fillKeywords[Math.floor(Math.random() * fillKeywords.length)];
            const scale = kw === 'rock' ? 15 + Math.random() * 15
                : kw === 'bush' ? 25 + Math.random() * 15
                : 120 + Math.random() * 40;

            const proc = tryProceduralModel(kw, scale);
            if (proc) {
                proc.position.set(fx, 0, fz);
                proc.rotation.y = Math.random() * Math.PI * 2;
                scene.add(proc);
                sceneObjects.push(proc);
                added++;
            }
        }
    }
    if (added > 0) log(`Extended environment (${added} objects)`, 'scene');
}

function updatePathVisualization(charPath) {
    // Remove old path
    if (pathLine) { scene.remove(pathLine); pathLine = null; }
    if (!charPath || charPath.length < 2) return;

    const group = new THREE.Group();

    // Main path line
    const baseY = charPath[0]?.[1] || 94;
    const points = charPath.map(p => {
        const elevation = (p[1] || 94) - baseY; // height relative to baseline
        return new THREE.Vector3(p[0], elevation + 3, p[2] || 0); // slightly above terrain
    });
    const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
    const lineMat = new THREE.LineBasicMaterial({ color: 0x7c5cbf });
    const line = new THREE.Line(lineGeo, lineMat);
    group.add(line);

    // Dashed ground shadow of the path
    const shadowPoints = charPath.map(p => {
        const elevation = (p[1] || 94) - baseY;
        return new THREE.Vector3(p[0], elevation + 0.5, p[2] || 0);
    });
    const shadowGeo = new THREE.BufferGeometry().setFromPoints(shadowPoints);
    const shadowMat = new THREE.LineDashedMaterial({ color: 0x7c5cbf, dashSize: 8, gapSize: 6, opacity: 0.3, transparent: true });
    const shadowLine = new THREE.Line(shadowGeo, shadowMat);
    shadowLine.computeLineDistances();
    group.add(shadowLine);

    // Waypoint markers
    const markerGeo = new THREE.SphereGeometry(3, 8, 6);
    const markerMat = new THREE.MeshStandardMaterial({ color: 0x7c5cbf, emissive: 0x4a3080, emissiveIntensity: 0.3 });
    charPath.forEach((p, i) => {
        const marker = new THREE.Mesh(markerGeo, markerMat);
        const elev = (p[1] || 94) - baseY;
        marker.position.set(p[0], elev + 3, p[2] || 0);
        group.add(marker);

        // Start/end labels — larger markers
        if (i === 0 || i === charPath.length - 1) {
            const big = new THREE.Mesh(
                new THREE.SphereGeometry(5, 10, 8),
                new THREE.MeshStandardMaterial({
                    color: i === 0 ? 0x2d8a4e : 0xc53030,
                    emissive: i === 0 ? 0x1a5530 : 0x801a1a,
                    emissiveIntensity: 0.4
                })
            );
            const bigElev = (p[1] || 94) - baseY;
            big.position.set(p[0], bigElev + 5, p[2] || 0);
            group.add(big);
        }
    });

    // Direction arrows along the path
    const arrowMat = new THREE.MeshStandardMaterial({ color: 0x7c5cbf });
    for (let i = 0; i < points.length - 1; i += 2) {
        const from = points[i], to = points[Math.min(i + 1, points.length - 1)];
        const dir = new THREE.Vector3().subVectors(to, from);
        if (dir.length() < 5) continue;
        const mid = new THREE.Vector3().lerpVectors(from, to, 0.5);
        const arrow = new THREE.Mesh(new THREE.ConeGeometry(2.5, 8, 4), arrowMat);
        arrow.position.copy(mid);
        arrow.position.y = 3;
        arrow.lookAt(to);
        arrow.rotateX(Math.PI / 2);
        group.add(arrow);
    }

    group.visible = pathVisible;
    pathLine = group;
    scene.add(group);
}

function togglePath() {
    pathVisible = !pathVisible;
    if (pathLine) pathLine.visible = pathVisible;
    const btn = document.getElementById('path-toggle');
    btn.classList.toggle('active', pathVisible);
    btn.textContent = pathVisible ? 'Hide Path' : 'Show Path';
}
document.getElementById('path-toggle').addEventListener('click', togglePath);

// ========== BOUNDING BOX DEBUG ==========
let bboxHelpers = [];
let bboxVisible = false;

function showBoundingBoxes() {
    // Clear old
    bboxHelpers.forEach(h => scene.remove(h));
    bboxHelpers = [];

    for (const obj of sceneObjects) {
        if (obj.userData._isGround) continue;
        if (!obj.isGroup && !obj.isMesh && !obj.children?.length) continue;

        const box = new THREE.Box3().setFromObject(obj);
        if (box.isEmpty()) continue;

        const helper = new THREE.Box3Helper(box, 0x00ff88);
        helper.userData._isBbox = true;
        scene.add(helper);
        bboxHelpers.push(helper);
    }
}

function toggleBBox() {
    bboxVisible = !bboxVisible;
    if (bboxVisible) {
        showBoundingBoxes();
    } else {
        bboxHelpers.forEach(h => scene.remove(h));
        bboxHelpers = [];
    }
    const bboxBtn = document.getElementById('bbox-toggle');
    if (bboxBtn) bboxBtn.classList.toggle('active', bboxVisible);
}
const bboxBtn = document.getElementById('bbox-toggle');
if (bboxBtn) bboxBtn.addEventListener('click', toggleBBox);

// ========== PROCEDURAL FALLBACK MODELS ==========
// Used when Poly Pizza doesn't have a good match

function _mat(color, opts = {}) {
    return new THREE.MeshStandardMaterial({ color, roughness: opts.r || 0.8, metalness: opts.m || 0, emissive: opts.e || 0, emissiveIntensity: opts.ei || 0 });
}

function makeTrainingDummy(h) {
    const g = new THREE.Group();
    const wood = _mat(0x8B6914);
    // Base
    const base = new THREE.Mesh(new THREE.CylinderGeometry(12, 14, h * 0.05, 12), _mat(0x5C3A1E));
    base.position.y = h * 0.025;
    g.add(base);
    // Main post
    const post = new THREE.Mesh(new THREE.CylinderGeometry(4, 5, h * 0.7, 10), wood);
    post.position.y = h * 0.4;
    g.add(post);
    // Head target (padded cylinder)
    const head = new THREE.Mesh(new THREE.SphereGeometry(8, 12, 10), _mat(0xcc3333));
    head.position.y = h * 0.85;
    g.add(head);
    // Cross arms at different heights
    for (let i = 0; i < 3; i++) {
        const arm = new THREE.Mesh(new THREE.CylinderGeometry(2, 2, 25, 8), wood);
        arm.rotation.z = Math.PI / 2;
        arm.rotation.y = i * 1.2;
        arm.position.y = h * 0.4 + i * h * 0.15;
        arm.position.x = (i % 2 === 0 ? 1 : -1) * 5;
        g.add(arm);
        // Pad on arm end
        const pad = new THREE.Mesh(new THREE.CylinderGeometry(3.5, 3.5, 6, 8), _mat(0xcc3333));
        pad.rotation.z = Math.PI / 2;
        pad.rotation.y = i * 1.2;
        pad.position.set(arm.position.x + (i % 2 === 0 ? 14 : -14), arm.position.y, 0);
        g.add(pad);
    }
    return g;
}

function makeTatamiMat(h) {
    const g = new THREE.Group();
    // Large floor mat with tatami texture pattern
    const size = 250;
    const base = new THREE.Mesh(new THREE.BoxGeometry(size, 3, size), _mat(0xC2B280));
    base.position.y = 1.5;
    g.add(base);
    // Individual tatami rectangles with borders
    const matW = size / 3 - 2, matD = size / 3 - 2;
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            const border = new THREE.Mesh(new THREE.BoxGeometry(matW, 0.5, matD),
                _mat(r % 2 === c % 2 ? 0xB8A870 : 0xC4B888));
            border.position.set(-size/3 + c * (size/3), 3.5, -size/3 + r * (size/3));
            g.add(border);
            // Edge trim
            const trim = new THREE.Mesh(new THREE.BoxGeometry(matW, 1, 2), _mat(0x2d4a1e));
            trim.position.set(border.position.x, 3.5, border.position.z - matD/2);
            g.add(trim);
        }
    }
    return g;
}

function makePunchingBag(h) {
    const g = new THREE.Group();
    // Ceiling mount bracket
    const bracket = new THREE.Mesh(new THREE.BoxGeometry(8, 3, 8), _mat(0x555555, {m: 0.8}));
    bracket.position.y = h;
    g.add(bracket);
    // Chains (3 of them)
    for (let i = 0; i < 3; i++) {
        const angle = (i / 3) * Math.PI * 2;
        const chain = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, h * 0.2, 4), _mat(0x888888, {m: 0.8}));
        chain.position.set(Math.cos(angle) * 3, h * 0.88, Math.sin(angle) * 3);
        g.add(chain);
    }
    // Bag body (tapered cylinder + rounded bottom)
    const bag = new THREE.Mesh(new THREE.CylinderGeometry(11, 9, h * 0.55, 16), _mat(0x8B1A1A));
    bag.position.y = h * 0.52;
    g.add(bag);
    const bottom = new THREE.Mesh(new THREE.SphereGeometry(9, 16, 8), _mat(0x8B1A1A));
    bottom.scale.y = 0.5;
    bottom.position.y = h * 0.24;
    g.add(bottom);
    // Stitching lines
    for (let i = 0; i < 4; i++) {
        const stitch = new THREE.Mesh(new THREE.BoxGeometry(0.5, h * 0.5, 0.5), _mat(0x440000));
        const a = (i / 4) * Math.PI * 2;
        stitch.position.set(Math.cos(a) * 10.5, h * 0.52, Math.sin(a) * 10.5);
        g.add(stitch);
    }
    return g;
}

function makeLantern(h) {
    const g = new THREE.Group();
    // Hanging cord
    const cord = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, h * 0.15, 4), _mat(0x333333));
    cord.position.y = h * 0.93;
    g.add(cord);
    // Top cap
    const topCap = new THREE.Mesh(new THREE.CylinderGeometry(h * 0.12, h * 0.18, h * 0.08, 6), _mat(0x222222));
    topCap.position.y = h * 0.82;
    g.add(topCap);
    // Paper body (glowing)
    const body = new THREE.Mesh(new THREE.CylinderGeometry(h * 0.2, h * 0.18, h * 0.5, 12),
        _mat(0xdd4444, {e: 0xff6633, ei: 0.4}));
    body.position.y = h * 0.55;
    g.add(body);
    // Ribs
    for (let i = 0; i < 4; i++) {
        const rib = new THREE.Mesh(new THREE.BoxGeometry(0.8, h * 0.5, 0.8), _mat(0x5C3A1E));
        const a = (i / 4) * Math.PI * 2;
        rib.position.set(Math.cos(a) * h * 0.19, h * 0.55, Math.sin(a) * h * 0.19);
        g.add(rib);
    }
    // Bottom cap
    const botCap = new THREE.Mesh(new THREE.CylinderGeometry(h * 0.15, h * 0.08, h * 0.06, 6), _mat(0x222222));
    botCap.position.y = h * 0.28;
    g.add(botCap);
    // Tassel
    const tassel = new THREE.Mesh(new THREE.ConeGeometry(2, h * 0.1, 6), _mat(0xcc2222));
    tassel.position.y = h * 0.2;
    tassel.rotation.x = Math.PI;
    g.add(tassel);
    return g;
}

function makeWeaponRack(h) {
    const g = new THREE.Group();
    const wood = _mat(0x5C3A1E);
    // Vertical posts
    const postL = new THREE.Mesh(new THREE.BoxGeometry(4, h, 4), wood);
    postL.position.set(-25, h/2, 0);
    g.add(postL);
    const postR = new THREE.Mesh(new THREE.BoxGeometry(4, h, 4), wood);
    postR.position.set(25, h/2, 0);
    g.add(postR);
    // Horizontal supports
    for (let i = 0; i < 3; i++) {
        const support = new THREE.Mesh(new THREE.BoxGeometry(54, 3, 6), wood);
        support.position.y = h * 0.2 + i * h * 0.3;
        g.add(support);
        // Weapon on each shelf
        const weaponColor = [0xaa8844, 0x666666, 0x8B6914][i];
        const weapon = new THREE.Mesh(new THREE.CylinderGeometry(1, 1.2, 45, 6), _mat(weaponColor));
        weapon.rotation.z = Math.PI / 2;
        weapon.position.set(0, h * 0.25 + i * h * 0.3, 5);
        g.add(weapon);
    }
    // Top ornament
    const orn = new THREE.Mesh(new THREE.BoxGeometry(58, 3, 8), _mat(0x3a2210));
    orn.position.y = h * 0.95;
    g.add(orn);
    return g;
}

function makeCushion(h) {
    const g = new THREE.Group();
    // Round meditation cushion (zafu)
    const cushion = new THREE.Mesh(new THREE.CylinderGeometry(14, 16, h * 0.4, 16), _mat(0x4a0080));
    cushion.position.y = h * 0.2;
    g.add(cushion);
    // Pleated sides
    for (let i = 0; i < 12; i++) {
        const pleat = new THREE.Mesh(new THREE.BoxGeometry(1, h * 0.35, 3), _mat(0x3a0060));
        const a = (i / 12) * Math.PI * 2;
        pleat.position.set(Math.cos(a) * 14.5, h * 0.2, Math.sin(a) * 14.5);
        pleat.rotation.y = -a;
        g.add(pleat);
    }
    // Flat mat underneath (zabuton)
    const mat = new THREE.Mesh(new THREE.BoxGeometry(40, 3, 40), _mat(0x222244));
    mat.position.y = 1.5;
    g.add(mat);
    return g;
}

function makeShoji(h) {
    const g = new THREE.Group();
    const wood = _mat(0x8B7355);
    const paper = _mat(0xF5F0E0, {e: 0xFFEECC, ei: 0.1});
    // Outer frame
    const frameL = new THREE.Mesh(new THREE.BoxGeometry(3, h, 3), wood);
    frameL.position.set(-40, h/2, 0); g.add(frameL);
    const frameR = new THREE.Mesh(new THREE.BoxGeometry(3, h, 3), wood);
    frameR.position.set(40, h/2, 0); g.add(frameR);
    const frameT = new THREE.Mesh(new THREE.BoxGeometry(83, 3, 3), wood);
    frameT.position.set(0, h - 1.5, 0); g.add(frameT);
    const frameB = new THREE.Mesh(new THREE.BoxGeometry(83, 3, 3), wood);
    frameB.position.set(0, 1.5, 0); g.add(frameB);
    // Inner grid (3x4 panels)
    for (let c = 0; c < 3; c++) {
        for (let r = 0; r < 4; r++) {
            const pw = 24, ph = h * 0.22;
            const panel = new THREE.Mesh(new THREE.PlaneGeometry(pw, ph), paper);
            panel.position.set(-25 + c * 25, h * 0.15 + r * h * 0.23, 0.5);
            g.add(panel);
            const panelB = panel.clone();
            panelB.position.z = -0.5;
            panelB.rotation.y = Math.PI;
            g.add(panelB);
        }
        // Vertical divider
        if (c < 2) {
            const div = new THREE.Mesh(new THREE.BoxGeometry(2, h - 6, 2), wood);
            div.position.set(-13 + c * 25, h/2, 0);
            g.add(div);
        }
    }
    // Horizontal dividers
    for (let r = 1; r < 4; r++) {
        const hdiv = new THREE.Mesh(new THREE.BoxGeometry(77, 2, 2), wood);
        hdiv.position.set(0, h * 0.14 + r * h * 0.23 - h * 0.11, 0);
        g.add(hdiv);
    }
    return g;
}

function makeScroll(h) {
    const g = new THREE.Group();
    // Hanging scroll (kakejiku)
    const scrollBody = new THREE.Mesh(new THREE.PlaneGeometry(h * 0.4, h * 0.8),
        _mat(0xF5F0E0, {e: 0xFFEECC, ei: 0.05}));
    scrollBody.position.y = h * 0.5;
    g.add(scrollBody);
    // Top roller
    const topRoll = new THREE.Mesh(new THREE.CylinderGeometry(2, 2, h * 0.45, 8), _mat(0x3a2210));
    topRoll.rotation.z = Math.PI / 2;
    topRoll.position.y = h * 0.92;
    g.add(topRoll);
    // Bottom roller
    const botRoll = new THREE.Mesh(new THREE.CylinderGeometry(1.5, 1.5, h * 0.48, 8), _mat(0x3a2210));
    botRoll.rotation.z = Math.PI / 2;
    botRoll.position.y = h * 0.1;
    g.add(botRoll);
    // Calligraphy mark (simple rectangle as character)
    const mark = new THREE.Mesh(new THREE.PlaneGeometry(h * 0.15, h * 0.3), _mat(0x111111));
    mark.position.set(0, h * 0.5, 0.3);
    g.add(mark);
    return g;
}

function makeGenericBox(h, color) {
    const g = new THREE.Group();
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(h * 0.6, h, h * 0.6), _mat(color || 0x888888));
    mesh.position.y = h / 2;
    g.add(mesh);
    return g;
}

// Keywords that should use procedural fallback instead of Poly Pizza
// Basic nature procedural models (used for fill and welcome scene)
function makeTree(h) {
    const g = new THREE.Group();
    // Trunk
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(h*0.04, h*0.06, h*0.45, 6), _mat(0x6B4226));
    trunk.position.y = h * 0.22;
    g.add(trunk);
    // Foliage layers (3 cones stacked)
    const leafMat = _mat(0x3a7a2a);
    for (let i = 0; i < 3; i++) {
        const r = h * (0.25 - i * 0.05);
        const cone = new THREE.Mesh(new THREE.ConeGeometry(r, h * 0.28, 7), leafMat);
        cone.position.y = h * (0.42 + i * 0.18);
        g.add(cone);
    }
    return g;
}

function makePine(h) {
    const g = new THREE.Group();
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(h*0.03, h*0.05, h*0.5, 5), _mat(0x5C3A1E));
    trunk.position.y = h * 0.25;
    g.add(trunk);
    // Tall narrow pine shape (4 cones)
    const leafMat = _mat(0x2d5a1e);
    for (let i = 0; i < 4; i++) {
        const r = h * (0.18 - i * 0.03);
        const cone = new THREE.Mesh(new THREE.ConeGeometry(r, h * 0.22, 6), leafMat);
        cone.position.y = h * (0.4 + i * 0.15);
        g.add(cone);
    }
    return g;
}

function makeBush(h) {
    h = Math.min(h, 20);
    const g = new THREE.Group();
    const mat = _mat(0x4a8a3a);
    // Cluster of spheres
    for (let i = 0; i < 3; i++) {
        const s = new THREE.Mesh(new THREE.SphereGeometry(h * (0.35 + Math.random() * 0.15), 7, 5), mat);
        s.position.set((Math.random() - 0.5) * h * 0.3, h * 0.3, (Math.random() - 0.5) * h * 0.3);
        g.add(s);
    }
    return g;
}

function makeRock(h) {
    const g = new THREE.Group();
    const geo = new THREE.DodecahedronGeometry(h * 0.4, 1);
    // Slightly deform vertices for organic look
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
        pos.setY(i, pos.getY(i) * 0.6);
        pos.setX(i, pos.getX(i) * (0.8 + Math.random() * 0.4));
        pos.setZ(i, pos.getZ(i) * (0.8 + Math.random() * 0.4));
    }
    geo.computeVertexNormals();
    const rock = new THREE.Mesh(geo, _mat(0x888078));
    rock.position.y = h * 0.2;
    g.add(rock);
    return g;
}

// Trees and bushes are deliberately NOT here any more: they fall through to
// the library's own tree.glb / bush.glb (see MODEL_MAP), so generated scenes
// match the soft low-poly look of the Library instead of code-built cones.
const PROCEDURAL_KEYWORDS = {
    'rock': makeRock,
    'boulder': makeRock,
    'training dummy': makeTrainingDummy,
    'dummy': makeTrainingDummy,
    'punching bag': makePunchingBag,
    'heavy bag': makePunchingBag,
    'tatami': makeTatamiMat,
    'tatami mat': makeTatamiMat,
    'lantern': makeLantern,
    'paper lantern': makeLantern,
    'weapon rack': makeWeaponRack,
    'rack': makeWeaponRack,
    'cushion': makeCushion,
    'meditation cushion': makeCushion,
    'shoji': makeShoji,
    'shoji screen': makeShoji,
    'scroll': makeScroll,
    'calligraphy': makeScroll,
    'banner': makeScroll,
};

function tryProceduralModel(keyword, height) {
    const kw = keyword.toLowerCase();
    for (const [key, builder] of Object.entries(PROCEDURAL_KEYWORDS)) {
        if (kw.includes(key)) return builder(height);
    }
    return null;
}

// Model lookup + GLB loading (with client-side layout for positioning)
const modelCache = {};

// Pre-generated GLB models are served from local /models/ folder
// Map keywords to local model files — multiple keywords per model
const MODEL_MAP = {
    'tree': ['tree','oak','maple','willow','birch','elm','pine','spruce','fir','palm','cypress'],
    'bush': ['bush','shrub','hedge','plant','fern'],
    'building': ['building','apartment','office','skyscraper','tower','warehouse','factory','hospital','school','museum','library','hotel','cinema','shop','store'],
    'house': ['house','cottage','cabin','hut','villa','home','bungalow'],
    'truck': ['truck','van','bus','pickup','lorry'],
};

// Build reverse lookup: keyword -> filename
const KEYWORD_TO_FILE = {};
for (const [file, keywords] of Object.entries(MODEL_MAP)) {
    for (const kw of keywords) {
        KEYWORD_TO_FILE[kw] = `models/${file}.glb`;
    }
}

async function findLocalModel(keyword, category) {
    const kw = keyword.toLowerCase().trim();
    // Exact match
    if (KEYWORD_TO_FILE[kw]) return KEYWORD_TO_FILE[kw];
    // Partial match — check if any mapped keyword is contained in the request
    for (const [mapped, file] of Object.entries(KEYWORD_TO_FILE)) {
        if (kw.includes(mapped) || mapped.includes(kw)) return file;
    }
    return null;
}

// The library GLBs don't set metallicFactor, and glTF's default for that is
// 1.0 — fully metal. With no environment map to reflect, metal renders
// nearly black, which is why props came in so much darker than their
// thumbnails. They're painted, not metal: make them matte so their texture
// shows at its real colour.
function matteGLBMaterials(model) {
    model.traverse(child => {
        if (!child.isMesh || !child.material) return;
        for (const mat of Array.isArray(child.material) ? child.material : [child.material]) {
            if (mat.isMeshStandardMaterial && !mat.metalnessMap) {
                mat.metalness = 0;
                mat.roughness = Math.max(mat.roughness, 0.85);
                // A little of the prop's own colour as self-light: shaded
                // sides stay soft and pastel instead of going muddy, for the
                // bright painted look of the Create stage.
                if (mat.map) {
                    mat.emissive = new THREE.Color(0xfff6ec);
                    mat.emissiveMap = mat.map;
                    mat.emissiveIntensity = 0.32;
                }
            }
        }
    });
}

// A filled-in city or forest places the same file dozens of times; fetch and
// parse each file once, then hand out clones. Each clone gets its own
// materials (textures stay shared), so recolouring one copy never repaints
// the rest.
const glbTemplates = {};

// The library's trees and bushes are painted in olive, dusky greens that
// read as dark next to a sunny meadow. Push their colour toward fresh
// yellow-green (a colour multiplier above 1 brightens the texture) and give
// them more of their own soft glow, so foliage looks sunlit.
function brightenFoliage(model) {
    model.traverse(child => {
        if (!child.isMesh || !child.material) return;
        for (const mat of Array.isArray(child.material) ? child.material : [child.material]) {
            if (!mat.isMeshStandardMaterial) continue;
            mat.color.setRGB(1.25, 1.42, 1.05);
            mat.emissive = new THREE.Color(0xeaffc8);
            mat.emissiveIntensity = 0.5;
        }
    });
}

// Some generated models (house.glb) stand on a thin dark base plate baked in
// with the mesh, about twice as wide as the walls. One house hides it; a
// street of them joins the plates into black puddles. The plate is the only
// thing at the very bottom that reaches well past the body above it, so trim
// those triangles. A model whose bottom isn't wider than its body — a tree
// whose canopy is wider than its roots, a lamp — is left untouched.
function trimBasePlate(model) {
    model.traverse(child => {
        if (!child.isMesh || !child.geometry) return;
        const g = child.geometry;
        const pos = g.getAttribute('position');
        if (!pos || pos.count < 50) return;
        g.computeBoundingBox();
        const { min, max } = g.boundingBox;
        const h = max.y - min.y;
        if (h <= 0) return;
        const cx = (min.x + max.x) / 2, cz = (min.z + max.z) / 2;
        const lowY = min.y + h * 0.05, bodyY = min.y + h * 0.1;
        let bodyX = 0, bodyZ = 0, lowX = 0, lowZ = 0;
        for (let i = 0; i < pos.count; i++) {
            const dx = Math.abs(pos.getX(i) - cx), dz = Math.abs(pos.getZ(i) - cz), y = pos.getY(i);
            if (y > bodyY) { bodyX = Math.max(bodyX, dx); bodyZ = Math.max(bodyZ, dz); }
            else if (y < lowY) { lowX = Math.max(lowX, dx); lowZ = Math.max(lowZ, dz); }
        }
        const hasPlate = bodyX && bodyZ && (lowX >= bodyX * 1.3 || lowZ >= bodyZ * 1.3);
        // The plate is also painted near-black, and some of it hides inside
        // the body's outline (under roof eaves, or on a model whose plate is
        // no wider than its body), so dark, flat triangles right at the bottom
        // go too — looked up in the model's own texture.
        const uv = g.getAttribute('uv');
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        const img = (mats.find(m => m && m.map && m.map.image) || {}).map?.image;
        let texel = null;
        if (uv && img && img.width && img.height) {
            const cv = document.createElement('canvas');
            cv.width = img.width; cv.height = img.height;
            const cctx = cv.getContext('2d', { willReadFrequently: true });
            cctx.drawImage(img, 0, 0);
            const data = cctx.getImageData(0, 0, img.width, img.height).data;
            texel = (u, v) => { // glTF textures aren't flipped: v runs down the image
                const x = Math.min(img.width - 1, Math.floor((((u % 1) + 1) % 1) * img.width));
                const y = Math.min(img.height - 1, Math.floor((((v % 1) + 1) % 1) * img.height));
                const o = (y * img.width + x) * 4;
                return (0.2126 * data[o] + 0.7152 * data[o + 1] + 0.0722 * data[o + 2]) / 255;
            };
        }
        if (!hasPlate && !texel) return;
        const darkY = min.y + h * 0.08;
        const ta = new THREE.Vector3(), tb = new THREE.Vector3(), tc = new THREE.Vector3(), tn = new THREE.Vector3();
        const idx = g.index ? g.index.array : null;
        const triCount = idx ? idx.length / 3 : pos.count / 3;
        const kept = [];
        for (let t = 0; t < triCount; t++) {
            const v = [0, 1, 2].map(k => idx ? idx[t * 3 + k] : t * 3 + k);
            const flatLow = v.every(i => pos.getY(i) < lowY);
            const outside = hasPlate && flatLow && (Math.abs(v.reduce((s, i) => s + pos.getX(i), 0) / 3 - cx) > bodyX * 1.02
                || Math.abs(v.reduce((s, i) => s + pos.getZ(i), 0) / 3 - cz) > bodyZ * 1.02);
            let darkFloor = false;
            if (!outside && texel && v.every(i => pos.getY(i) < darkY)) {
                ta.fromBufferAttribute(pos, v[0]); tb.fromBufferAttribute(pos, v[1]); tc.fromBufferAttribute(pos, v[2]);
                tn.subVectors(tc, tb).cross(ta.sub(tb)).normalize();
                if (Math.abs(tn.y) > 0.6) { // lying flat, not a wall or a bush's side
                    const u = (uv.getX(v[0]) + uv.getX(v[1]) + uv.getX(v[2])) / 3;
                    const w = (uv.getY(v[0]) + uv.getY(v[1]) + uv.getY(v[2])) / 3;
                    darkFloor = texel(u, w) < 0.2;
                }
            }
            if (!outside && !darkFloor) kept.push(...v);
        }
        if (kept.length !== triCount * 3) g.setIndex(kept);
    });
}

function loadGLBTemplate(url) {
    if (!glbTemplates[url]) {
        glbTemplates[url] = gltfLoader.loadAsync(url).then(gltf => {
            trimBasePlate(gltf.scene);
            matteGLBMaterials(gltf.scene);
            if (/\/(tree|bush)\.glb$/.test(url)) brightenFoliage(gltf.scene);
            return gltf.scene;
        });
        glbTemplates[url].catch(() => delete glbTemplates[url]); // let a failed load retry later
    }
    return glbTemplates[url];
}

// Fetch and parse the props Create scenes are built from in the background
// as soon as the app is up, so building a scene never waits on them.
setTimeout(() => CREATE_PROP_FILES.forEach(url => loadGLBTemplate(url).catch(() => {})), 0);

async function loadGLBModel(url, position, targetHeight, rotationY) {
    return new Promise((resolve) => {
        loadGLBTemplate(url).then((template) => {
            const model = template.clone();
            model.traverse(child => {
                if (child.isMesh && child.material) {
                    child.material = Array.isArray(child.material) ? child.material.map(m => m.clone()) : child.material.clone();
                }
            });
            // Only hide untextured meshes that are pure white or very dark (import artifacts)
            // Textured meshes (with .map) should never be hidden
            model.traverse(child => {
                if (child.isMesh && child.material && !child.material.map) {
                    const c = child.material.color;
                    if (!c) return;
                    if (c.r > 0.95 && c.g > 0.95 && c.b > 0.95) child.visible = false;
                    if (c.r < 0.08 && c.g < 0.08 && c.b < 0.08) child.visible = false;
                }
            });
            const box = new THREE.Box3().setFromObject(model);
            const size = new THREE.Vector3();
            box.getSize(size);
            const s = targetHeight / (size.y || 1);
            model.scale.setScalar(s);
            const scaledBox = new THREE.Box3().setFromObject(model);
            // A model can sit a little into the ground (PROP_SHAPES sink) so
            // what's left of a baked base plate stays below the surface.
            const sink = ((PROP_SHAPES[url] && PROP_SHAPES[url].sink) || 0) * (scaledBox.max.y - scaledBox.min.y);
            model.position.set(position[0], position[1] - scaledBox.min.y - sink, position[2]);
            if (rotationY) model.rotation.y = rotationY;
            scene.add(model);
            sceneObjects.push(model);
            resolve(model);
        }).catch((err) => { console.error('GLB load error:', err); resolve(null); });
    });
}

// Scenes start empty: only props you actually ask for get placed. Flip this
// on to bring back the old behaviour of scattering filler trees/rocks around.
const AUTO_FILL_ENVIRONMENT = false;

// Set once per generate() from Gemini's own scene.type, read back later by
// handleAddObject so a follow-up "add trees" uses the same indoor/outdoor
// size table the opening scene did, instead of guessing.
let currentSceneType = 'outdoor';

// Client-side layout engine — positions objects based on size and character path
const BANNED = ['fence','gate','wall','barrier','shelter','bus stop','bus shelter','canopy','awning','stop sign','road barrier','barricade'];

// Outdoor objects that should NEVER appear indoors
const OUTDOOR_ONLY = ['building','apartment','skyscraper','tower','factory','warehouse','church','castle','hotel','hospital','school','museum','house','cottage','cabin','car','sedan','truck','bus','taxi','van','ambulance','motorcycle','bicycle','scooter','boat','hydrant','mailbox','lamppost','traffic','road','highway','bridge','crane','windmill','container'];

const TREE_KEYWORDS = ['tree','pine','oak','willow','palm','cactus'];
const NATURE_KEYWORDS = [...TREE_KEYWORDS, 'bush','rock','boulder','flower','mushroom','stump'];
// Some library models are far from the proportions the size table assumes:
// house.glb is squat and wide, so at a "large" building's height it came in
// ~880 units across and swallowed the street. An entry gives that model its
// own height, and its real half-width (walls, not base plate) as a fraction
// of that height, so spacing matches what's actually drawn.
// Tree size relative to the person, chosen by comparing builds: 1.4× read as
// "about my height", 2.2× crowded the sky out of shot, 1.9× (so 1.5-2.4× per
// tree with the variety below) clearly out-tops the character and still
// leaves sky between the crowns. Keep these for future scenes.
const TREE_HEIGHT_RATIO = 1.9;           // tree height as a multiple of the character's
const NATURE_SIZE_RANGE = [0.8, 1.25];   // per-tree variety around that
const PROP_SHAPES = {
    'models/house.glb': { height: 300, footprint: 0.64, sink: 0.06 }, // sink: its dark base plate sits in the bottom ~6%
    'models/building.glb': { sink: 0.05 },
    'models/truck.glb': { height: 130, footprint: 1.1 }, // ~320 long: a little bigger than a car, well under a house
    // Trees are sized against the person: TREE_HEIGHT_RATIO × the character's
    // height, then 80-125% of that per tree (see computeLayout). 1.4× read as
    // "about my height"; a flat 435 with more variety towered and hid the sky.
    'models/tree.glb': { height: CHAR_HEIGHT * TREE_HEIGHT_RATIO, footprint: 0.25 },
};
// Which library file a keyword loads — the same exact-then-partial lookup as
// findLocalModel, so it always names the same file.
function modelFileFor(keyword) {
    const kw = keyword.toLowerCase().trim();
    return KEYWORD_TO_FILE[kw]
        || (Object.entries(KEYWORD_TO_FILE).find(([mapped]) => kw.includes(mapped) || mapped.includes(kw)) || [])[1]
        || null;
}
function propShape(keyword) {
    const file = modelFileFor(keyword);
    return (file && PROP_SHAPES[file]) || null;
}

// Every library file a Create scene can use (see MODEL_MAP) — preloaded at startup.
const CREATE_PROP_FILES = Object.keys(MODEL_MAP).map(name => `models/${name}.glb`);
const footprintRadius = (keyword, scale) => {
    const shape = propShape(keyword);
    if (shape && shape.footprint) return scale * shape.footprint;
    return scale * (NATURE_KEYWORDS.some(n => keyword.toLowerCase().includes(n)) ? 0.25 : 0.45);
};

const SIZE_SCALES = {
    outdoor: { large: 435, medium: 120, small: 60 },
    indoor:  { large: 175, medium: 80, small: 50 }
};

// Props ring the edge of the stage; this much of the middle stays open so
// the character has room to move.
const CENTER_CLEAR = { outdoor: 260, indoor: 150 };

// What a place is made of. A scene counts as one when Gemini repeats (gives
// a count to) one of `madeOf`; its edges then fill from `mix` (see
// computeLayout), and `only`, if set, is all a generated scene of that kind
// may contain besides what the user named (see buildScene).
const PLACE_PALETTES = {
    city: {
        // A belt of trees this deep behind the buildings, around the edge.
        treeBelt: 260, beltTrees: 60,
        madeOf: ['building', 'apartment', 'skyscraper', 'office', 'tower', 'shop', 'house'],
        mix: [
            { keyword: 'house', size: 'large', category: 8, share: 0.5 },
            { keyword: 'building', size: 'large', category: 8, share: 0.3 },
            { keyword: 'tree', size: 'large', category: 6, share: 0.2 },
        ],
        // Exactly one of each of these joins the street (no cars at all).
        single: [{ keyword: 'truck', size: 'medium', category: 3 }],
    },
    forest: {
        // A full wood, but with sky between the crowns: canopies sit a
        // little closer than the default, the band reaches a bit further in,
        // and the fill is capped well short of a wall of trees. A smaller
        // open middle (still clear of the path) keeps trees around the
        // character, not only in a far ring.
        natureSpacing: 0.85, depth: 0.65, rows: 2, maxExtras: 70, clear: 180, pathClear: 70, falloff: 1.5,
        madeOf: ['tree', 'pine', 'oak', 'willow', 'birch', 'spruce', 'fir', 'maple'],
        mix: [
            { keyword: 'tree', size: 'large', category: 6, share: 0.65 },
            { keyword: 'bush', size: 'medium', category: 6, share: 0.35 },
        ],
        only: ['tree', 'pine', 'oak', 'willow', 'birch', 'spruce', 'fir', 'maple', 'bush', 'shrub', 'hedge', 'fern'],
    },
};
function placeKind(models) {
    const repeated = models.filter(m => (Number(m.count) || 1) >= 2).map(m => m.keyword.toLowerCase());
    for (const [kind, p] of Object.entries(PLACE_PALETTES)) {
        if (repeated.some(kw => p.madeOf.some(w => kw.includes(w)))) return kind;
    }
    return null;
}

function computeLayout(models, sceneType, charPath) {
    const isIndoor = sceneType === 'indoor';

    // Filter banned keywords
    models = models.filter(m => !BANNED.some(b => m.keyword.toLowerCase().includes(b)));

    // For indoor scenes, also filter out outdoor-only objects
    if (isIndoor) {
        models = models.filter(m => !OUTDOOR_ONLY.some(o => m.keyword.toLowerCase().includes(o)));
    }

    // Separate by size
    const large = models.filter(m => m.size === 'large');
    const medium = models.filter(m => m.size === 'medium');
    const small = models.filter(m => m.size === 'small');
    const scales = SIZE_SCALES[isIndoor ? 'indoor' : 'outdoor'];

    const placed = [];

    if (isIndoor) {
        // Indoor: arrange furniture around the character in a room-like layout
        // Back wall items (large)
        large.forEach((m, i) => {
            const spread = large.length > 1 ? (i / (large.length - 1) - 0.5) * 300 : 0;
            placed.push({
                ...m,
                position: [spread, 0, 250],
                scale: scales.large,
                rotationY: 3.14 // face toward character
            });
        });

        // Side items (medium) — alternate left and right
        medium.forEach((m, i) => {
            const side = i % 2 === 0 ? -1 : 1;
            const z = 50 + (i % 3) * 80;
            placed.push({
                ...m,
                position: [side * 200, 0, z],
                scale: scales.medium,
                rotationY: side === -1 ? 1.57 : 4.71
            });
        });

        // Front/scattered items (small)
        small.forEach((m, i) => {
            const angle = (i / small.length) * Math.PI - Math.PI * 0.3;
            const r = 120 + i * 30;
            placed.push({
                ...m,
                position: [Math.cos(angle) * r, 0, Math.sin(angle) * r + 100],
                scale: scales.small,
                rotationY: 0
            });
        });
    } else {
        // Outdoor: props line the edges of the square platform, leaving its
        // middle open for the character. The platform is sized here, from
        // the character's path and how much edge the props need, and handed
        // back as placed.platformHalf so buildScene builds the ground to match.
        const scaleOf = (m) => (propShape(m.keyword) || {}).height || scales[m.size] || scales.medium;
        const fp = (m) => footprintRadius(m.keyword, scaleOf(m));
        // Mix sizes around the edge instead of all the big ones on one side.
        const ordered = [];
        for (let i = 0; i < Math.max(large.length, medium.length, small.length); i++) {
            if (medium[i]) ordered.push(medium[i]);
            if (large[i]) ordered.push(large[i]);
            if (small[i]) ordered.push(small[i]);
        }
        // Props a place is made of take that place's size, so Gemini's own
        // copies and the fill agree. Gemini often calls forest trees
        // "medium"; the same-keyword pass below then shrank every tree to
        // match, leaving the fill spaced for big trees but drawn small — a
        // sparse field of saplings instead of a wood.
        const sizePalette = PLACE_PALETTES[placeKind(ordered)];
        if (sizePalette) for (const m of ordered) {
            const kw = m.keyword.toLowerCase();
            // "oak" or "pine" doesn't contain "tree" — anything the place is
            // made of counts as its main prop (the mix's first entry).
            const item = sizePalette.mix.find(it => kw.includes(it.keyword))
                || (sizePalette.madeOf.some(w => kw.includes(w)) ? sizePalette.mix[0] : null);
            if (item) m.size = item.size;
        }
        // A city of only tall buildings reads as an office district, not a
        // town: every other copy of Gemini's own buildings becomes a house.
        if (sizePalette === PLACE_PALETTES.city) {
            let flip = false;
            for (const m of ordered) {
                const kw = m.keyword.toLowerCase();
                if (kw.includes('house') || !sizePalette.madeOf.some(w => kw.includes(w))) continue;
                if (flip) { m.keyword = 'house'; m.size = 'large'; }
                flip = !flip;
            }
        }

        let pathExtent = 0;
        for (const [px, , pz] of charPath) pathExtent = Math.max(pathExtent, Math.abs(px), Math.abs(pz || 0));
        // Include the fill's own props: a plan of only small ones must not
        // shrink the platform below what the fill needs to stand on.
        const maxFp = [...ordered, ...((sizePalette && sizePalette.mix) || [])].reduce((a, m) => Math.max(a, fp(m)), 0);
        // Nature can crowd (canopies overlap nicely); buildings need elbow room.
        const edgeNeeded = ordered.reduce((a, m) => a + fp(m) * (NATURE_KEYWORDS.some(n => m.keyword.toLowerCase().includes(n)) ? 1.3 : 2.1), 0);
        // The open middle clears the whole path. The edge band starts past it
        // with room for a rim row plus the thinner rows inside, so props can
        // land along every side — when the open circle reached past the rim,
        // only the corners had room and a city stalled at a handful of houses.
        const kindPalette = PLACE_PALETTES[placeKind(ordered)] || {};
        const openR = Math.max(kindPalette.clear || CENTER_CLEAR.outdoor, pathExtent + (kindPalette.pathClear || 120));
        const H = Math.max(420, openR + maxFp * 3.2, edgeNeeded / 8 + maxFp);
        placed.platformHalf = H;

        // A place should cover its edges, not dot them — the prompt only asks
        // for 6-12 copies. When Gemini repeats the prop a location is built
        // from (see PLACE_PALETTES), fill the edge band from that place's
        // mix, up to what the band holds: one spaced row along the rim plus
        // the thinner rows inward (~1.8 rows' worth). Extras go in after
        // Gemini's own props, the big tier (houses, buildings) before the
        // small one so the big ones get their spots and the rest fill gaps,
        // round-robin within a tier so the street stays mixed. Any extra that
        // finds no uncrowded spot is dropped, so filling never overlaps.
        const palette = PLACE_PALETTES[placeKind(ordered)];
        // How tightly nature packs, how far in the band reaches (as a share of
        // the half-size) and how many rows' worth it holds — a palette can
        // override these (forests pack far denser than a city's street trees).
        const natureSpacing = (palette && palette.natureSpacing) || 0.9;
        const bandDepth = (palette && palette.depth) || 0.6;
        if (palette) {
            const MAX_EXTRAS = palette.maxExtras || 60;
            const spacingOf = (item) => (NATURE_KEYWORDS.some(n => item.keyword.includes(n)) ? natureSpacing : 1.6) * fp(item);
            const avgSpacing = palette.mix.reduce((a, item) => a + item.share * spacingOf(item), 0);
            const bandHolds = Math.round(8 * (H - maxFp * 1.2) / avgSpacing * (palette.rows || 1.8));
            const perItem = palette.mix.map(item => ({
                item,
                want: Math.max(0, Math.round(bandHolds * item.share) - ordered.filter(m => m.keyword.toLowerCase().includes(item.keyword)).length),
            }));
            const total = perItem.reduce((a, p) => a + p.want, 0);
            const squeeze = total > MAX_EXTRAS ? MAX_EXTRAS / total : 1;
            // One-off props go in ahead of the fill, so they're sure of a spot.
            for (const s of palette.single || []) {
                if (!ordered.some(m => m.keyword.toLowerCase().includes(s.keyword))) ordered.push({ ...s, _extra: true });
            }
            for (const bigTier of [true, false]) {
                const queues = perItem
                    .filter(p => (spacingOf(p.item) >= 200) === bigTier)
                    .map(p => Array.from({ length: Math.round(p.want * squeeze) },
                        () => ({ keyword: p.item.keyword, size: p.item.size, category: p.item.category, _extra: true })));
                while (queues.some(q => q.length)) queues.forEach(q => { if (q.length) ordered.push(q.shift()); });
            }
        }

        // Scatter, not a row: each prop picks a random spot along a random
        // edge, then a depth in from it drawn from u^2.2 — so most land
        // right at the rim and progressively fewer reach inward, like a
        // treeline thinning into a clearing. Candidates that crowd another
        // prop, the path, or the open middle are re-rolled.
        const isNat =(m) => NATURE_KEYWORDS.some(n => m.keyword.toLowerCase().includes(n));
        ordered.forEach((m) => {
            const f = fp(m);
            const rim = f * 1.4; // nature gets up to 120% size variety below; keep even the biggest on the platform
            let x = 0, z = 0, found = false;
            for (let attempt = 0; attempt < 40; attempt++) {
                const inset = H - rim - Math.pow(Math.random(), (palette && palette.falloff) || 2.2) * H * bandDepth;
                const along = (Math.random() * 2 - 1) * (H - rim);
                const side = Math.floor(Math.random() * 4);
                [x, z] = side === 0 ? [along, -inset] : side === 1 ? [inset, along] : side === 2 ? [along, inset] : [-inset, along];
                if (Math.hypot(x, z) < openR + f) continue;
                // Leave the side facing the default camera (+Z) open, rim
                // included, so the camera looks into the clearing at the
                // character instead of standing inside the treeline. The
                // scenery wraps the other three sides as a backdrop.
                if (z > 0 && Math.abs(x) < z * 0.5 + 100) continue;
                if (charPath.some(([px, , pz]) => Math.hypot(x - px, z - (pz || 0)) < 120 + f * 0.5)) continue;
                const crowded = placed.some(o => {
                    const sep = (isNat(m) && isNat(o) ? natureSpacing : 1.6) * (f + footprintRadius(o.keyword, o.scale)) / 2;
                    return Math.hypot(x - o.position[0], z - o.position[2]) < sep;
                });
                if (!crowded) { found = true; break; }
            }
            if (!found && m._extra) return; // band's full — a filler copy isn't worth an overlap
            placed.push({ ...m, position: [x, 0, z], scale: scaleOf(m), rotationY: Math.random() * Math.PI * 2 });
        });

        // Trees behind the buildings: a belt around the outside of the
        // platform (which grows to hold it), so a town sits in a green
        // landscape instead of ending at a bare edge. It keeps the same
        // camera-facing opening, and keeps clear of the buildings and of
        // each other like any other prop.
        if (palette && palette.treeBelt) {
            const tree = { keyword: 'tree', size: 'large', category: 6 };
            const f = fp(tree), rim = f * 1.4;
            const outer = H + palette.treeBelt;
            let planted = 0;
            for (let attempt = 0; attempt < 600 && planted < palette.beltTrees; attempt++) {
                const inset = outer - rim - Math.random() * (palette.treeBelt - rim);
                const along = (Math.random() * 2 - 1) * (outer - rim);
                const side = Math.floor(Math.random() * 4);
                const [x, z] = side === 0 ? [along, -inset] : side === 1 ? [inset, along] : side === 2 ? [along, inset] : [-inset, along];
                if (z > 0 && Math.abs(x) < z * 0.5 + 100) continue;
                const crowded = placed.some(o => {
                    const sep = (isNat(o) ? natureSpacing : 1.6) * (f + footprintRadius(o.keyword, o.scale)) / 2;
                    return Math.hypot(x - o.position[0], z - o.position[2]) < sep;
                });
                if (crowded) continue;
                placed.push({ ...tree, position: [x, 0, z], scale: scaleOf(tree), rotationY: Math.random() * Math.PI * 2 });
                planted++;
            }
            placed.platformHalf = outer;
        }

        // === AUTO-FILL: scatter randomly across the map, avoid character path ===
        const fillPositions = [];
        const MIN_FILL_DIST = 50; // min distance between fill objects
        const PATH_FILL_CLEARANCE = 80; // clearance around character path

        function canPlaceFill(x, z) {
            // Don't place at the very center where character spawns
            if (Math.sqrt(x * x + z * z) < 80) return false;
            // Keep clear of character path
            for (const [px, , pz] of charPath) {
                const d = Math.sqrt((x - px) ** 2 + (z - pz) ** 2);
                if (d < PATH_FILL_CLEARANCE) return false;
            }
            // Min spacing from other fill objects
            for (const [fx, fz] of fillPositions) {
                const d = Math.sqrt((x - fx) ** 2 + (z - fz) ** 2);
                if (d < MIN_FILL_DIST) return false;
            }
            // Min spacing from placed scene objects
            for (const obj of placed) {
                const d = Math.sqrt((x - obj.position[0]) ** 2 + (z - obj.position[2]) ** 2);
                if (d < 50) return false;
            }
            return true;
        }

        function addFill(x, z, kw) {
            if (!canPlaceFill(x, z)) return false;
            fillPositions.push([x, z]);
            const fillScale = kw === 'rock' ? 15 + Math.random() * 15
                : kw === 'bush' ? 25 + Math.random() * 15
                : 120 + Math.random() * 40;
            placed.push({
                keyword: kw, category: 6, size: 'large',
                position: [x, 0, z], scale: fillScale,
                rotationY: Math.random() * Math.PI * 2, _isFill: true
            });
            return true;
        }

        // Scatter fill objects randomly across the map
        if (AUTO_FILL_ENVIRONMENT) {
            const FILL_TYPES = ['tree', 'tree', 'pine', 'pine', 'bush', 'rock'];
            const MAP_EXTENT = 550;
            for (let i = 0; i < 60; i++) {
                const x = (Math.random() - 0.5) * 2 * MAP_EXTENT;
                const z = (Math.random() - 0.5) * 2 * MAP_EXTENT;
                const kw = FILL_TYPES[Math.floor(Math.random() * FILL_TYPES.length)];
                addFill(x, z, kw);
            }
        }
    }

    // Push objects away from character path (wider clearance the bigger the
    // object is, so a building doesn't just barely clear the path while
    // still visually looming over it).
    // Outdoor props already sit on the platform's edge band, sized clear of
    // the path — these nudges (and the random relocation below) would only
    // drag them back toward the middle, so they're indoor-only now.
    const PATH_CLEARANCE = 100;
    if (isIndoor) for (const obj of placed) {
        const clearance = PATH_CLEARANCE + footprintRadius(obj.keyword, obj.scale);
        for (const [px, , pz] of charPath) {
            const dx = obj.position[0] - px;
            const dz = obj.position[2] - pz;
            const dist = Math.sqrt(dx * dx + dz * dz);
            if (dist < clearance && dist > 0) {
                const push = (clearance - dist) / dist;
                obj.position[0] += dx * push;
                obj.position[2] += dz * push;
            }
        }
    }

    // Extra: clear a wide corridor in front of the character (Z > 0, near X=0),
    // widened per object so a big prop doesn't just clear the corridor's edge
    // and still block the view.
    if (isIndoor) for (const obj of placed) {
        const [ox, , oz] = obj.position;
        const fr = footprintRadius(obj.keyword, obj.scale);
        if (oz > -50 && oz < 300 && Math.abs(ox) < 150 + fr) {
            // Push outward from center
            const side = ox >= 0 ? 1 : -1;
            obj.position[0] = side * (150 + fr + Math.random() * 50);
        }
    }

    // Same-keyword consistency:
    // - Trees/shrubs: vary within 40% of each other
    // - Everything else: exact same size
    const keywordScales = {};
    for (const obj of placed) {
        if (obj._isFill) continue;
        const kw = obj.keyword.toLowerCase();
        const isTree = NATURE_KEYWORDS.some(n => kw.includes(n));

        if (isTree) {
            // Trees, bushes and rocks: the first sets the baseline and every
            // one (the first included) lands within NATURE_SIZE_RANGE of it,
            // so a wood has younger and older trees rather than rows of equal
            // ones — while even the smallest still clearly out-tops a person.
            if (keywordScales[kw] === undefined) keywordScales[kw] = obj.scale;
            const [lo, hi] = NATURE_SIZE_RANGE;
            obj.scale = keywordScales[kw] * (lo + Math.random() * (hi - lo));
        } else {
            // Everything else: exact same size
            if (keywordScales[kw] === undefined) {
                keywordScales[kw] = obj.scale;
            } else {
                obj.scale = keywordScales[kw];
            }
        }
    }

    // Bounding box collision resolution
    // Trees can overlap slightly, everything else gets relocated
    const mapRange = isIndoor ? 250 : 500;
    const centerClear = CENTER_CLEAR[isIndoor ? 'indoor' : 'outdoor'];
    function getRadius(obj) {
        const isNature = NATURE_KEYWORDS.some(n => obj.keyword.toLowerCase().includes(n));
        return (obj.scale || 50) * (isNature ? 0.2 : 0.4);
    }
    function getVolume(obj) {
        const s = obj.scale || 50;
        return s * s * s;
    }
    function isOnPath(x, z) {
        for (const [px, , pz] of charPath) {
            const d = Math.sqrt((x - px) ** 2 + (z - (pz || 0)) ** 2);
            if (d < 100) return true;
        }
        return false;
    }
    function collidesWithAny(obj, others, skipIdx) {
        const r1 = getRadius(obj);
        for (let i = 0; i < others.length; i++) {
            if (i === skipIdx) continue;
            const other = others[i];
            const r2 = getRadius(other);
            const dx = obj.position[0] - other.position[0];
            const dz = obj.position[2] - other.position[2];
            const dist = Math.sqrt(dx * dx + dz * dz);
            const bothNature = NATURE_KEYWORDS.some(n => obj.keyword.toLowerCase().includes(n))
                && NATURE_KEYWORDS.some(n => other.keyword.toLowerCase().includes(n));
            if (bothNature) continue; // trees can overlap
            if (dist < r1 + r2) return true;
        }
        return false;
    }

    // Find collisions and relocate the smaller object
    const MAX_RELOCATE_ATTEMPTS = 15;
    if (isIndoor) for (let i = 0; i < placed.length; i++) {
        for (let j = i + 1; j < placed.length; j++) {
            const a = placed[i], b = placed[j];
            const aIsNature = NATURE_KEYWORDS.some(n => a.keyword.toLowerCase().includes(n));
            const bIsNature = NATURE_KEYWORDS.some(n => b.keyword.toLowerCase().includes(n));
            if (aIsNature && bIsNature) continue; // both nature = allow overlap

            const r1 = getRadius(a), r2 = getRadius(b);
            const dx = a.position[0] - b.position[0];
            const dz = a.position[2] - b.position[2];
            const dist = Math.sqrt(dx * dx + dz * dz);

            if (dist < r1 + r2) {
                // Relocate the smaller volume object
                const smaller = getVolume(a) < getVolume(b) ? a : b;
                const smallerIdx = smaller === a ? i : j;
                let relocated = false;

                for (let attempt = 0; attempt < MAX_RELOCATE_ATTEMPTS; attempt++) {
                    // Random position within map range
                    const nx = (Math.random() - 0.5) * 2 * mapRange;
                    const nz = (Math.random() - 0.5) * 2 * mapRange;

                    // Skip if on character path
                    if (isOnPath(nx, nz)) continue;
                    // Skip if inside the open middle
                    if (Math.sqrt(nx * nx + nz * nz) < centerClear + getRadius(smaller)) continue;

                    smaller.position[0] = nx;
                    smaller.position[2] = nz;

                    if (!collidesWithAny(smaller, placed, smallerIdx)) {
                        relocated = true;
                        break;
                    }
                }

                // If still can't find a spot, push it far out
                if (!relocated) {
                    const angle = Math.random() * Math.PI * 2;
                    smaller.position[0] = Math.cos(angle) * (mapRange - 50);
                    smaller.position[2] = Math.sin(angle) * (mapRange - 50);
                }
            }
        }
    }

    // Last word: nothing's footprint may reach into the open middle. Anything
    // the passes above left too close slides straight outward along its own
    // bearing, so the ring keeps its spread instead of bunching up.
    // (Outdoor placement already keeps its own open middle, which a forest
    // sets smaller than this — so this pass is indoor-only.)
    if (isIndoor) for (const obj of placed) {
        const minR = centerClear + getRadius(obj);
        let [x, , z] = obj.position;
        let r = Math.hypot(x, z);
        if (r >= minR) continue;
        if (r < 1) { const a = Math.random() * Math.PI * 2; x = Math.cos(a); z = Math.sin(a); r = 1; }
        obj.position[0] = x / r * minR;
        obj.position[2] = z / r * minR;
    }

    return placed;
}

async function buildScene(config) {
    sceneObjects.forEach(obj => scene.remove(obj));
    sceneObjects = [];
    grid.visible = false;
    Object.keys(modelCache).forEach(k => delete modelCache[k]);

    // Ground created after layout so we know the world extent
    const groundBase = (config.scene.ground && config.scene.ground.color) || '#888877';

    // Lights
    for (const light of (config.scene.lights || [])) {
        let l;
        // Gemini's lights set the mood, but at full strength (a grey "city"
        // ambient, a blue dusk key) they muddied the soft bright look, so they
        // come in at under half strength and washed toward white.
        const color = new THREE.Color(light.color || '#ffffff').lerp(new THREE.Color(0xffffff), 0.5);
        const intensity = (light.intensity || 1) * 0.45;
        if (light.type === 'ambient') {
            l = new THREE.AmbientLight(color, intensity);
        } else if (light.type === 'directional') {
            l = new THREE.DirectionalLight(color, intensity);
            const p = light.position || [100, 200, 100];
            l.position.set(p[0], p[1], p[2]);
        } else if (light.type === 'point') {
            l = new THREE.PointLight(color, intensity, 2000);
            const p = light.position || [0, 200, 0];
            l.position.set(p[0], p[1], p[2]);
        }
        if (l) { scene.add(l); sceneObjects.push(l); }
    }
    // A soft sky-blue from above and grass-green bounce from below lifts the
    // shadowed sides of props, for a gentler, painted outdoor look. It's a
    // scene object, so it goes away with the scene like Gemini's own lights.
    const skyFill = new THREE.HemisphereLight(0xcfe4ff, 0x9ad27a, 0.6);
    scene.add(skyFill);
    sceneObjects.push(skyFill);

    createFog = null; // set again once the ground's size is known
    setCreateLook(true);
    // Layout engine: Gemini picks objects, code positions them
    const charPath = config._characterPath || [[0, 0]];
    const sceneType = config.scene.type || 'outdoor';
    currentSceneType = sceneType;
    // A prop can ask to be placed several times ("count") when the location is
    // made of it — a forest, a street of buildings. Expanding here keeps the
    // layout engine unaware of counts: it just sees more props, and its
    // same-keyword handling (nature varies 60-120%, everything else matches) already
    // makes repeats look deliberate rather than cloned.
    const MAX_SCENE_COPIES = 14;
    let rawModels = (config.scene.models || []).flatMap(m => {
        const n = Math.max(1, Math.min(Math.round(Number(m.count) || 1), MAX_SCENE_COPIES));
        return Array.from({ length: n }, () => ({ ...m }));
    });
    // Only props the library actually has (or that are built in code, like
    // rocks) — cars, lamps and the like are dropped rather than left as empty
    // placeholders taking up room — and at most one truck. (An animation
    // Gemini gave a dropped prop just finds nothing to move and is skipped.)
    let truckKept = false;
    rawModels = rawModels.filter(m => {
        const kw = m.keyword.toLowerCase();
        const file = modelFileFor(kw);
        if (!file && !Object.keys(PROCEDURAL_KEYWORDS).some(k => kw.includes(k))) return false;
        if (file === 'models/truck.glb') {
            if (truckKept) return false;
            truckKept = true;
        }
        return true;
    });
    // A forest is trees and bushes. Gemini tends to add rocks, logs and
    // mushrooms on its own; keep those only if the user actually named them.
    const palette = PLACE_PALETTES[placeKind(rawModels)];
    if (palette && palette.only && config._prompt) {
        const said = config._prompt.toLowerCase();
        rawModels = rawModels.filter(m => {
            const kw = m.keyword.toLowerCase();
            return palette.only.some(w => kw.includes(w)) || said.includes(kw);
        });
    }
    const placedModels = computeLayout(rawModels, sceneType, charPath);

    // Size ground to the furthest object's footprint + small margin, so the
    // props ring the platform's edge. (Adding the full height here made a
    // tall tree push the edge ~400 units past where it actually stands.)
    let maxExtent = 200; // minimum
    for (const m of placedModels) {
        const dist = Math.sqrt(m.position[0] ** 2 + m.position[2] ** 2) + footprintRadius(m.keyword, m.scale || 50) + 40;
        if (dist > maxExtent) maxExtent = dist;
    }
    // Also include character path
    for (const [px, , pz] of charPath) {
        const dist = Math.sqrt(px * px + (pz || 0) * (pz || 0)) + 100;
        if (dist > maxExtent) maxExtent = dist;
    }
    // Outdoor layouts choose the platform themselves (props line its edges).
    const groundSize = placedModels.platformHalf ? placedModels.platformHalf * 2 + 20 : maxExtent * 2 + 100;
    const groundStyle = placeKind(rawModels) === 'city' ? 'town' : 'meadow';
    groundMesh = createTexturedGround(groundSize, stylizeGroundColor(groundStyle), groundStyle);
    setCreateFog(groundSize);
    groundMesh.userData._isGround = true;
    scene.add(groundMesh);
    sceneObjects.push(groundMesh);
    addGroundCover(groundSize, charPath, groundStyle);

    // Place loading placeholders at computed positions BEFORE loading models
    const placeholders = [];
    const phMat = new THREE.MeshStandardMaterial({
        color: 0x7c5cbf, transparent: true, opacity: 0.2, roughness: 1
    });
    placedModels.forEach(m => {
        if (m._isFill) return; // No placeholders for background fill
        const h = m.scale || 100;
        const phGeo = new THREE.BoxGeometry(h * 0.4, h, h * 0.4);
        const ph = new THREE.Mesh(phGeo, phMat.clone());
        ph.position.set(m.position[0], h / 2, m.position[2]);
        // Pulse animation via userData
        ph.userData._phaseOffset = Math.random() * Math.PI * 2;
        scene.add(ph);
        placeholders.push(ph);
    });

    // Animate placeholders with pulsing
    let phAnimId = null;
    function animatePlaceholders() {
        const t = Date.now() * 0.003;
        placeholders.forEach(ph => {
            if (ph.parent) { // still in scene
                ph.material.opacity = 0.15 + Math.sin(t + ph.userData._phaseOffset) * 0.1;
            }
        });
        phAnimId = requestAnimationFrame(animatePlaceholders);
    }
    if (placeholders.length > 0) animatePlaceholders();

    // Load actual models, replacing placeholders
    let phIdx = 0;
    const mainModels = placedModels.filter(m => !m._isFill);
    const fillModels = placedModels.filter(m => m._isFill);
    let loadedCount = 0;
    const totalMain = mainModels.length;

    log(`Placing models... (0/${totalMain})`, 'scene', 'model-progress');

    async function loadModel(m, myPhIdx) {
        const isFill = m._isFill;
        const procedural = tryProceduralModel(m.keyword, m.scale);
        if (procedural) {
            procedural.position.set(m.position[0], 0, m.position[2]);
            if (m.rotationY) procedural.rotation.y = m.rotationY;
            procedural.userData._keyword = m.keyword;
            procedural.userData._footprint = footprintRadius(m.keyword, m.scale);
            procedural.userData._scenery = !!placedModels.platformHalf;
            if (myPhIdx >= 0 && placeholders[myPhIdx]) scene.remove(placeholders[myPhIdx]);
            scene.add(procedural);
            sceneObjects.push(procedural);
            if (!isFill) {
                loadedCount++;
                log(`Placing models... (${loadedCount}/${totalMain}) — ${m.keyword}`, 'scene', 'model-progress');
            }
            return;
        }

        const glbUrl = await findLocalModel(m.keyword, m.category);
        if (glbUrl) {
            const loaded = await loadGLBModel(glbUrl, m.position, m.scale, m.rotationY || 0);
            if (loaded) {
                loaded.userData._keyword = m.keyword;
                loaded.userData._footprint = footprintRadius(m.keyword, m.scale);
                loaded.userData._scenery = !!placedModels.platformHalf; // edge-laid backdrop, see computeCreateFraming
            }
            if (myPhIdx >= 0 && placeholders[myPhIdx]) scene.remove(placeholders[myPhIdx]);
            if (!isFill) {
                loadedCount++;
                log(`Placing models... (${loadedCount}/${totalMain}) — ${m.keyword}`, 'scene', 'model-progress');
            }
        }
    }

    // Load main models and fill models concurrently
    let mainPhIdx = 0;
    const loadPromises = [
        ...mainModels.map(m => loadModel(m, mainPhIdx++)),
        ...fillModels.map(m => loadModel(m, -1))
    ];
    await Promise.all(loadPromises);

    log(`All ${totalMain} models placed`, 'scene', 'model-progress');

    // Clean up remaining placeholders and stop animation
    if (phAnimId) cancelAnimationFrame(phAnimId);
    placeholders.forEach(ph => { if (ph.parent) scene.remove(ph); ph.geometry.dispose(); ph.material.dispose(); });

    // Wire up any object-path animations Gemini specified (a car driving, a
    // door swinging, etc.) — matched to the placed object by keyword, offset
    // from wherever the layout engine actually put it.
    activeObjectAnimations = [];
    for (const anim of (config.scene.animations || [])) {
        const target = sceneObjects.find(o => o.userData && o.userData._keyword === anim.keyword);
        if (!target || !anim.path || anim.path.length === 0) continue;
        activeObjectAnimations.push({
            object3D: target,
            path: anim.path,
            duration: anim.path[anim.path.length - 1].t || 4,
            basePos: target.position.clone(),
            baseYaw: target.rotation.y,
        });
        log(`Animating "${anim.keyword}" along its own path`, 'scene');
    }
}

// An opening prompt whose main subject is a non-human thing ("make a car
// move into the scene") — no Kimodo call, no human character. Reuses the
// same scene-planning call and buildScene() as the person path (so the car
// still gets ground/lights/a couple of sensible extra props, placed the
// same way), then gives the named object its own entrance — Gemini's
// "animations" field is written for a small background wiggle, not a full
// arrival, so this builds that path directly: start well back from wherever
// computeLayout rested it, drive up to that spot, stop there. A follow-up
// like "a person gets out of the car" (handleAddCharacter, "near <object>"
// spawn) finds it by the same _keyword tag every other placed object uses.
async function generateObjectScene(prompt, primary, startTime) {
    setPill('pill-scene', true);
    log('Planning scene layout...', 'scene');

    const config = await callGemini(prompt, null, primary); // nobody to keep objects clear of; pin the keyword so we can find it again
    const geminiElapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    log(`Planned ${(config.scene.models || []).length} models (${geminiElapsed}s)`, 'scene');

    await buildScene(config);
    log('Scene built', 'scene');
    setPill('pill-scene', false);
    createSceneStarted = true;

    setPill('pill-render', true);
    const target = sceneObjects.find(o => o.userData._keyword === primary.keyword);
    if (target) {
        const restPos = target.position.clone();
        const APPROACH = 320; // units it arrives from
        // Random side each scene, but a STRAIGHT line once picked: yaw is 0
        // at both keyframes (interpolating 180deg -> 0deg while ALSO
        // translating made it visibly spin as it drove, not travel in a
        // line). baseYaw is set to the actual travel direction instead of
        // keeping computeLayout's arbitrary spawn rotation, so it also ends
        // up facing the way it just drove in, not some unrelated angle.
        const approachAngle = Math.random() * Math.PI * 2;
        const startOffset = [Math.sin(approachAngle) * APPROACH, 0, Math.cos(approachAngle) * APPROACH];
        const travelYaw = approachAngle + Math.PI;
        activeObjectAnimations = activeObjectAnimations.filter(a => a.object3D !== target); // drop Gemini's own small wiggle for this one — this beat replaces it
        activeObjectAnimations.push({
            object3D: target,
            path: [
                { t: 0, pos: startOffset, yaw: 0 },
                { t: 4, pos: [0, 0, 0], yaw: 0 },
            ],
            duration: 4,
            basePos: restPos,
            baseYaw: travelYaw,
        });
        // Later beats (a character spawning "near" it) need where it RESTS,
        // not wherever the mutated .position happens to read if asked about
        // mid-arrival.
        target.userData._restPosition = restPos;
        objectBeats.push({ keyword: primary.keyword, prompt, duration: 4 });
        log(`"${primary.keyword}" arrives into the scene`, 'render');
    } else {
        log(`Could not find a "${primary.keyword}" model for this scene`, 'error');
    }

    recomputePlaybackDuration();
    renderTimelineClips();
    showTimeline();
    isPlaying = true;
    playbackFinished = false;
    frameCreateStage();
    await warmUpScene();
    hideGenLoadingOverlay();
    updatePlayPauseIcon();

    log('Scene complete', 'render');
    showEditor();
    setPill('pill-render', false);
    log(`Total generation time: ${((Date.now() - startTime) / 1000).toFixed(1)}s`, 'system');
}

// Console logging
const LOG_ICONS = {
    system: `<svg width="8" height="8" viewBox="0 0 8 8"><circle cx="4" cy="4" r="2.5" fill="white"/></svg>`,
    agent:  `<svg width="8" height="8" viewBox="0 0 8 8"><rect x="1" y="1" width="6" height="6" rx="1" fill="white"/></svg>`,
    success:`<svg width="8" height="8" viewBox="0 0 8 8"><path d="M1.5 4.5L3 6 6.5 2" stroke="white" stroke-width="1.3" fill="none"/></svg>`,
    error:  `<svg width="8" height="8" viewBox="0 0 8 8"><path d="M2 2l4 4M6 2l-4 4" stroke="white" stroke-width="1.3"/></svg>`,
    motion: `<svg width="8" height="8" viewBox="0 0 8 8"><path d="M2 1v6l5-3z" fill="white"/></svg>`,
    scene:  `<svg width="8" height="8" viewBox="0 0 8 8"><path d="M1 6L4 2l3 4z" fill="white"/></svg>`,
    render: `<svg width="8" height="8" viewBox="0 0 8 8"><circle cx="4" cy="4" r="2" fill="none" stroke="white" stroke-width="1.2"/><circle cx="4" cy="4" r="0.8" fill="white"/></svg>`,
    path:   `<svg width="8" height="8" viewBox="0 0 8 8"><path d="M1 6Q4 1 7 6" stroke="white" stroke-width="1.2" fill="none"/></svg>`,
    user:   `<svg width="8" height="8" viewBox="0 0 8 8"><circle cx="4" cy="2.5" r="1.5" fill="white"/><path d="M1.5 7a2.5 2.5 0 015 0" fill="white"/></svg>`,
    music:  `<svg width="8" height="8" viewBox="0 0 8 8"><path d="M3 1v5M6 0v4.5" stroke="white" stroke-width="1.2"/><circle cx="2" cy="6" r="1.2" fill="white"/><circle cx="5" cy="5" r="1.2" fill="white"/></svg>`,
};

function log(msg, type = 'system', id = null) {
    const console_el = document.getElementById('console');

    // If an id is provided, update existing line instead of creating new one
    if (id) {
        const existing = document.getElementById('log-' + id);
        if (existing) {
            existing.querySelector('.log-msg').textContent = msg;
            console_el.scrollTop = console_el.scrollHeight;
            return existing;
        }
    }

    const line = document.createElement('div');
    line.className = `log-line log-${type}`;
    if (id) line.id = 'log-' + id;

    const icon = document.createElement('div');
    icon.className = 'log-icon';
    icon.innerHTML = LOG_ICONS[type] || LOG_ICONS.system;

    const msgEl = document.createElement('span');
    msgEl.className = 'log-msg';
    msgEl.textContent = msg;

    line.appendChild(icon);
    line.appendChild(msgEl);
    console_el.appendChild(line);
    console_el.scrollTop = console_el.scrollHeight;
    return line;
}

function setPill(id, active) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('active', active);
}

// Generate
const btn = document.getElementById('generate-btn');
const input = document.getElementById('prompt-input');

let isGenerating = false;
// Whether ANY scene exists yet — separate from currentClip, which mirrors
// characters[0] and stays null for an object-only scene ("make a car move
// into the scene", no person). handleChat uses this, not currentClip, to
// decide opening prompt vs. follow-up beat.
let createSceneStarted = false;
async function generate() {
    const prompt = input.value.trim();
    if (!prompt || isGenerating) return;
    isGenerating = true;

    const duration = 5; // Default duration for initial generation

    btn.disabled = true;
    btn.textContent = 'Generating...';
    document.getElementById('create-welcome').style.display = 'none';
    document.getElementById('panel').classList.remove('collapsed');
    document.getElementById('panel-reopen').classList.remove('visible');

    const shimmer = document.createElement('div');
    shimmer.className = 'shimmer';
    viewport.appendChild(shimmer);

    // Freeze whatever was playing and hide it behind a loading overlay —
    // the old scene shouldn't keep animating while a new one is generating.
    isPlaying = false;
    showGenLoadingOverlay('Generating...');

    log(`> "${prompt}"`, 'agent');
    const startTime = Date.now();

    try {
        // === STEP 0: is this scene about a person, or a thing moving on its
        // own? Has to happen before Kimodo runs at all — otherwise "make a
        // car move into the scene" always got a human motion generated for
        // it (Kimodo only knows how to animate people, and this used to run
        // unconditionally), and a person showed up in a scene that never
        // asked for one.
        const primary = await classifyPrimaryType(prompt);
        if (primary.primary === 'object' && primary.keyword) {
            await generateObjectScene(prompt, primary, startTime);
            shimmer.remove();
            btn.disabled = false;
            btn.textContent = 'Generate';
            isGenerating = false;
            return;
        }

        // === STEP 1: Generate motion FIRST to get character path ===
        // Silently enhance the motion prompt for better demo results.
        // Bias toward locomotion and exaggerated movement — never shown to user.
        const motionPrompt = enhanceMotionPrompt(prompt);

        setPill('pill-motion', true);
        log(`Generating motion (${duration}s)...`, 'motion');

        const MIN_TRAVEL_DIST = 50; // minimum distance between start and end
        const MAX_RETRIES = 3;
        let bvhText = null;
        let characterPath = null;

        for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
            const motionStart = Date.now();
            const { bvhText: fetchedBvh } = await fetchMotionBVH(motionPrompt, duration);
            bvhText = fetchedBvh;

            const motionElapsed = ((Date.now() - motionStart) / 1000).toFixed(1);

            // Check if character actually moved
            const { path, rawPath } = extractPathFromBVH(bvhText);
            characterPath = path;

            if (path.length >= 2) {
                const startPt = rawPath[0];
                const endPt = rawPath[rawPath.length - 1];
                const travelDist = Math.sqrt(
                    (endPt[0] - startPt[0]) ** 2 + (endPt[1] - startPt[1]) ** 2
                );

                if (travelDist >= MIN_TRAVEL_DIST || attempt === MAX_RETRIES) {
                    log(`Motion received (${(bvhText.length / 1024).toFixed(0)}KB, ${motionElapsed}s)`, 'motion');
                    break;
                }
            } else {
                break;
            }
        }

        setPill('pill-motion', false);
        log(`Extracted ${characterPath.length} waypoints`, 'path');

        // === STEP 3: Plan scene AROUND the path ===
        setPill('pill-scene', true);
        log('Planning scene layout...', 'scene');

        const config = await callGemini(prompt, characterPath);
        config._characterPath = characterPath;
        config._prompt = prompt; // lets buildScene tell props the user named from ones Gemini added
        const geminiElapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        const modelCount = (config.scene.models || []).length;
        log(`Planned ${modelCount} models (${geminiElapsed}s)`, 'scene');

        // === STEP 4: Build scene + load animation ===
        await buildScene(config);
        log('Scene built', 'scene');
        setPill('pill-scene', false);
        createSceneStarted = true;

        setPill('pill-render', true);
        log('Loading animation...', 'render');
        // Person 1 of a fresh scene: first model in the rotation.
        const firstAvatar = avatarForIndex(0);
        await ensureAvatar(firstAvatar).catch(err => console.warn('VRM avatar failed to load, using the mannequin:', err));
        loadBVH(bvhText, prompt, avatarCache[firstAvatar] ? firstAvatar : null);
        lastBvhText = bvhText;

        // Build path visualization and apply terrain
        updatePathVisualization(characterPath);
        if (groundMesh) applyTerrainFromPath(groundMesh, characterPath);
        pathVisible = false;
        if (pathLine) pathLine.visible = false;
        const pathBtn = document.getElementById('path-toggle');
        pathBtn.style.display = '';
        pathBtn.classList.remove('active');
        pathBtn.textContent = 'Show Path';

        // Initialize timeline with first clip
        // (totalDuration is set by recomputePlaybackDuration() inside loadBVH(), which
        // accounts for mixer.timeScale and any object-path animations — don't overwrite it here.)
        renderTimelineClips();
        showTimeline();
        frameCreateStage();      // point the camera at where the action actually is
        await warmUpScene();     // everything on the GPU before the first visible frame
        hideGenLoadingOverlay(); // new scene is ready — reveal + play it now
        updatePlayPauseIcon();

        log('Scene complete', 'render');
        showEditor();
        setPill('pill-render', false);

        const totalElapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        log(`Total generation time: ${totalElapsed}s`, 'system');

        // Generate soundtrack in background (non-blocking)
        // Music disabled during live playback — only plays during video render
        // generateMusic(prompt);

    } catch (err) {
        log(`Error: ${err.message}`, 'error');
        setPill('pill-scene', false);
        setPill('pill-motion', false);
        setPill('pill-render', false);
        hideGenLoadingOverlay();
    }

    shimmer.remove();
    btn.disabled = false;
    btn.textContent = 'Generate';
    isGenerating = false;
}

btn.addEventListener('click', generate);
input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); generate(); }
});

// Something is still generating or recording. It would land in a cleared stage
// when it finished (a new person standing on nothing, or the whole scene back),
// so a clear waits for it.
function createBusy() {
    return isGenerating || chatInput.disabled || isRecording;
}

// Clear Scene: Create back to how it opens — no cast, props, ground or beats,
// an empty activity log, the camera home — ready for a fresh first prompt.
// References in the Library stay; they're meant for the next scene too.
function clearCreateScene() {
    if (createBusy()) {
        log('Wait for the current step to finish, then clear the scene', 'system');
        return;
    }
    clearAllCharacters();
    for (const obj of sceneObjects) scene.remove(obj);
    if (groundMesh) scene.remove(groundMesh);
    if (pathLine) { scene.remove(pathLine); pathLine = null; }
    selectedObject = null; selectedType = null;
    removeSelectionBox();
    exitBuildMode();
    modelPicker.classList.remove('visible');
    resetToBlankScene();
    setCreateLook(true); // a modify_scene beat sets every ambient light, the stage's own included
    createSnapshot = null;
    lastBvhText = null;
    selectedClipIndex = -1;
    renderTimelineClips();
    edToolbar.classList.remove('visible');
    const pathBtn = document.getElementById('path-toggle');
    pathVisible = false;
    pathBtn.style.display = 'none';
    pathBtn.classList.remove('active');
    pathBtn.textContent = 'Show Path';
    grid.visible = true;
    camera.position.copy(HOME_VIEW.position);
    controls.target.copy(HOME_VIEW.target);
    controls.update();
    consoleEl.replaceChildren();
    reportMotionBackend();
    chatInput.value = '';
    chatInput.placeholder = 'Describe a scene, then build it up...';
    updatePlayPauseIcon();
    syncCreateWelcome();
    chatInput.focus();
}

// Two clicks when there's something to lose, since nothing brings a cleared
// scene back: the first arms the button for a few seconds, the second clears.
const clearSceneBtn = document.getElementById('clear-scene-btn');
let clearSceneDisarm = null; // pending timeout while armed
function disarmClearScene() {
    clearTimeout(clearSceneDisarm);
    clearSceneDisarm = null;
    clearSceneBtn.classList.remove('armed');
    clearSceneBtn.querySelector('span').textContent = 'Clear Scene';
}
clearSceneBtn.addEventListener('click', () => {
    if (!clearSceneDisarm && createHasContent() && !createBusy()) {
        clearSceneBtn.classList.add('armed');
        clearSceneBtn.querySelector('span').textContent = 'Confirm clear';
        clearSceneDisarm = setTimeout(disarmClearScene, 3000);
        return;
    }
    disarmClearScene();
    clearCreateScene();
});

// The logo does the same in Create, in one click as it always has (it used to
// reopen the welcome screen, which no longer exists).
document.getElementById('logo-btn').addEventListener('click', () => {
    if (document.body.dataset.mode === 'create') clearCreateScene();
});

// Render loop
const _charWorldPos = new THREE.Vector3();
let _userOrbiting = false;
let _orbitTimeout = null;
controls.addEventListener('start', () => { _userOrbiting = true; clearTimeout(_orbitTimeout); });
controls.addEventListener('end', () => { _orbitTimeout = setTimeout(() => { _userOrbiting = false; }, 2000); });
let isRecording = false;
let selectedClipIndex = -1;
let _actionLogSeq = 0;
const nextActionLogId = () => `chat-action-${++_actionLogSeq}`;
function animate() {
    requestAnimationFrame(animate);
    const dt = clock.getDelta();
    if (isRecording) return;
    if (isPlaying) {
        // characters[0]'s mixer is the same object as the legacy `mixer`
        // global (see syncPrimaryGlobals), so this covers the primary too.
        // For the Chat persona this single update advances whichever mix of
        // idle/talk actions is currently weighted in — see setPersonaTalking.
        for (const c of characters) c.mixer.update(dt);
        if (characters.length === 0 && mixer) mixer.update(dt);
        if (objectMixer) {
            objectMixer.update(dt);
            updateObjectAnimations(objectMixer.time);
        }
    }
    // Capsule bodies follow their skeletons every frame (also while paused,
    // so a scrubbed/held pose still renders correctly). A VRM-skinned
    // character carries no capsules, so this is a no-op for them and the cast
    // can freely mix the two.
    for (const c of characters) updateBodyMeshesIn(c.bodyMeshesArr);
    if (characters.length === 0) updateBodyMeshes();
    if (timelineClips.length > 0) updatePlayhead();
    // The platform stays put, centred where the scene was laid out, so the
    // props keep lining its edges. (It used to slide along under the
    // character every frame, which made props look placed around the player
    // instead of the platform.) Only if a motion carries them off the edge
    // does it follow again, so they never step into empty space.
    if (currentBones && characterGroup) {
        currentBones.getWorldPosition(_charWorldPos);
        if (groundMesh) {
            const half = (groundMesh.geometry.parameters.width || 0) / 2 - 20;
            const offPlatform = Math.abs(_charWorldPos.x) > half || Math.abs(_charWorldPos.z) > half;
            groundMesh.position.x = offPlatform ? _charWorldPos.x : 0;
            groundMesh.position.z = offPlatform ? _charWorldPos.z : 0;
        }
        // (Environment no longer grows as the character walks — see
        // AUTO_FILL_ENVIRONMENT. Scenes only contain what you asked for.)
    }
    // Pulse selection box
    if (selectionBox) {
        const pulse = 0.6 + Math.sin(Date.now() * 0.004) * 0.4;
        selectionBox.material.opacity = pulse;
    }
    // The sky is "at infinity": it travels with the camera so you can never
    // orbit out to its edge.
    if (skyGroup.visible) skyGroup.position.set(camera.position.x, 0, camera.position.z);
    renderFrame();
}
animate();

// The title and the "one instruction at a time" line are the Create overlay
// now (see #create-intro), so they aren't repeated in the activity panel.
// Report whether a motion backend is actually reachable (scenes still build
// without one, using the bundled sample motions). Clearing the scene empties
// the log, so it reports again then.
function reportMotionBackend() {
    fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) })
        .then(r => (r.ok ? r.json() : Promise.reject()))
        .then(h => log(`Motion backend ready${h.model ? ` (${h.model})` : ''}`, 'success'))
        .catch(() => log('No motion backend reachable — sample motions will be used until one is running', 'system'));
}
reportMotionBackend();


// Resize. Follows the viewport's own box rather than the window: switching
// modes changes the panel's width (half the window in Chat, a rail in Create),
// and so does collapsing it, without the window resizing at all. On the window
// event alone the canvas kept its old drawing size and CSS stretched it to the
// new box — the sideways-stretched render on coming back to Chat.
new ResizeObserver(() => {
    const w = viewport.clientWidth, h = viewport.clientHeight;
    if (!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    composer.setSize(w, h);
    // Resizing clears the canvas, and observers run after the frame's own
    // render — draw again now, or a panel slide flickers blank every frame.
    renderFrame();
}).observe(viewport);

// ========== TIMELINE & INTERACTION SYSTEM ==========
const raycaster = new THREE.Raycaster();
const ptrStart = new THREE.Vector2();
let ptrDownTime = 0;

// Track animation clips as timeline segments
// (timelineClips and totalDuration declared with other state vars above)

// Merge two AnimationClips into one continuous clip: B starts where A ended,
// and A's final pose crossfades into B's motion over `blendTime`.
//
// The previous version inserted two "blend" keyframes AFTER clip A's keys and
// BEFORE clip B's, at seam-0.001 and seam+blendTime — so every track's times
// ran backwards twice at every junction. Three.js finds keyframes by binary
// search over times it assumes are sorted, which turned the whole blend window
// into garbage and ended in a hard snap: measured on a real two-beat clip, a
// 109-unit joint jump in a single frame at seam+0.37s against a typical 5
// units per frame inside a beat. Times are strictly increasing now, and the
// crossfade is done by re-weighting B's own keys (slerp for rotations, lerp
// for positions) so the pose leaves A's last frame exactly and eases into B.
function mergeClips(clipA, clipB, blendTime = 0.4) {
    const offsetTime = clipA.duration;
    const mergedTracks = [];

    // Root position offset so clip B continues from where clip A ended. The
    // Hips track carries the absolute XZ, so B is shifted to start on A's end.
    const hipsTrackA = clipA.tracks.find(t => t.name.includes('Hips') && t.name.endsWith('.position'));
    const hipsTrackB = clipB.tracks.find(t => t.name.includes('Hips') && t.name.endsWith('.position'));
    let posOffsetX = 0, posOffsetZ = 0;
    if (hipsTrackA && hipsTrackB) {
        const vA = hipsTrackA.values, vB = hipsTrackB.values;
        posOffsetX = vA[vA.length - 3] - vB[0];
        posOffsetZ = vA[vA.length - 1] - vB[2];
    }

    const qa = new THREE.Quaternion(), qb = new THREE.Quaternion();
    for (const trackA of clipA.tracks) {
        const trackB = clipB.tracks.find(t => t.name === trackA.name);
        if (!trackB) { mergedTracks.push(trackA.clone()); continue; }

        const size = trackA.getValueSize();
        const isQuat = trackA.name.endsWith('.quaternion');
        const isHipsPos = trackA.name.endsWith('.position') && trackA.name.includes('Hips');

        const timesA = Array.from(trackA.times);
        const valuesA = Array.from(trackA.values);
        const lastA = valuesA.slice(-size);

        // B's first key has to land strictly after A's last one.
        const tLastA = timesA[timesA.length - 1];
        let shift = offsetTime;
        if (trackB.times[0] + shift <= tLastA) shift = tLastA + 1e-3 - trackB.times[0];

        const timesB = [], valuesB = [];
        for (let k = 0; k < trackB.times.length; k++) {
            const tLocal = trackB.times[k];
            const v = Array.from(trackB.values.subarray(k * size, (k + 1) * size));
            if (isHipsPos) { v[0] += posOffsetX; v[2] += posOffsetZ; }
            if (blendTime > 0 && tLocal < blendTime) {
                const w = tLocal / blendTime, e = w * w * (3 - 2 * w);   // smoothstep, 0 at the seam
                if (isQuat) { qa.fromArray(lastA); qb.fromArray(v); qa.slerp(qb, e); qa.toArray(v); }
                else for (let c = 0; c < size; c++) v[c] = lastA[c] * (1 - e) + v[c] * e;
            }
            timesB.push(tLocal + shift);
            valuesB.push(...v);
        }
        mergedTracks.push(new trackA.constructor(trackA.name,
            new Float32Array([...timesA, ...timesB]), new Float32Array([...valuesA, ...valuesB])));
    }

    // Tracks only clip B has: just shift them.
    for (const trackB of clipB.tracks) {
        if (!clipA.tracks.find(t => t.name === trackB.name)) {
            const clone = trackB.clone();
            clone.times = new Float32Array(Array.from(clone.times).map(t => t + offsetTime));
            mergedTracks.push(clone);
        }
    }

    return new THREE.AnimationClip('merged', clipA.duration + clipB.duration, mergedTracks);
}

function showTimeline() {
    document.getElementById('timeline-bar').classList.add('visible');
    createControls.classList.add('visible');
}

// One row per character, each row holding that character's own beats.
// Click a beat to select/seek it, or hit its x to cut it from the story.
function renderTimelineClips() {
    const container = document.getElementById('timeline-clips');
    container.innerHTML = '';
    const cast = characters.length > 0 ? characters : [];

    // The opening beat for an object-primary scene ("a car arrives") has no
    // character to hang a row off — give it its own. No seeking yet, but its
    // beats can be removed like any other.
    if (objectBeats.length > 0) {
        const row = document.createElement('div');
        row.className = 'timeline-row';
        if (cast.length > 0) {
            const label = document.createElement('span');
            label.className = 'timeline-row-label';
            label.textContent = objectBeats[0].keyword.slice(0, 3).toUpperCase();
            label.title = objectBeats[0].keyword;
            row.appendChild(label);
        }
        objectBeats.forEach((beat, beatIdx) => {
            const el = document.createElement('div');
            el.className = 'timeline-clip';
            const widthSeconds = beat.duration * CHAR_TIME_SCALE; // same units character rows scale by, so widths compare fairly
            el.style.width = Math.max(80, widthSeconds * 40) + 'px';
            el.innerHTML = `<span>${beat.prompt}</span><span class="clip-dur">${beat.duration.toFixed(1)}s</span>` +
                `<button class="clip-remove" title="Remove this beat">&times;</button>`;
            el.querySelector('.clip-remove').addEventListener('click', (e) => {
                e.stopPropagation();
                removeObjectBeat(beatIdx);
            });
            row.appendChild(el);
        });
        container.appendChild(row);
    }

    cast.forEach((entry, charIdx) => {
        const row = document.createElement('div');
        row.className = 'timeline-row';

        if (cast.length > 1) {
            const label = document.createElement('span');
            label.className = 'timeline-row-label';
            label.textContent = `P${charIdx + 1}`;
            label.title = entry.label;
            row.appendChild(label);
        }

        entry.clips.forEach((seg, clipIdx) => {
            const el = document.createElement('div');
            el.className = 'timeline-clip';
            if (charIdx === 0 && clipIdx === selectedClipIndex) el.classList.add('selected');
            el.style.width = Math.max(80, seg.duration * 40) + 'px';
            // Every beat can be removed, a character's last one included
            // (that takes the character out — see removeTimelineClip).
            el.innerHTML = `<span>${seg.prompt || 'motion'}</span><span class="clip-dur">${seg.duration.toFixed(1)}s</span>` +
                `<button class="clip-remove" title="Remove this beat">&times;</button>`;

            el.addEventListener('click', (e) => {
                e.stopPropagation();
                if (charIdx === 0) {
                    selectedClipIndex = (selectedClipIndex === clipIdx) ? -1 : clipIdx;
                    renderTimelineClips();
                }
                // Seek the whole scene to the start of this beat. Seeking only
                // this character left the rest of the cast where they were, out
                // of step from then on. Beat durations are in clip time.
                let t = 0;
                for (let j = 0; j < clipIdx; j++) t += entry.clips[j].duration;
                seekPlayback(t / CHAR_TIME_SCALE);
            });

            const removeBtn = el.querySelector('.clip-remove');
            if (removeBtn) {
                removeBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    removeTimelineClip(charIdx, clipIdx);
                });
            }
            row.appendChild(el);
        });

        container.appendChild(row);
    });
}

// Cut one beat out of a character's story and rebuild their animation.
function removeTimelineClip(charIdx, clipIdx) {
    const entry = characters[charIdx];
    if (!entry) return;
    // A character's last beat is the character: removing it takes them out.
    if (entry.clips.length <= 1) return removeCharacter(charIdx);
    const [removed] = entry.clips.splice(clipIdx, 1);
    selectedClipIndex = -1;
    rebuildCharacterClip(entry);
    recomputePlaybackDuration();
    renderTimelineClips();
    restartPlayback();
    // handleAddMotion re-frames after every edit; this path didn't, so a
    // deletion that shortens someone's path (or leaves a former fight
    // partner standing wherever their OWN beat now ends, no longer where
    // the deleted beat had them) could leave the camera pointed at empty
    // ground or two characters overlapping — replay would restart the
    // animation correctly, but with nothing visibly happening where you're
    // looking, which reads as "the replay button doesn't work".
    frameCreateStage();
    log(`Removed "${removed.prompt || 'motion'}" from ${entry.label}`, 'system');
}

// Take a character (their last remaining beat) off the stage.
function removeCharacter(charIdx) {
    const [entry] = characters.splice(charIdx, 1);
    if (!entry) return;
    entry.mixer.stopAllAction();
    scene.remove(entry.group);
    entry.bodyMeshesArr.forEach(m => scene.remove(m));
    selectedClipIndex = -1;
    syncPrimaryGlobals();
    log(`Removed ${entry.label}`, 'system');
    finishTimelineEdit();
}

// Remove an object's beat ("a car arrives"): its arrival stops, and the
// prop stays where it rests.
function removeObjectBeat(beatIdx) {
    const [beat] = objectBeats.splice(beatIdx, 1);
    if (!beat) return;
    activeObjectAnimations = activeObjectAnimations.filter(a => {
        if (a.object3D.userData._keyword !== beat.keyword) return true;
        a.object3D.position.copy(a.object3D.userData._restPosition || a.basePos);
        return false;
    });
    log(`Removed "${beat.prompt}"`, 'system');
    finishTimelineEdit();
}

// After a beat or a whole character is removed: replay what's left, or, if
// the timeline is now empty, go back to "describe a scene" — the props stay
// on stage, and the next prompt builds a fresh scene rather than directing a
// cast that no longer exists.
function finishTimelineEdit() {
    if (characters.length === 0 && objectBeats.length === 0) {
        totalDuration = 0;
        isPlaying = false;
        playbackFinished = false;
        createSceneStarted = false;
        renderTimelineClips();
        document.getElementById('timeline-bar')?.classList.remove('visible');
        syncCreateWelcome();
        updatePlayPauseIcon();
        return;
    }
    recomputePlaybackDuration();
    renderTimelineClips();
    restartPlayback();
    frameCreateStage();
}

function updatePlayhead() {
    // An object-primary scene has totalDuration > 0 with zero characters —
    // this used to bail out on that case entirely, so the playhead and
    // clock never moved for a scene whose only beat is "a car arrives".
    if (totalDuration === 0) return;
    // Plays once then holds at the end (no more auto-loop) — see createPlaybackClock().
    const clockTime = objectMixer ? objectMixer.time : (mixer ? mixer.time : 0);
    const t = Math.min(clockTime, totalDuration);

    const clipsEl = document.getElementById('timeline-clips');
    const controlsEl = document.getElementById('timeline-controls');
    // Width of the widest character row — the playhead spans the whole story.
    let rowWidth = 0;
    clipsEl.querySelectorAll('.timeline-row').forEach(row => {
        let w = 0;
        row.querySelectorAll('.timeline-clip').forEach(el => { w += el.offsetWidth + 4; });
        if (w > rowWidth) rowWidth = w;
    });
    rowWidth = Math.max(rowWidth, 1);
    const offset = controlsEl.offsetWidth + 16;
    const frac = Math.min(t / totalDuration, 1.0);
    document.getElementById('timeline-playhead').style.left = (offset + frac * rowWidth) + 'px';
    document.getElementById('tl-time').textContent = t.toFixed(1) + 's';

    // Highlight whichever beat each character is currently playing. The
    // object-beat row, if there is one, is always first and isn't a
    // character — skip it when matching rows to characters[], or every
    // character's highlighting was off by one row (and the last character
    // never got highlighted at all).
    const charRowOffset = objectBeats.length > 0 ? 1 : 0;
    clipsEl.querySelectorAll('.timeline-row').forEach((row, rowIdx) => {
        const entry = characters[rowIdx - charRowOffset];
        if (!entry) return;
        const charTime = entry.mixer.time;
        let elapsed = 0;
        row.querySelectorAll('.timeline-clip').forEach((el, i) => {
            const seg = entry.clips[i];
            if (!seg) return;
            el.classList.toggle('active', charTime >= elapsed && charTime < elapsed + seg.duration);
            elapsed += seg.duration;
        });
    });
}

// Click to select/delete scene objects (silent — no panel)
renderer.domElement.addEventListener('pointerdown', (e) => {
    ptrStart.set(e.clientX, e.clientY);
    ptrDownTime = Date.now();
});

renderer.domElement.addEventListener('pointerup', (e) => {
    const dx = e.clientX - ptrStart.x, dy = e.clientY - ptrStart.y;
    if (Math.sqrt(dx*dx + dy*dy) > 5 || Date.now() - ptrDownTime > 300) return;

    const rect = renderer.domElement.getBoundingClientRect();
    const ptr = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    raycaster.setFromCamera(ptr, camera);

    const targets = [];
    sceneObjects.forEach(obj => {
        if (obj.userData._isGround) return; // skip ground
        if (obj.isMesh) targets.push(obj);
        else if (obj.isGroup || obj.children) obj.traverse(c => { if (c.isMesh) targets.push(c); });
    });

    const hits = raycaster.intersectObjects(targets, false);
    if (hits.length > 0) {
        let root = hits[0].object;
        while (root.parent && !sceneObjects.includes(root)) root = root.parent;
        if (sceneObjects.includes(root)) {
            // Clear previous selection
            if (selectedObject) {
                selectedObject.traverse(c => {
                    if (c.isMesh && c.userData._oe) c.material.emissive.copy(c.userData._oe);
                });
            }
            selectedObject = root;
            selectedType = 'scene';
            // Purple emissive tint
            root.traverse(c => {
                if (c.isMesh && c.material && c.material.emissive) {
                    c.userData._oe = c.material.emissive.clone();
                    c.material.emissive.set(0x553399);
                }
            });
            // Purple wireframe box
            showSelectionBox(root);
            enterBuildMode();
        }
    } else {
        // Deselect
        if (selectedObject) {
            selectedObject.traverse(c => {
                if (c.isMesh && c.userData._oe) c.material.emissive.copy(c.userData._oe);
            });
            selectedObject = null;
            selectedType = null;
            removeSelectionBox();
            exitBuildMode();
        }
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

    // Spacebar = play/pause (or replay, if the clip already finished)
    if (e.key === ' ') {
        e.preventDefault();
        togglePlayPause();
        return;
    }

    if (e.key !== 'Delete' && e.key !== 'Backspace') return;

    // Delete selected timeline beat (primary character)
    if (selectedClipIndex >= 0 && characters[0] && characters[0].clips.length > 1) {
        removeTimelineClip(0, selectedClipIndex);
        return;
    }

    // Delete selected scene object
    if (selectedObject && selectedType === 'scene') {
        scene.remove(selectedObject);
        const idx = sceneObjects.indexOf(selectedObject);
        if (idx !== -1) sceneObjects.splice(idx, 1);
        selectedObject.traverse(c => {
            if (c.geometry) c.geometry.dispose();
            if (c.material) (Array.isArray(c.material) ? c.material : [c.material]).forEach(m => m.dispose());
        });
        selectedObject = null;
        selectedType = null;
        removeSelectionBox();
    }
});

// Play/Pause button — replays from the start if the clip already finished
function togglePlayPause() {
    if (playbackFinished) {
        restartPlayback();
        return;
    }
    isPlaying = !isPlaying;
    updatePlayPauseIcon();
}
document.getElementById('tl-playpause').addEventListener('click', togglePlayPause);

// Deselect timeline clip when clicking outside
document.addEventListener('click', (e) => {
    if (!e.target.closest('.timeline-clip')) {
        if (selectedClipIndex >= 0) {
            selectedClipIndex = -1;
            renderTimelineClips();
        }
    }
});

// ========== CHAT INTERFACE ==========
const chatInput = document.getElementById('chat-input');

// Find nearest scene object matching a keyword
function findNearestObject(keyword) {
    const kw = keyword.toLowerCase();
    let best = null, bestDist = Infinity;
    // Get character's current position
    const charPos = new THREE.Vector3();
    if (currentBones) currentBones.getWorldPosition(charPos);
    for (const obj of sceneObjects) {
        if (obj.userData._isGround) continue;
        const objKw = (obj.userData._keyword || '').toLowerCase();
        if (!objKw) continue; // no keyword to match against - an empty string is a substring of everything, so leaving this in would match any search term
        if (!(objKw.includes(kw) || kw.includes(objKw))) continue;
        const d = charPos.distanceTo(obj.position);
        if (d < bestDist) { bestDist = d; best = obj; }
    }
    return best;
}

// Rotate a BVH clip's root motion to face a target angle
function rotateBVHToward(bvhText, targetAngle) {
    const loader = new BVHLoader();
    const result = loader.parse(bvhText);
    const clip = result.clip;
    const hipsTrack = clip.tracks.find(t => t.name.includes('Hips') && t.name.endsWith('.position'));
    if (!hipsTrack) return { clip, skeleton: result.skeleton };

    const v = hipsTrack.values;
    // Compute the current direction of travel from BVH
    const startX = v[0], startZ = v[2];
    const endX = v[v.length - 3], endZ = v[v.length - 1];
    const currentAngle = Math.atan2(endX - startX, endZ - startZ);
    const rotation = targetAngle - currentAngle;
    const cos = Math.cos(rotation), sin = Math.sin(rotation);

    // Rotate all Hips positions around the start point
    for (let i = 0; i < v.length; i += 3) {
        const dx = v[i] - startX;
        const dz = v[i + 2] - startZ;
        v[i] = startX + dx * cos - dz * sin;
        v[i + 2] = startZ + dx * sin + dz * cos;
    }
    return { clip, skeleton: result.skeleton };
}

// Classify user chat intent via Gemini
const CHAT_CLASSIFY_PROMPT = `You are the director of a 3D scene that the user is building up one instruction at a time (e.g. "a person walks to the middle" → "add another person walking in from the other side" → "they fight"). Classify the user's next instruction into ONE action and extract parameters. Output ONLY valid JSON (no markdown).

ACTION TYPES:
1. "add_motion" — existing people in the scene should do something next. This APPENDS to their timelines, continuing the story.
   Output: {"action":"add_motion","motions":[{"target":0,"prompt":"A person ..."}],"duration":5,"target_object":null}
   - "target" is the index of the character from CURRENT CAST below.
   - If the instruction involves EVERYONE ("they fight", "they hug", "both start running"), emit ONE entry per character in the cast, each with its own prompt describing only THAT person's body action (e.g. index 0 "A person throws a punch and steps forward", index 1 "A person blocks and staggers backward"). The motions are generated independently, so never describe contact with the other person.
   - If they reference a scene object ("run towards the tree"), set target_object to that object name.

2. "add_character" — the user wants ANOTHER person in the scene ("add a second person", "someone else walks in", "a person gets out of the car").
   Output: {"action":"add_character","motion_prompt":"A person ...","spawn":"opposite"|"left"|"right"|"behind"|"near <object>","duration":5}
   - motion_prompt MUST start with "A person" and describe only what this new person does.
   - "spawn" is where they start relative to the existing cast; use "opposite" for someone approaching from the other side.
   - If they're emerging from or getting out of a specific scene object (a car, a door), use "near <object>" naming it exactly as it appears in OBJECTS IN SCENE below, instead of opposite/left/right/behind.

3. "add_object" — user wants to add a prop/object to the scene
   Output: {"action":"add_object","keyword":"search term","category":number,"size":"large"|"medium"|"small","direction":"left"|"right"|"front"|"behind"|"near <object>"|"random","count":number}
   CATEGORIES: 0=Food, 1=Clutter, 3=Transport, 4=Furniture, 5=Objects, 6=Nature, 7=Animals, 8=Buildings, 11=Other
   If user says where to place it, use that direction. Otherwise "random".
   - "count" is how many to place. A number in the instruction ("10 trees") is that number exactly.
     A plural with no number ("add trees", "trees around the scene", "some rocks") means several — use
     5. A singular ("a tree", "add a rock") means 1. Cap at 20 even if asked for more.
   - "size" is real-world scale, not visual importance — a tree or a building is "large" (both dwarf a
     person) even when the user only wants one of them and isn't asking for it to stand out. Anything
     roughly human-scale or smaller (furniture, signs, a bike, a bench) is "medium". Ground-level clutter
     (a rock, a flower, a trash can) is "small".

4. "modify_scene" — user wants to change lighting, ground color, time of day, weather
   Output: {"action":"modify_scene","changes":{"ground_color":"#hex","ambient_intensity":number,"fog":bool}}

Respond with ONLY the JSON object.`;

// A short description of what's currently in the scene, so the director can
// target the right people ("they fight" needs to know how many there are).
function describeSceneForDirector() {
    const cast = characters.map((c, i) => {
        const last = c.clips[c.clips.length - 1];
        return `  [${i}] ${c.label} — last did: "${last ? last.prompt || 'unspecified motion' : 'nothing'}"`;
    }).join('\n') || '  (nobody yet)';
    const props = [...new Set(sceneObjects
        .map(o => o.userData && o.userData._keyword)
        .filter(Boolean))].slice(0, 12).join(', ') || '(none)';
    return `\n\nCURRENT CAST:\n${cast}\n\nOBJECTS IN SCENE: ${props}`;
}

async function classifyChat(userMsg) {
    const data = await fetchGeminiWithRetry(GEMINI_URL, {
        contents: [{ role: 'user', parts: [{ text: CHAT_CLASSIFY_PROMPT + describeSceneForDirector() + '\n\nUser says: ' + userMsg }] }],
        generationConfig: jsonGenConfig(0.2, 800)
    });
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!text) throw new Error(geminiErrorMessage(data));
    const clean = text.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
    return JSON.parse(clean);
}

const GOLDEN_ANGLE = 2.399963; // radians — same spiral computeLayout uses for the opening scene

// Where added objects go: a golden-angle spiral around wherever the cast (or
// a "near X" target) currently is, radius scaled to the object's own
// footprint so buildings start further out and space out more than rocks
// do. Rejection-samples against every object and character already on
// stage — not just its own siblings — so "add 10 trees" reads as placed on
// purpose (spread, not overlapping the cast or each other) instead of
// dropped at independent random points, which is what made repeated adds
// look scattered rather than designed.
function planObjectPositions(count, direction, footprint) {
    const charPos = new THREE.Vector3();
    if (currentBones) currentBones.getWorldPosition(charPos);
    let cx = charPos.x, cz = charPos.z;

    let sectorCenter = null;
    if (direction && direction.startsWith('near ')) {
        const target = findNearestObject(direction.replace('near ', ''));
        if (target) { cx = target.position.x; cz = target.position.z; }
    } else {
        const sectors = { left: Math.PI, right: 0, front: Math.PI / 2, behind: -Math.PI / 2 };
        sectorCenter = sectors[direction] ?? null;
    }

    const obstacles = sceneObjects
        .filter(o => !o.userData._isGround)
        .map(o => ({ x: o.position.x, z: o.position.z, r: o.userData._footprint || 60 }));
    for (const c of characters) {
        const p = new THREE.Vector3();
        c.group.getWorldPosition(p);
        obstacles.push({ x: p.x, z: p.z, r: 70 });
    }

    const spacing = Math.max(50, footprint * 0.8);
    // Clustering "near" another prop can hug it; anything else starts past
    // the open middle so the character keeps room to move.
    const nearTarget = cx !== charPos.x || cz !== charPos.z;
    const innerClear = nearTarget ? 150 : CENTER_CLEAR[currentSceneType === 'indoor' ? 'indoor' : 'outdoor'];
    const baseRadius = Math.max(innerClear, footprint * 1.2);

    // Outdoors, a plain or sided add ("add trees", "add a rock on the left")
    // lines the platform's edge band like the opening scene did, rather
    // than spiralling out from wherever the character happens to stand.
    if (!nearTarget && groundMesh && currentSceneType !== 'indoor') {
        const H = groundMesh.geometry.parameters.width / 2;
        const sideFor = { 0: 1, [Math.PI]: 3 }; // right → +X edge, left → -X edge
        const edgeBatch = [];
        for (let i = 0; i < count; i++) {
            let x = 0, z = 0;
            for (let attempt = 0; attempt < 30; attempt++) {
                // Same edge-weighted depth as the opening layout: dense at the rim, thinning inward.
                const inset = H - footprint - Math.pow(Math.random(), 2.2) * H * 0.6;
                const along = (Math.random() * 2 - 1) * (H - footprint);
                // 0: behind (-Z), 1: right (+X), 2: front (+Z), 3: left (-X)
                const side = sectorCenter === null ? Math.floor(Math.random() * 4)
                    : sectorCenter === Math.PI / 2 ? 2
                    : sectorCenter === -Math.PI / 2 ? 0
                    : sideFor[sectorCenter];
                [x, z] = side === 0 ? [along, -inset] : side === 1 ? [inset, along] : side === 2 ? [along, inset] : [-inset, along];
                const clear = !obstacles.some(o => Math.hypot(x - o.x, z - o.z) < o.r + footprint * 0.4)
                    && !edgeBatch.some(([px, pz]) => Math.hypot(x - px, z - pz) < spacing);
                if (clear) break;
            }
            edgeBatch.push([x, z]);
        }
        return edgeBatch;
    }
    // Vogel/sunflower spiral: r_i = c * sqrt(i) spreads N points with roughly
    // even, constant nearest-neighbor spacing REGARDLESS of N — the area of
    // a disc grows with r², so compensating with sqrt(i) keeps density (and
    // so spacing) uniform as points are added, instead of a linear r_i = i
    // step needing to be re-tuned per count. A first attempt scaled the
    // outer radius by count*footprint directly and overshot badly — 8
    // buildings ended up 739 units apart, an "isolated houses" spread, not
    // a city block. c = spacing/1.6 keeps adjacent rings roughly `spacing`
    // apart (derived from equal-area packing, tuned against the golden-angle
    // jitter actually used below).
    const spiralScale = spacing / 1.6;
    const spiralOffset = Math.random() * Math.PI * 2; // repeat adds don't overlay the exact same spiral

    const placedThisBatch = [];
    for (let i = 0; i < count; i++) {
        let x = cx, z = cz, ok = false;
        // On repeated failures, push outward as well as varying the angle —
        // a saturated inner ring needs more room, not just a different spot
        // on the same ring.
        for (let attempt = 0; attempt < 24 && !ok; attempt++) {
            const angle = sectorCenter !== null
                ? sectorCenter + (Math.random() - 0.5) * (Math.PI * 0.7) // stay in a ~125° wedge on that side
                : spiralOffset + i * GOLDEN_ANGLE + attempt * 0.7;
            const growth = 1 + Math.floor(attempt / 6) * 0.3;
            const r = (baseRadius + spiralScale * Math.sqrt(i + 1)) * growth;
            x = cx + Math.cos(angle) * r;
            z = cz + Math.sin(angle) * r;
            ok = !obstacles.some(o => Math.hypot(x - o.x, z - o.z) < o.r + footprint * 0.4)
              && !placedThisBatch.some(([px, pz]) => Math.hypot(x - px, z - pz) < spacing);
        }
        placedThisBatch.push([x, z]);
    }
    return placedThisBatch;
}

// Handle add_object action — "count" places several, sized and spread the
// same way the opening scene's own objects are (see SIZE_SCALES /
// computeLayout), instead of a separate, smaller ad-hoc scale table and
// independent random placement.
const LARGE_SCALE_KEYWORDS = [...TREE_KEYWORDS, 'building', 'house', 'tower', 'skyscraper', 'castle', 'windmill', 'apartment', 'warehouse', 'church', 'hotel'];

async function handleAddObject(params, actionId = nextActionLogId()) {
    const count = Math.max(1, Math.min(20, Math.round(params.count) || 1));
    const scales = SIZE_SCALES[currentSceneType === 'indoor' ? 'indoor' : 'outdoor'];
    // Trust Gemini's size when it gives one; if it doesn't, a tree or a
    // building should still dwarf a person rather than defaulting to
    // medium — that default was exactly how a "building" ended up shorter
    // than the character standing next to it.
    const fallbackSize = LARGE_SCALE_KEYWORDS.some(k => params.keyword.toLowerCase().includes(k)) ? 'large' : 'medium';
    const targetScale = (propShape(params.keyword) || {}).height || scales[params.size] || scales[fallbackSize];
    const footprint = footprintRadius(params.keyword, targetScale);

    log(`Adding ${count > 1 ? count + '× ' : ''}"${params.keyword}" (${params.direction || 'random'})...`, 'scene', actionId);

    const positions = planObjectPositions(count, params.direction, footprint);
    let placed = 0;
    for (const [x, z] of positions) {
        const rotY = Math.random() * Math.PI * 2;

        const procedural = tryProceduralModel(params.keyword, targetScale);
        if (procedural) {
            procedural.position.set(x, 0, z);
            procedural.rotation.y = rotY;
            procedural.userData._keyword = params.keyword;
            procedural.userData._footprint = footprint;
            scene.add(procedural);
            sceneObjects.push(procedural);
            placed++;
            continue;
        }

        const glbUrl = await findLocalModel(params.keyword, params.category || 11);
        if (glbUrl) {
            const loaded = await loadGLBModel(glbUrl, [x, 0, z], targetScale, rotY);
            if (loaded) { loaded.userData._keyword = params.keyword; loaded.userData._footprint = footprint; placed++; }
        } else {
            break; // no model for this keyword — retrying won't find one either
        }
    }

    if (placed > 0) {
        const where = (!params.direction || params.direction === 'random') ? 'around the scene' : params.direction;
        log(`Placed ${placed}× "${params.keyword}" ${where}`, 'success', actionId);
    } else {
        log(`Could not find "${params.keyword}" model`, 'error', actionId);
    }
}

// Handle add_motion — appends a new motion onto one or more characters'
// own timelines, so the story continues from where each of them left off.
async function handleAddMotion(params, actionId = nextActionLogId()) {
    const dur = params.duration || 5;

    // Normalise into [{ entry, prompt }] — supports the multi-character form
    // ({motions:[{target,prompt}]}) and the older single-prompt shape.
    let requests = [];
    if (Array.isArray(params.motions) && params.motions.length > 0) {
        for (const m of params.motions) {
            if (m.target === 'all' || m.target === undefined) {
                characters.forEach(entry => requests.push({ entry, prompt: m.prompt }));
            } else {
                const entry = characters[Number(m.target)];
                if (entry) requests.push({ entry, prompt: m.prompt });
            }
        }
    } else if (params.prompt && characters[0]) {
        requests.push({ entry: characters[0], prompt: params.prompt });
    }
    if (requests.length === 0) {
        log('No character to apply that motion to', 'error', actionId);
        return;
    }

    // Only the single-character case rotates toward a scene object — with
    // several people moving at once there's no one sensible facing to pick.
    let targetAngle = null;
    if (params.target_object && requests.length === 1) {
        const target = findNearestObject(params.target_object);
        if (target) {
            const charPos = new THREE.Vector3();
            requests[0].entry.bones.getWorldPosition(charPos);
            targetAngle = Math.atan2(target.position.x - charPos.x, target.position.z - charPos.z);
            log(`Targeting nearest "${params.target_object}"`, 'path');
        } else {
            log(`No "${params.target_object}" found in scene, generating motion anyway`, 'system');
        }
    }

    log(`Generating ${requests.length} motion${requests.length > 1 ? 's' : ''} (${dur}s)...`, 'motion', actionId);
    setPill('pill-motion', true);

    // An interaction ("they fight", "they shake hands") is two or more
    // people in the same beat.
    const interaction = requests.length > 1;

    try {
        // All characters' motions are generated in parallel so adding a beat
        // to the whole cast isn't N times slower than adding one.
        //
        // enhanceMotionPrompt is skipped for an interaction. It exists to
        // make a SOLO demo look lively — it unconditionally appends "make
        // all movements large, exaggerated, and continuous... travel
        // forward through space, not stay in place", and even inserts that
        // instruction itself when it doesn't spot a movement verb. Telling
        // BOTH participants of a handshake to individually "travel forward,
        // not stay in place" is exactly what sent them walking through each
        // other — measured the plain, un-enhanced prompt for a handshake at
        // a peak travel of 5-7 units, essentially stationary, which is what
        // an interaction like that should look like. The classifier already
        // writes specific per-person action prompts for a shared beat (see
        // CHAT_CLASSIFY_PROMPT), so they don't need generic verb inflation.
        const bvhs = await Promise.all(requests.map(r =>
            fetchMotionBVH(interaction ? r.prompt : enhanceMotionPrompt(r.prompt), dur).then(res => res.bvhText)
        ));

        // Their motions are still generated independently — nothing in the
        // raw BVH knows the other person exists, so each one plays out in
        // whatever direction Kimodo baked. Before appending, turn each of
        // them to face where the others actually are, and cap how far the
        // beat can carry them so they stay within arm's reach of each other.
        const endPoints = requests.map(r => characterEndState(r.entry).point);

        // Strip Kimodo's dead padding and fill the beat with the action.
        const newClips = requests.map((r, i) => fillBeat((targetAngle !== null)
            ? rotateBVHToward(bvhs[i], targetAngle).clip
            : new BVHLoader().parse(bvhs[i]).clip, dur));

        // A shared beat is one moment in the story, so it starts and ends at
        // the same time for everyone in it. It goes on the end of each
        // person's own timeline, and those needn't line up (someone joined
        // later, or had a beat the others didn't), which played the "same"
        // fight seconds apart; fillBeat can also leave the clips a little
        // different in length. Whoever would be early holds their pose.
        const beatStart = Math.max(...requests.map(r => r.entry.mergedClip.duration));
        if (interaction) {
            const beatLength = Math.max(...newClips.map(c => c.duration));
            requests.forEach((r, i) => {
                const lag = beatStart - r.entry.mergedClip.duration;
                if (lag > BEAT_SYNC_TOLERANCE) appendMotionToCharacter(r.entry, holdEndPoseClip(r.entry.mergedClip, lag), '(waiting)');
                const shortfall = beatLength - newClips[i].duration;
                if (shortfall > BEAT_SYNC_TOLERANCE) newClips[i] = mergeClips(newClips[i], holdEndPoseClip(newClips[i], shortfall), 0.15);
            });
        }
        // Where the new beat begins on the shared clock (a moment before, as
        // a lead-in) — playback jumps there once it's appended.
        const beatStartWall = beatStart / CHAR_TIME_SCALE - 0.25;

        requests.forEach((r, i) => {
            const newClip = newClips[i];

            // Every fresh Kimodo generation is baked facing its own arbitrary
            // "forward" — with no correction a new beat visibly snapped back
            // to roughly that direction regardless of which way the character
            // actually ended the previous beat facing, which is what made
            // beat-to-beat transitions look disconnected. yawClipToFace turns
            // the WHOLE clip (travel + body) to start at a chosen world angle:
            // toward the other person, recomputed fresh each beat so a fight
            // doesn't drift into swinging at empty air; otherwise just
            // wherever they already are, for plain continuity. The
            // target_object case (targetAngle set) already points the walk
            // itself via rotateBVHToward — leave that path alone.
            let travelCap;
            if (interaction) {
                const focus = new THREE.Vector3();
                endPoints.forEach((p, j) => { if (j !== i) focus.add(p); });
                focus.divideScalar(requests.length - 1);
                const me = endPoints[i];
                const desired = Math.atan2(focus.x - me.x, focus.z - me.z);
                yawClipToFace(r.entry, newClip, desired);

                let nearestGap = Infinity;
                endPoints.forEach((p, j) => { if (j !== i) nearestGap = Math.min(nearestGap, me.distanceTo(p)); });
                travelCap = interactionTravelCap(nearestGap);
            } else if (targetAngle === null) {
                yawClipToFace(r.entry, newClip, characterEndFacing(r.entry));
            }

            appendMotionToCharacter(r.entry, newClip, r.prompt, travelCap, interaction);
            log(`${r.entry.label}: "${r.prompt}" (+${newClip.duration.toFixed(1)}s)`, 'success', actionId);
        });

        recomputePlaybackDuration();
        renderTimelineClips();
        playFrom(beatStartWall); // show the new beat; Replay plays the whole story

        // The debug path line keeps growing with the story. Terrain shape
        // does not: it's set once from the OPENING prompt (in generate()),
        // because a follow-up beat is an action — waving, fighting, dying,
        // sitting — not a redescription of the ground. Re-running the
        // terrain sculptor on every beat used to read a "die" beat's Hips-Y
        // drop as a hillside and carve a crater under wherever the character
        // fell; measured on a real collapse, the drop is ~85 units over
        // ~135 units of incidental stagger-forward travel, which isn't even
        // steep enough for a slope-based check to catch reliably (its worst
        // single-step ratio was under half of a generous walkable incline) —
        // simply not re-deforming after the first beat sidesteps the
        // ambiguity entirely instead of trying to out-guess it.
        if (characters[0]) {
            updatePathVisualization(extractPathFromClip(characters[0].mergedClip));
        }
    } catch (err) {
        log(`Error: ${err.message}`, 'error', actionId);
    }
    setPill('pill-motion', false);
}

// Handle add_character — brings another person into the existing scene,
// standing near the current cast and facing them.
async function handleAddCharacter(params, actionId = nextActionLogId()) {
    const dur = params.duration || 5;
    log('Bringing in another person...', 'motion', actionId);
    setPill('pill-motion', true);
    try {
        const { bvhText } = await fetchMotionBVH(enhanceMotionPrompt(params.motion_prompt), dur);
        const previewClip = normalizeRootTravel(new BVHLoader().parse(bvhText).clip);

        // "near <object>" only means something when there's no cast yet to
        // spawn relative to instead — an object-primary opening scene never
        // created a person, so this is how "a person gets out of the car"
        // finds the car. If that object is still mid-arrival (or hasn't
        // started), delaySeconds holds this character still until it's
        // actually done — otherwise "getting out of the car" started
        // walking at t=0, the same instant the car itself starts arriving.
        let spawn = null, delaySeconds = 0;
        if (characters.length === 0 && typeof params.spawn === 'string' && params.spawn.startsWith('near ')) {
            const objectKeyword = params.spawn.slice(5);
            spawn = pickSpawnNearObject(objectKeyword);
            if (spawn) {
                const target = findNearestObject(objectKeyword);
                const anim = target && activeObjectAnimations.find(a => a.object3D === target);
                if (anim) delaySeconds = anim.duration;
            }
        }
        if (!spawn) spawn = pickSpawnForNewCharacter(params.spawn, previewClip);

        // Each newcomer takes the next model in the rotation, so a crowd isn't
        // one face repeated.
        const avatar = avatarForIndex(characters.length);
        await ensureAvatar(avatar).catch(err => console.warn('VRM avatar failed to load, using the mannequin:', err));
        const entry = createCharacter(bvhText, {
            position: spawn.position,
            yaw: spawn.yaw,
            label: `Person ${characters.length + 1}`,
            prompt: params.motion_prompt,
            avatar: avatarCache[avatar] ? avatar : null,
        });
        if (delaySeconds > 0) prependHoldToCharacter(entry, delaySeconds);
        recomputePlaybackDuration();
        renderTimelineClips();
        playFrom(delaySeconds - 0.25); // from where the newcomer starts moving; Replay plays the whole story
        log(`${entry.label} joined the scene (${entry.mergedClip.duration.toFixed(1)}s)`, 'success', actionId);
    } catch (err) {
        log(`Error: ${err.message}`, 'error', actionId);
    }
    setPill('pill-motion', false);
}

// Handle modify_scene action
function handleModifyScene(params) {
    const changes = params.changes || {};
    // Ground colour is fixed: every Create scene stands on the same grass
    // (see CREATE_GROUND_COLOR), so a "make the ground sandy" edit is ignored.
    if (changes.ambient_intensity !== undefined) {
        scene.traverse(obj => {
            if (obj.isAmbientLight) obj.intensity = changes.ambient_intensity;
        });
        log(`Ambient light set to ${changes.ambient_intensity}`, 'scene');
    }
    log('Scene updated', 'success');
}

// Main chat handler
async function handleChat(userMsg) {
    if (!userMsg.trim()) return;

    // With the welcome screen gone this bar is the only way into Create, so an
    // empty stage means this is the opening prompt: build the scene. Every
    // prompt after that is a beat, routed through the director.
    if (!createSceneStarted) {
        input.value = userMsg;
        await generate();
        return;
    }

    chatInput.disabled = true;
    chatInput.placeholder = 'Processing...';
    log(`> ${userMsg}`, 'user');

    try {
        const intent = await classifyChat(userMsg);

        switch (intent.action) {
            case 'add_motion':
                await handleAddMotion(intent);
                break;
            case 'add_character':
                await handleAddCharacter(intent);
                break;
            case 'add_object':
                await handleAddObject(intent);
                break;
            case 'modify_scene':
                handleModifyScene(intent);
                break;
            default:
                log('Not sure what to do with that — try adding a motion, another person, an object, or a scene change', 'system');
        }
        frameCreateStage();
    } catch (err) {
        log(`Error: ${err.message}`, 'error');
    }

    chatInput.disabled = false;
    chatInput.placeholder = 'Add a person, a motion, an object...';
    chatInput.focus();
}

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        const msg = chatInput.value.trim();
        chatInput.value = '';
        handleChat(msg);
    }
});

// Panel collapse/reopen
document.getElementById('panel-collapse').addEventListener('click', () => {
    document.getElementById('panel').classList.add('collapsed');
    setTimeout(() => document.getElementById('panel-reopen').classList.add('visible'), 200);
});
document.getElementById('panel-reopen').addEventListener('click', () => {
    document.getElementById('panel-reopen').classList.remove('visible');
    document.getElementById('panel').classList.remove('collapsed');
});

// Create-mode view controls, top centre of the viewport: Replay and Reset
// View. Always on once a scene exists (they used to hide until the camera had
// drifted 200 units, which is how "the reset button is missing" happened).
const createControls = document.getElementById('create-controls');
let _resetAnimating = false;
document.getElementById('replay-view').addEventListener('click', restartPlayback);
document.getElementById('reset-view').addEventListener('click', () => {
    const f = computeCreateFraming();
    if (!f || _resetAnimating) return;
    _resetAnimating = true;
    const startCam = camera.position.clone(), startTarget = controls.target.clone();
    let step = 0; const duration = 30; // frames
    (function animateReset() {
        step++;
        const t = step / duration, ease = t * t * (3 - 2 * t);
        camera.position.lerpVectors(startCam, f.pos, ease);
        controls.target.lerpVectors(startTarget, f.target, ease);
        controls.update();
        if (step < duration) requestAnimationFrame(animateReset); else _resetAnimating = false;
    })();
});

// ========== SCENE EDITOR ==========
const edToolbar = document.getElementById('editor-toolbar');
const edInfo = document.getElementById('editor-info');
const modelPicker = document.getElementById('model-picker');
let editorMode = null; // 'move' | 'rotate' | 'scale' | null
let dragObject = null;
let dragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let dragOffset = new THREE.Vector3();

// Show editor toolbar after first generation
function showEditor() { edToolbar.classList.add('visible'); }

function enterBuildMode() {
    document.querySelectorAll('.ed-build-only').forEach(b => b.style.display = 'flex');
}
function exitBuildMode() {
    document.querySelectorAll('.ed-build-only').forEach(b => b.style.display = 'none');
    setEditorMode(null);
}

function showEditorInfo(msg) {
    edInfo.textContent = msg;
    edInfo.classList.add('visible');
    clearTimeout(edInfo._t);
    edInfo._t = setTimeout(() => edInfo.classList.remove('visible'), 2000);
}

// === Panel tabs: Activity / Add ===
const tabActivity = document.getElementById('tab-activity');
const tabAdd = document.getElementById('tab-add');
const consoleEl = document.getElementById('console');
const chatBar = document.getElementById('chat-input-bar');
const addContent = document.getElementById('add-tab-content');

function switchTab(tab) {
    if (tab === 'activity') {
        tabActivity.classList.add('active'); tabAdd.classList.remove('active');
        consoleEl.style.display = ''; chatBar.style.display = '';
        addContent.classList.remove('visible');
    } else {
        tabAdd.classList.add('active'); tabActivity.classList.remove('active');
        consoleEl.style.display = 'none'; chatBar.style.display = 'none';
        addContent.classList.add('visible');
    }
}
tabActivity.addEventListener('click', () => switchTab('activity'));
tabAdd.addEventListener('click', () => switchTab('add'));

// The Add tab's grid of ready-made props (models/).
const addGrid = document.getElementById('add-tab-grid');

// Populate model picker with thumbnails
const AVAILABLE_MODELS = Object.keys(MODEL_MAP);

const MODEL_ICONS = { tree: '🌳', bush: '🌿', building: '🏢', house: '🏠', truck: '🚛' };

// Only ~a third of the models ship with a rendered thumbnail; the rest fall
// back to an emoji. models/thumbnails.json lists the ones that exist so we
// never fire off a request we know will 404.
let thumbnailManifest = null;
const thumbnailManifestReady = fetch('models/thumbnails.json')
    .then(r => r.ok ? r.json() : [])
    .catch(() => [])
    .then(list => { thumbnailManifest = new Set(list); });

function addModelToPicker(name, thumbUrl) {
    const div = document.createElement('div');
    div.className = 'mp-item';

    const label = document.createElement('span');
    label.textContent = name.replace(/_/g, ' ');

    const hasThumb = thumbUrl || (thumbnailManifest && thumbnailManifest.has(name));
    if (hasThumb) {
        const img = document.createElement('img');
        img.className = 'mp-thumb';
        img.src = thumbUrl || `models/${name}.png`;
        img.onerror = () => {
            img.remove();
            const icon = document.createElement('div');
            icon.className = 'mp-icon';
            icon.textContent = MODEL_ICONS[name] || '📦';
            div.insertBefore(icon, label);
        };
        div.appendChild(img);
    } else {
        const icon = document.createElement('div');
        icon.className = 'mp-icon';
        icon.textContent = MODEL_ICONS[name] || '📦';
        div.appendChild(icon);
    }

    div.appendChild(label);
    div.addEventListener('click', () => {
        enterPlaceMode(name);
        switchTab('activity');
    });
    addGrid.appendChild(div);
}

thumbnailManifestReady.then(() => AVAILABLE_MODELS.forEach(name => addModelToPicker(name)));

// Add button → switch to Add tab
if (document.getElementById('ed-add')) document.getElementById('ed-add').addEventListener('click', () => {
    switchTab('add');
});

// ========== PLACE MODE — model follows cursor until clicked ==========
let placeMode = false;
let placePreview = null;
let placeName = null;
const placeRaycaster = new THREE.Raycaster();
const placeMouse = new THREE.Vector2();
const placeGroundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const placeIntersect = new THREE.Vector3();

async function enterPlaceMode(name) {
    placeName = name;
    log(`Click in scene to place "${name}"`, 'system');

    // Try to load the GLB as a preview
    try {
        const url = `models/${name}.glb`;
        placePreview = (await loadGLBTemplate(url)).clone(); // the same trimmed, matte model the real placement uses
        const box = new THREE.Box3().setFromObject(placePreview);
        const size = new THREE.Vector3();
        box.getSize(size);
        const s = 100 / (size.y || 1);
        placePreview.scale.setScalar(s);
        // Make it semi-transparent
        placePreview.traverse(c => {
            if (c.isMesh) {
                c.material = c.material.clone();
                c.material.transparent = true;
                c.material.opacity = 0.5;
            }
        });
        scene.add(placePreview);
    } catch(e) {
        // Use a simple box as placeholder
        const geo = new THREE.BoxGeometry(40, 80, 40);
        const mat = new THREE.MeshStandardMaterial({ color: 0x7c5cbf, transparent: true, opacity: 0.4 });
        placePreview = new THREE.Mesh(geo, mat);
        placePreview.position.y = 40;
        const group = new THREE.Group();
        group.add(placePreview);
        placePreview = group;
        scene.add(placePreview);
    }

    placeMode = true;
    renderer.domElement.style.cursor = 'crosshair';
}

// Update preview position on mouse move
renderer.domElement.addEventListener('mousemove', (e) => {
    if (!placeMode || !placePreview) return;
    const rect = renderer.domElement.getBoundingClientRect();
    placeMouse.set(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    placeRaycaster.setFromCamera(placeMouse, camera);
    if (placeRaycaster.ray.intersectPlane(placeGroundPlane, placeIntersect)) {
        placePreview.position.x = placeIntersect.x;
        placePreview.position.z = placeIntersect.z;
    }
});

// Place on click
renderer.domElement.addEventListener('click', async (e) => {
    if (!placeMode || !placePreview) return;

    const finalPos = [placePreview.position.x, 0, placePreview.position.z];

    // Remove the transparent preview
    scene.remove(placePreview);
    placePreview = null;
    placeMode = false;
    renderer.domElement.style.cursor = '';

    // Load the real model at this position
    const url = `models/${placeName}.glb`;
    log(`Placing "${placeName}"...`, 'scene');
    try {
        await loadGLBModel(url, finalPos, 100, 0);
        log(`Placed "${placeName}"`, 'scene');
    } catch(e) {
        log(`Failed to place "${placeName}"`, 'error');
    }
});

// Cancel with Escape
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && placeMode) {
        if (placePreview) scene.remove(placePreview);
        placePreview = null;
        placeMode = false;
        renderer.domElement.style.cursor = '';
        log('Placement cancelled', 'system');
    }
});

// Mode buttons
function setEditorMode(mode) {
    editorMode = editorMode === mode ? null : mode;
    document.querySelectorAll('.ed-btn').forEach(b => b.classList.remove('active'));
    if (editorMode) {
        document.getElementById('ed-' + editorMode).classList.add('active');
        showEditorInfo(editorMode === 'move' ? 'Click + drag to move' :
            editorMode === 'rotate' ? 'Click object, drag to rotate' :
            'Click object, drag to scale');
    }
}
document.getElementById('ed-move').addEventListener('click', () => setEditorMode('move'));
document.getElementById('ed-rotate').addEventListener('click', () => setEditorMode('rotate'));
document.getElementById('ed-scale').addEventListener('click', () => setEditorMode('scale'));

// Keyboard shortcuts for editor
document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'm') setEditorMode('move');
    if (e.key === 'r') setEditorMode('rotate');
    if (e.key === 's' && !e.ctrlKey) setEditorMode('scale');
    if (e.key === 'Escape') {
        setEditorMode(null);
        if (selectedObject) {
            selectedObject.traverse(c => {
                if (c.isMesh && c.userData._oe) c.material.emissive.copy(c.userData._oe);
            });
            selectedObject = null;
            selectedType = null;
            removeSelectionBox();
        }
        exitBuildMode();
        modelPicker.classList.remove('visible');
    }
});

// Drag handling for move/rotate/scale
let _dragStartX = 0, _dragStartY = 0;
let _dragStartRot = 0, _dragStartScale = 1;

renderer.domElement.addEventListener('pointerdown', (e) => {
    if (!editorMode || !selectedObject || selectedType !== 'scene') return;
    dragObject = selectedObject;
    _dragStartX = e.clientX;
    _dragStartY = e.clientY;
    if (editorMode === 'rotate') _dragStartRot = dragObject.rotation.y;
    if (editorMode === 'scale') _dragStartScale = dragObject.scale.x;

    if (editorMode === 'move') {
        // Calculate drag offset on ground plane
        const rect = renderer.domElement.getBoundingClientRect();
        const mouse = new THREE.Vector2(
            ((e.clientX - rect.left) / rect.width) * 2 - 1,
            -((e.clientY - rect.top) / rect.height) * 2 + 1
        );
        raycaster.setFromCamera(mouse, camera);
        const hit = new THREE.Vector3();
        raycaster.ray.intersectPlane(dragPlane, hit);
        dragOffset.subVectors(dragObject.position, hit);
        controls.enabled = false; // disable orbit while dragging
    } else {
        controls.enabled = false;
    }
});

renderer.domElement.addEventListener('pointermove', (e) => {
    if (!dragObject || !editorMode) return;

    if (editorMode === 'move') {
        const rect = renderer.domElement.getBoundingClientRect();
        const mouse = new THREE.Vector2(
            ((e.clientX - rect.left) / rect.width) * 2 - 1,
            -((e.clientY - rect.top) / rect.height) * 2 + 1
        );
        raycaster.setFromCamera(mouse, camera);
        const hit = new THREE.Vector3();
        if (raycaster.ray.intersectPlane(dragPlane, hit)) {
            dragObject.position.x = hit.x + dragOffset.x;
            dragObject.position.z = hit.z + dragOffset.z;
        }
    } else if (editorMode === 'rotate') {
        const dx = e.clientX - _dragStartX;
        dragObject.rotation.y = _dragStartRot + dx * 0.01;
    } else if (editorMode === 'scale') {
        const dy = _dragStartY - e.clientY;
        const s = Math.max(0.1, _dragStartScale * (1 + dy * 0.005));
        dragObject.scale.setScalar(s);
    }
});

renderer.domElement.addEventListener('pointerup', () => {
    if (dragObject) {
        dragObject = null;
        controls.enabled = true;
    }
});

// Show editor after first scene generation
const _origGenerate = generate;

// ========== LYRIA MUSIC ==========
const LYRIA_URL = `https://generativelanguage.googleapis.com/v1beta/models/lyria-3-clip-preview:generateContent?key=${GEMINI_KEY}`;
let musicAudio = null; // <audio> element for current soundtrack
let musicMuted = false;
let musicAudioCtx = null;
let musicStreamDest = null;

async function generateMusic(scenePrompt) {
    setPill('pill-music', true);
    log('Generating soundtrack...', 'music', 'music-status');

    const musicPrompt = `Create a 30-second instrumental soundtrack for this scene: "${scenePrompt}".
Make it atmospheric and cinematic. No vocals, no lyrics. Match the mood and energy of the scene.
If the scene is dark or spooky, use minor keys and tension. If it's happy or active, use upbeat tempo.
Keep it subtle enough to be background music for a 3D animation.`;

    try {
        const res = await fetch(LYRIA_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                contents: [{ role: 'user', parts: [{ text: musicPrompt }] }],
                generationConfig: { responseModalities: ['AUDIO'] }
            })
        });

        const data = await res.json();
        const parts = data?.candidates?.[0]?.content?.parts;
        if (!parts) throw new Error('No audio in response');

        const audioPart = parts.find(p => p.inlineData);
        if (!audioPart) throw new Error('No audio data returned');

        const b64 = audioPart.inlineData.data;
        const mime = audioPart.inlineData.mimeType || 'audio/mp3';

        // Create audio element from base64
        if (musicAudio) { musicAudio.pause(); musicAudio.remove(); }
        musicAudioCtx = null;
        musicStreamDest = null;
        musicAudio = document.createElement('audio');
        musicAudio.src = `data:${mime};base64,${b64}`;
        musicAudio.loop = true;
        musicAudio.volume = 0.35;
        musicAudio.muted = musicMuted;

        // Set up persistent audio capture route
        try {
            musicAudioCtx = new AudioContext();
            const source = musicAudioCtx.createMediaElementSource(musicAudio);
            musicStreamDest = musicAudioCtx.createMediaStreamDestination();
            source.connect(musicStreamDest);
            source.connect(musicAudioCtx.destination); // still plays through speakers
        } catch (e) { /* fallback: no capture */ }

        // Fade in
        musicAudio.volume = 0;
        musicAudio.play().catch(() => {}); // may need user interaction
        let fadeIn = setInterval(() => {
            if (musicAudio.volume < 0.35) {
                musicAudio.volume = Math.min(0.35, musicAudio.volume + 0.02);
            } else {
                clearInterval(fadeIn);
            }
        }, 100);

        log('Soundtrack playing', 'music', 'music-status');
    } catch (err) {
        log(`Music: ${err.message}`, 'error', 'music-status');
    }
    setPill('pill-music', false);
}

// Music toggle removed — music only used during video render

// Get audio stream for mixing into video recording
function getMusicStream() {
    if (!musicStreamDest) return null;
    return musicStreamDest.stream;
}

// ========== RENDER VIDEO (Orbit Capture) ==========
const renderOverlay = document.getElementById('render-overlay');
const renderDot = document.getElementById('render-dot');
const renderHint = document.getElementById('render-hint');
const renderProgress = document.getElementById('render-progress');
const renderFill = document.getElementById('render-fill');
const renderBtn = document.getElementById('render-btn');
let renderMode = false;

renderBtn.addEventListener('click', async () => {
    if (isRecording) return;
    if (!currentClip) { log('Nothing to render yet', 'system'); return; }

    // Find orbit center — use character position
    const orbitCenter = new THREE.Vector3();
    if (currentBones) currentBones.getWorldPosition(orbitCenter);
    orbitCenter.y = 0;

    // Save original camera state
    const origPos = camera.position.clone();
    const origTarget = controls.target.clone();
    controls.enabled = false;

    // Zoom out to a high angle first
    const ORBIT_RADIUS = 700;
    const ORBIT_HEIGHT = 550;
    const startAngle = Math.atan2(
        camera.position.x - orbitCenter.x,
        camera.position.z - orbitCenter.z
    );

    // Smooth transition to zoomed-out view
    const zoomTarget = new THREE.Vector3(
        orbitCenter.x + Math.sin(startAngle) * ORBIT_RADIUS,
        ORBIT_HEIGHT,
        orbitCenter.z + Math.cos(startAngle) * ORBIT_RADIUS
    );
    const ZOOM_DURATION = 800;
    const zoomStart = Date.now();
    const zoomFrom = camera.position.clone();
    await new Promise(resolve => {
        function zoomStep() {
            const t = Math.min((Date.now() - zoomStart) / ZOOM_DURATION, 1);
            const e = t * t * (3 - 2 * t); // smoothstep
            camera.position.lerpVectors(zoomFrom, zoomTarget, e);
            camera.lookAt(orbitCenter.x, 40, orbitCenter.z);
            renderFrame();
            if (t < 1) requestAnimationFrame(zoomStep);
            else resolve();
        }
        requestAnimationFrame(zoomStep);
    });

    // Generate music if not already available
    if (!musicAudio || musicAudio.paused) {
        const scenePrompt = timelineClips.map(c => c.prompt).join('. ') || 'ambient scene';
        renderFill.style.width = '0%';
        renderProgress.style.display = 'block';
        log('Generating soundtrack...', 'music', 'render-status');
        await generateMusic(scenePrompt);
    }

    // Restart music from beginning for clean recording
    if (musicAudio) {
        musicAudio.currentTime = 0;
        musicAudio.muted = false;
        musicAudio.volume = 0.35;
        await musicAudio.play().catch(() => {});
        // Resume AudioContext if suspended (autoplay policy)
        if (musicAudioCtx && musicAudioCtx.state === 'suspended') {
            await musicAudioCtx.resume();
        }
    }

    // Show progress bar
    renderProgress.style.display = 'block';
    renderFill.style.width = '0%';

    // Setup MediaRecorder
    const canvas = renderer.domElement;
    const videoStream = canvas.captureStream(30);
    const stream = new MediaStream([...videoStream.getTracks()]);
    if (musicStreamDest) {
        musicStreamDest.stream.getAudioTracks().forEach(t => stream.addTrack(t));
    }

    const chunks = [];
    const recorder = new MediaRecorder(stream, {
        mimeType: 'video/webm;codecs=vp9',
        videoBitsPerSecond: 8000000
    });
    recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'animo-render.webm';
        a.click();
        URL.revokeObjectURL(url);
        log('Video saved', 'success');
    };

    // Orbit parameters
    const DURATION = 6000;
    const ROTATION_AMOUNT = Math.PI * 1.5;
    let lastFrameTime = performance.now();

    const startTime = Date.now();
    isRecording = true;
    recorder.start();
    log('Recording orbit...', 'render', 'render-status');

    function orbitFrame(now) {
        const rawDt = (now - lastFrameTime) / 1000;
        const dt = Math.min(rawDt, 0.05);
        lastFrameTime = now;
        const elapsed = Date.now() - startTime;
        const t = Math.min(elapsed / DURATION, 1);
        renderFill.style.width = (t * 100) + '%';

        const eased = t < 0.5
            ? 2 * t * t
            : 1 - Math.pow(-2 * t + 2, 2) / 2;
        const angle = startAngle + eased * ROTATION_AMOUNT;
        camera.position.set(
            orbitCenter.x + Math.sin(angle) * ORBIT_RADIUS,
            ORBIT_HEIGHT,
            orbitCenter.z + Math.cos(angle) * ORBIT_RADIUS
        );
        camera.lookAt(orbitCenter.x, 40, orbitCenter.z);

        for (const c of characters) c.mixer.update(dt);
        if (objectMixer) { objectMixer.update(dt); updateObjectAnimations(objectMixer.time); }
        for (const c of characters) updateBodyMeshesIn(c.bodyMeshesArr);
        renderFrame();

        if (t < 1) {
            requestAnimationFrame(orbitFrame);
        } else {
            isRecording = false;
            recorder.stop();
            camera.position.copy(origPos);
            controls.target.copy(origTarget);
            controls.enabled = true;
            controls.update();

            // Stop music after render
            if (musicAudio) { musicAudio.pause(); musicAudio.currentTime = 0; }

            renderProgress.style.display = 'none';
            log('Export complete', 'success', 'render-status');
        }
    }
    requestAnimationFrame(orbitFrame);
});

// Render mode no longer uses overlay click — renders directly on button press

// ========== .ANIMO EXPORT / IMPORT ==========

// Export removed

// Import .animo file
async function handleImportFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    e.target.value = ''; // reset for re-import

    try {
        const text = await file.text();
        const data = JSON.parse(text);

        if (data.format !== 'animo') {
            log('Invalid .animo file', 'error');
            return;
        }

        log(`Importing "${file.name}"...`, 'system');

        document.getElementById('panel').classList.remove('collapsed');

        // Rebuild scene from saved data
        const config = {
            scene: {
                type: 'outdoor',
                models: data.scene.models.map(m => ({
                    keyword: m.keyword,
                    category: 6,
                    size: 'medium'
                })),
                ground: { color: data.scene.ground_color },
                lights: data.scene.lights || []
            },
            _characterPath: [[0, 0]]
        };

        // Build scene first
        await buildScene(config);

        // Now reposition models to their saved positions
        let modelIdx = 0;
        for (const obj of sceneObjects) {
            if (obj.userData._isGround) continue;
            if (obj.isLight) continue;
            if (modelIdx < data.scene.models.length) {
                const saved = data.scene.models[modelIdx];
                obj.position.set(saved.position[0], saved.position[1], saved.position[2]);
                obj.rotation.y = saved.rotation || 0;
                obj.scale.setScalar(saved.scale || 1);
                modelIdx++;
            }
        }

        // Load BVH animation
        if (data.bvh) {
            const importAvatar = avatarForIndex(0);
        await ensureAvatar(importAvatar).catch(() => {});
        await loadBVH(data.bvh, '', avatarCache[importAvatar] ? importAvatar : null);
            lastBvhText = data.bvh;

            // Rebuild timeline
            timelineClips = data.timeline.map(t => ({
                prompt: t.prompt,
                duration: t.duration,
                clip: currentClip
            }));
            renderTimelineClips();
            showTimeline();
            updatePlayPauseIcon();
        }

        // Restore camera
        if (data.camera) {
            camera.position.set(...data.camera.position);
            controls.target.set(...data.camera.target);
            controls.update();
        }

        log(`Imported "${file.name}" successfully`, 'success');

    } catch (err) {
        log(`Import error: ${err.message}`, 'error');
    }
}
document.getElementById('import-file').addEventListener('change', handleImportFile);

// ========== VIDEO ORBIT RECORDING ==========
let recordMode = false, orbitMarker = null;
if (document.getElementById('ed-record')) document.getElementById('ed-record').addEventListener('click', () => {
    if (recordMode) { recordMode = false; if (orbitMarker) { scene.remove(orbitMarker); orbitMarker = null; } renderer.domElement.style.cursor = 'default'; document.getElementById('ed-record').classList.remove('active'); return; }
    recordMode = true; renderer.domElement.style.cursor = 'crosshair'; document.getElementById('ed-record').classList.add('active');
    showEditorInfo('Click where the camera should orbit');
    orbitMarker = new THREE.Mesh(new THREE.SphereGeometry(6,16,12), new THREE.MeshStandardMaterial({color:0x22c55e,emissive:0x16a34a,emissiveIntensity:0.5}));
    orbitMarker.position.y = 3; scene.add(orbitMarker);
});
renderer.domElement.addEventListener('mousemove', (ev) => {
    if (!recordMode || !orbitMarker) return;
    const rect = renderer.domElement.getBoundingClientRect();
    const m2 = new THREE.Vector2(((ev.clientX-rect.left)/rect.width)*2-1,-((ev.clientY-rect.top)/rect.height)*2+1);
    raycaster.setFromCamera(m2, camera);
    const hp = new THREE.Vector3();
    if (raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0,1,0),0), hp)) orbitMarker.position.set(hp.x,3,hp.z);
});
renderer.domElement.addEventListener('click', async (ev2) => {
    if (!recordMode || !orbitMarker) return;
    const ctr = orbitMarker.position.clone(); ctr.y = 100;
    recordMode = false; scene.remove(orbitMarker); orbitMarker = null;
    renderer.domElement.style.cursor = 'default'; document.getElementById('ed-record').classList.remove('active');
    const dur = Math.max(5, totalDuration/1.5), fps = 30, nF = Math.round(dur*fps);
    document.getElementById('rec-badge').classList.add('visible'); controls.enabled = false;
    const strm = renderer.domElement.captureStream(fps);
    const mt = MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm';
    const mrec = new MediaRecorder(strm, {mimeType:mt, videoBitsPerSecond:8000000});
    const chnks = []; mrec.ondataavailable = e3 => { if (e3.data.size>0) chnks.push(e3.data); };
    isRecording = true;
    mrec.start(); for (const c of characters) c.mixer.setTime(0); if (objectMixer) objectMixer.setTime(0);
    const savC = camera.position.clone(), savT = controls.target.clone();
    for (let f = 0; f < nF; f++) {
        const ang = (f/nF)*Math.PI*2;
        camera.position.set(ctr.x+Math.cos(ang)*500, ctr.y+250, ctr.z+Math.sin(ang)*500);
        camera.lookAt(ctr);
        for (const c of characters) c.mixer.update(1/fps);
        if (objectMixer) { objectMixer.update(1/fps); updateObjectAnimations(objectMixer.time); }
        for (const c of characters) updateBodyMeshesIn(c.bodyMeshesArr);
        renderFrame();
        await new Promise(r => requestAnimationFrame(r));
    }
    isRecording = false;
    mrec.stop(); document.getElementById('rec-badge').classList.remove('visible');
    camera.position.copy(savC); controls.target.copy(savT); controls.enabled = true; controls.update();
    mrec.onstop = () => {
        const bl = new Blob(chnks,{type:'video/webm'}); const u = URL.createObjectURL(bl);
        const dl = document.createElement('a'); dl.href = u; dl.download = 'animo_scene.webm'; dl.click();
        URL.revokeObjectURL(u); log('Video exported!','success');
    };
});

// ========== .ANIMO SAVE ==========
if (document.getElementById('ed-save')) document.getElementById('ed-save').addEventListener('click', () => {
    const d = {
        version:'1.0', name:'My Scene', created:new Date().toISOString(),
        timeline: timelineClips.map(c => ({prompt:c.prompt,duration:c.duration})),
        totalDuration: totalDuration,
        objects: sceneObjects.filter(o => !o.userData._isGround).map(o => ({
            pos:[Math.round(o.position.x),Math.round(o.position.y),Math.round(o.position.z)],
            rot:+(o.rotation.y).toFixed(2), scale:+(o.scale.x).toFixed(2)
        })),
        camera:{pos:[Math.round(camera.position.x),Math.round(camera.position.y),Math.round(camera.position.z)],
            target:[Math.round(controls.target.x),Math.round(controls.target.y),Math.round(controls.target.z)]},
        activity: Array.from(document.querySelectorAll('.log-msg')).slice(-50).map(el => el.textContent)
    };
    const bl = new Blob([JSON.stringify(d,null,2)],{type:'application/json'});
    const aa = document.createElement('a'); aa.href = URL.createObjectURL(bl);
    aa.download = 'my_scene.animo'; aa.click(); URL.revokeObjectURL(aa.href);
    log('Saved my_scene.animo','success');
});


// ========================================================================
// AUTH0 (stub) — sign-in gate so Backboard memory can be scoped per user.
// Swap the body of signIn()/getUserId() for the real Auth0 SPA SDK once
// AUTH0_DOMAIN / AUTH0_CLIENT_ID are set. Until then this fakes a stable
// per-browser user id so the memory features below are demoable today.
// ========================================================================
const AUTH0_DOMAIN = ''; // e.g. 'your-tenant.us.auth0.com'
const AUTH0_CLIENT_ID = '';
const authBtn = document.getElementById('auth-btn');

function getUserId() {
    let uid = localStorage.getItem('animo_user_id');
    if (!uid) {
        uid = 'guest-' + Math.random().toString(36).slice(2, 10);
        localStorage.setItem('animo_user_id', uid);
    }
    return uid;
}

function isSignedIn() {
    return !!localStorage.getItem('animo_signed_in');
}

function updateAuthBtn() {
    if (isSignedIn()) {
        authBtn.textContent = getUserId().replace('guest-', '');
        authBtn.title = 'Signed in — click to sign out';
    } else {
        authBtn.textContent = 'Sign in';
        authBtn.title = 'Sign in';
    }
}

authBtn.addEventListener('click', () => {
    if (!AUTH0_DOMAIN || !AUTH0_CLIENT_ID) {
        // TODO: replace with real Auth0 SPA SDK (createAuth0Client, loginWithRedirect)
        if (isSignedIn()) {
            localStorage.removeItem('animo_signed_in');
            log('Signed out', 'system');
        } else {
            localStorage.setItem('animo_signed_in', '1');
            log(`Signed in as ${getUserId()} (Auth0 not configured — using a local guest id)`, 'system');
        }
        updateAuthBtn();
        renderLearnedList();
        renderReferenceLibrary();
        return;
    }
    // Real Auth0 flow goes here once configured.
});
updateAuthBtn();

// ========================================================================
// BACKBOARD (persistent memory) — thin client with a localStorage fallback
// so "Skills learned" and "Reference library" actually work today. Point
// BACKBOARD_API_URL / BACKBOARD_API_KEY at a real Backboard project and
// backboardGet/backboardSet below hit the real API instead.
// ========================================================================
const BACKBOARD_API_URL = ''; // e.g. 'https://api.backboard.io'
const BACKBOARD_API_KEY = '';

async function backboardGet(collection) {
    const uid = getUserId();
    if (BACKBOARD_API_URL && BACKBOARD_API_KEY) {
        try {
            const res = await fetch(`${BACKBOARD_API_URL}/v1/users/${uid}/${collection}`, {
                headers: { 'Authorization': `Bearer ${BACKBOARD_API_KEY}` }
            });
            if (res.ok) return await res.json();
        } catch (e) { console.warn('Backboard unavailable, falling back to local storage', e); }
    }
    return JSON.parse(localStorage.getItem(`animo_bb_${uid}_${collection}`) || '[]');
}

async function backboardSet(collection, items) {
    const uid = getUserId();
    localStorage.setItem(`animo_bb_${uid}_${collection}`, JSON.stringify(items));
    if (BACKBOARD_API_URL && BACKBOARD_API_KEY) {
        try {
            await fetch(`${BACKBOARD_API_URL}/v1/users/${uid}/${collection}`, {
                method: 'PUT',
                headers: { 'Authorization': `Bearer ${BACKBOARD_API_KEY}`, 'Content-Type': 'application/json' },
                body: JSON.stringify(items)
            });
        } catch (e) { console.warn('Backboard write failed, kept locally only', e); }
    }
}

async function rememberSkill(skill) {
    const list = await backboardGet('skills_learned');
    const existing = list.find(s => s.skill.toLowerCase() === skill.toLowerCase());
    if (existing) { existing.count++; existing.lastAsked = Date.now(); }
    else list.unshift({ skill, count: 1, lastAsked: Date.now() });
    await backboardSet('skills_learned', list);
    return existing ? existing.count : 1;
}

async function renderLearnedList() {
    const listEl = document.getElementById('learned-list');
    if (!listEl) return;
    const skills = await backboardGet('skills_learned');
    listEl.innerHTML = '';
    if (skills.length === 0) {
        listEl.innerHTML = '<div class="learned-empty">Nothing yet — ask Help mode how to do something.</div>';
        return;
    }
    skills.forEach(s => {
        const row = document.createElement('div');
        row.className = 'learned-row';
        row.innerHTML = `<span class="learned-skill">${s.skill}</span><span class="learned-count">${s.count > 1 ? `asked ${s.count}x` : ''}</span>`;
        row.addEventListener('click', () => { setMode('help'); askHelp(`Teach me ${s.skill}`); });
        listEl.appendChild(row);
    });
}

// ========================================================================
// REFERENCE LIBRARY (Create mode) — style references that fold into the
// Gemini scene prompt. Persisted via Backboard (localStorage fallback).
// ========================================================================
async function renderReferenceLibrary() {
    const grid = document.getElementById('lib-references-grid');
    if (!grid) return;
    const refs = await backboardGet('references');
    grid.innerHTML = '';
    refs.forEach((ref, i) => {
        const card = document.createElement('div');
        card.className = 'ref-card';
        if (ref.type === 'image') {
            card.innerHTML = `<img src="${ref.value}" alt="reference"><button class="ref-remove" data-i="${i}">&times;</button>`;
        } else {
            card.innerHTML = `<div class="ref-note">${ref.value}</div><button class="ref-remove" data-i="${i}">&times;</button>`;
        }
        grid.appendChild(card);
    });
    grid.querySelectorAll('.ref-remove').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const refs2 = await backboardGet('references');
            refs2.splice(+btn.dataset.i, 1);
            await backboardSet('references', refs2);
            renderReferenceLibrary();
        });
    });
}

async function addReference(value, type = 'note') {
    const refs = await backboardGet('references');
    refs.unshift({ type, value, addedAt: Date.now() });
    await backboardSet('references', refs);
    renderReferenceLibrary();
}

// Returns a short text summary of the current reference library to fold
// into the Gemini scene prompt as style context.
async function getReferenceStyleContext() {
    const refs = await backboardGet('references');
    const notes = refs.filter(r => r.type === 'note').map(r => r.value);
    if (notes.length === 0) return '';
    return `\n\nSTYLE REFERENCES (incorporate this aesthetic): ${notes.join('; ')}`;
}

const libRefInput = document.getElementById('lib-add-reference-input');
const libRefBtn = document.getElementById('lib-add-reference-btn');
const libRefFile = document.getElementById('lib-add-reference-file');

libRefBtn.addEventListener('click', () => {
    const val = libRefInput.value.trim();
    if (!val) { libRefFile.click(); return; }
    const isUrl = /^https?:\/\//i.test(val) || val.startsWith('data:image');
    addReference(val, isUrl ? 'image' : 'note');
    libRefInput.value = '';
});
libRefInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') libRefBtn.click(); });
libRefFile.addEventListener('change', () => {
    const file = libRefFile.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => addReference(reader.result, 'image');
    reader.readAsDataURL(file);
    libRefFile.value = '';
});

// ========================================================================
// HELP MODE — voice Q&A with an optional live demonstration animation.
// ========================================================================
const HELP_SYSTEM_PROMPT = `You are a friendly, concise assistant that can physically demonstrate things in a 3D viewport — either by animating a human body, or by moving an object along a path (a car parking, a ball rolling, a door swinging). Given a user question, respond with ONLY valid JSON (no markdown, no backticks):

{"answer":"spoken-friendly answer, 2-4 short sentences","demo_type":"human_motion"|"object_path"|"none","motion_prompt":"A person ...","object_keyword":"car","path":[{"t":0,"pos":[0,0,0],"yaw":0}],"steps":["step one","step two"]}

RULES:
- demo_type "human_motion": the question is about a movement a PERSON'S BODY performs (backflip, push-up, riding a bike, a dance move, a stretch, a martial arts move). Set motion_prompt (MUST start with "A person", a single continuous demonstrable action, max ~8 seconds). Leave object_keyword/path empty.
- demo_type "object_path": the question is about something an OBJECT/VEHICLE/MACHINE does, not the human body (parallel parking, how a car does a 3-point turn, a garage door opening, a ball rolling downhill). Set object_keyword to a simple one-word English noun for the object (e.g. "car", "boat", "bicycle") and path to 3-6 keyframes tracing its motion: {"t": seconds from 0, "pos": [x,y,z] offset in scene units from its start point (roughly -80..80 per axis), "yaw": facing angle in degrees, 0 = forward, positive = turning left}. Keep total duration under 8 seconds. Leave motion_prompt empty.
- demo_type "none": factual/conceptual questions with nothing physical to demonstrate. Leave motion_prompt, object_keyword and path empty/blank.
- steps: 2-4 short spoken cues in the order they happen during the demo (empty array if demo_type is "none").
- Keep "answer" short — it will be read aloud.`;

async function callGeminiHelp(question) {
    const data = await fetchGeminiWithRetry(GEMINI_URL, {
        contents: [{ role: 'user', parts: [{ text: HELP_SYSTEM_PROMPT + '\n\nUser question: ' + question }] }],
        generationConfig: jsonGenConfig(0.6, 900)
    });
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!text) throw new Error(geminiErrorMessage(data));
    const clean = text.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
    return JSON.parse(clean);
}

// ========================================================================
// VOICE — the persona speaks her answers in a voice that matches whichever
// character is on stage. ElevenLabs when a key is present (set it in
// local-config.js, same as the Gemini key), otherwise the browser's own
// speech synthesis. The built-in voices only sound robotic because nothing
// ever picked one: naming a good installed voice per character gets most of
// the way there for free, offline, with no quota.
// ========================================================================
const ELEVENLABS_KEY = window.ANIMO_ELEVENLABS_KEY || '';
const ELEVENLABS_VOICE_IDS = window.ANIMO_ELEVENLABS_VOICES || {}; // optional { male, female } overrides
let elevenVoiceByKey = null;   // resolved once from the account's own voice list
let currentSpeech = null;      // the <audio> currently talking, so a new answer can cut it off

// Which ElevenLabs voice this character speaks with. Rather than hard-coding
// voice IDs (they differ per account), ask the account what it has and take
// the first one labelled with the right gender.
async function elevenVoiceFor(key) {
    if (ELEVENLABS_VOICE_IDS[key]) return ELEVENLABS_VOICE_IDS[key];
    if (!elevenVoiceByKey) {
        const res = await fetch('https://api.elevenlabs.io/v1/voices', { headers: { 'xi-api-key': ELEVENLABS_KEY } });
        if (!res.ok) throw new Error(`voice list failed (${res.status})`);
        const voices = (await res.json()).voices || [];
        const byGender = (g) => voices.find(v => ((v.labels && v.labels.gender) || '').toLowerCase() === g);
        elevenVoiceByKey = {
            male: (byGender('male') || voices[0] || {}).voice_id,
            female: (byGender('female') || voices[0] || {}).voice_id,
        };
    }
    return elevenVoiceByKey[key];
}

let speechSeq = 0;             // bumped on every stop, so a reply still downloading can tell it was superseded

// Feed the streamed MP3 into a MediaSource as it arrives, so she starts
// talking after the first chunk instead of after the whole answer downloads.
function streamIntoAudio(audio, body) {
    const ms = new MediaSource();
    const url = URL.createObjectURL(ms);
    audio.src = url;
    ms.addEventListener('sourceopen', async () => {
        const sb = ms.addSourceBuffer('audio/mpeg');
        const reader = body.getReader();
        const append = (chunk) => new Promise((resolve, reject) => {
            sb.addEventListener('updateend', resolve, { once: true });
            sb.addEventListener('error', reject, { once: true });
            sb.appendBuffer(chunk);
        });
        try {
            for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                if (ms.readyState !== 'open') return reader.cancel(); // stopped mid-answer
                await append(value);
            }
            if (ms.readyState === 'open') ms.endOfStream();
        } catch (err) {
            if (ms.readyState === 'open') ms.endOfStream('decode');
        }
    }, { once: true });
    return url;
}

// 64kbps is half the bytes of the default 128 with no audible loss for speech.
const elevenTtsUrl = (voiceId) => `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}/stream?output_format=mp3_44100_64`;
const elevenWarmedAt = {};

// Every TTS call needs a CORS preflight (~0.3s) that the browser caches for
// 10 minutes per URL. Called while Gemini is still thinking, this pays that
// cost up front: a GET on the same URL triggers the same preflight (which
// allows POST) but is refused with a 405 — no speech generated, no quota —
// leaving the preflight cached and the TLS connection warm for the reply.
function warmEleven() {
    if (!ELEVENLABS_KEY) return;
    const key = avatarKey;
    if (Date.now() - (elevenWarmedAt[key] || 0) < 9 * 60 * 1000) return;
    elevenWarmedAt[key] = Date.now();
    elevenVoiceFor(key)
        .then(voiceId => voiceId && fetch(elevenTtsUrl(voiceId), { headers: { 'xi-api-key': ELEVENLABS_KEY } }))
        .catch(() => {});
}

async function speakEleven(text, onEnd, onStart) {
    const seq = speechSeq;
    const voiceId = await elevenVoiceFor(avatarKey);
    if (!voiceId) throw new Error('no ElevenLabs voice available');
    const res = await fetch(elevenTtsUrl(voiceId), {
        method: 'POST',
        headers: { 'xi-api-key': ELEVENLABS_KEY, 'Content-Type': 'application/json', 'Accept': 'audio/mpeg' },
        body: JSON.stringify({
            text,
            model_id: 'eleven_turbo_v2_5', // natural delivery, first audio in ~0.3s
            voice_settings: { stability: 0.5, similarity_boost: 0.8, style: 0.15, use_speaker_boost: true, speed: 0.9 },
        }),
    });
    if (!res.ok) throw new Error(`text-to-speech failed (${res.status})`);
    if (seq !== speechSeq) { res.body && res.body.cancel(); return; } // a newer answer (or a stop) came in meanwhile

    const audio = new Audio();
    const canStream = res.body && window.MediaSource && MediaSource.isTypeSupported('audio/mpeg');
    const url = canStream ? streamIntoAudio(audio, res.body) : URL.createObjectURL(await res.blob());
    if (!canStream) audio.src = url;
    if (seq !== speechSeq) { URL.revokeObjectURL(url); return; }
    currentSpeech = audio;

    let done = false;
    const finish = () => {
        if (done) return;
        done = true;
        URL.revokeObjectURL(url);
        if (currentSpeech === audio) currentSpeech = null;
        if (onEnd) onEnd();
    };
    // The gesture starts on the first audible frame, not when the request goes out.
    audio.addEventListener('playing', () => { if (onStart) onStart(); }, { once: true });
    audio.onended = finish;
    audio.onerror = finish;
    audio.onpause = () => { if (currentSpeech !== audio) finish(); }; // stopSpeaking() cut her off
    await audio.play();
}

// Pick the best installed voice for this character: a named favourite if the
// browser has one, else anything whose own metadata matches the gender, else
// whatever the browser defaults to.
function systemVoiceFor(key) {
    const spec = AVATARS[key] || {};
    const voices = window.speechSynthesis.getVoices();
    if (!voices.length) return null; // Chrome populates this asynchronously; fine, we just use the default
    for (const name of spec.systemVoices || []) {
        const hit = voices.find(v => v.name.toLowerCase().includes(name.toLowerCase()));
        if (hit) return hit;
    }
    const g = spec.voiceGender;
    return (g && voices.find(v => new RegExp(g, 'i').test(v.name))) || null;
}

function speakSystem(text, onEnd, onStart) {
    // Roughly how long the line takes to say, used to end the gesture when
    // there is no speech engine to tell us (and as a backstop if onend never
    // fires, which some browsers do when the tab loses focus).
    const words = text.trim().split(/\s+/).length;
    const estimate = Math.max(1.5, words / 2.8) * 1000;

    if (!('speechSynthesis' in window)) {
        if (onStart) onStart();
        if (onEnd) setTimeout(onEnd, estimate);
        return;
    }
    let done = false, started = false;
    const start = () => { if (!started) { started = true; if (onStart) onStart(); } };
    const finish = () => { start(); if (!done) { done = true; if (onEnd) onEnd(); } };
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    const voice = systemVoiceFor(avatarKey);
    if (voice) { utter.voice = voice; utter.lang = voice.lang; }
    utter.rate = 0.92;
    utter.onstart = start;
    setTimeout(start, 1500); // some browsers never fire onstart
    utter.onend = finish;
    utter.onerror = finish;
    window.speechSynthesis.speak(utter);
    setTimeout(finish, estimate + 4000);
}

function stopSpeaking() {
    speechSeq++;
    if (currentSpeech) { currentSpeech.pause(); currentSpeech = null; }
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
}

// onStart fires when sound actually begins (so the talking gesture lines up
// with the voice), onEnd when it finishes or is cut off.
function speak(text, onEnd, onStart) {
    stopSpeaking();
    if (!ELEVENLABS_KEY) return speakSystem(text, onEnd, onStart);
    const seq = speechSeq;
    // Any failure — bad key, quota spent, offline — just falls back rather
    // than leaving her silent.
    speakEleven(text, onEnd, onStart).catch(err => {
        if (seq !== speechSeq) return; // superseded; nothing to fall back for
        if (currentSpeech) { currentSpeech.pause(); currentSpeech = null; }
        console.warn('ElevenLabs unavailable, using the browser voice:', err.message);
        speakSystem(text, onEnd, onStart);
    });
}

// Whose chat this is. ANIMO_USER_NAME (local-config.js) when set, else the
// signed-in id, else nobody in particular — the greeting drops the name
// rather than saying hello to "guest-4f2c1a".
function displayName() {
    const configured = (window.ANIMO_USER_NAME || '').trim();
    if (configured) return configured;
    if (isSignedIn()) return getUserId().replace('guest-', '');
    return '';
}

// Bubbles are what counts as history — the greeting is a child of the log too,
// so a plain children.length would always look like a started conversation.
function chatHasHistory() {
    return document.querySelectorAll('#help-chat-log .help-bubble').length > 0;
}

// Chat opens on a greeting rather than a blank panel.
function syncChatGreeting() {
    const el = document.getElementById('help-greeting');
    if (!el) return;
    const name = displayName();
    el.textContent = name ? `Hi ${name}, let's get into it` : `Hi, let's get into it`;
    el.style.display = chatHasHistory() ? 'none' : 'block';
}

function addHelpBubble(role, text) {
    const log_el = document.getElementById('help-chat-log');
    const bubble = document.createElement('div');
    bubble.className = `help-bubble help-${role}`;
    bubble.textContent = text;
    log_el.appendChild(bubble);
    syncChatGreeting(); // first bubble retires the empty state
    log_el.scrollTop = log_el.scrollHeight;
}

// ========================================================================
// CHAT PERSONA — Chat mode always has someone on screen: a character
// standing in the middle, facing the camera, breathing and shifting weight
// like a person actually standing there — not frozen, and not a raw motion
// clip snapping every time it loops. Two saved motions, both played on
// THREE.LoopRepeat and crossfaded between (no GPU call for either): a calm
// standing-idle loop while she's listening, a talking-and-gesturing loop
// while an answer is being delivered. Only an actual "teach me X" spends a
// fresh Kimodo generation, for the demo itself.
// ========================================================================
const CHAT_IDLE_MOTION_URL = 'assets/motions/persona_idle.bvh';
const CHAT_TALK_MOTION_URL = 'assets/motions/talking.bvh';
const CHAT_WAVE_MOTION_URL = 'assets/motions/persona_wave.bvh';
const CHAT_FACING_YAW = 0; // both saved motions already face +Z, where the camera sits
const PERSONA_LOOP_BLEND = 0.4;   // seconds eased back toward frame 0 — see loopifyClip
const PERSONA_CROSSFADE = 0.35;   // seconds to blend idle <-> talk
const PERSONA_SETTLE = 0.6;       // longer blend out of the hello, so the arm doesn't stop dead
// Her mixer runs at CHAR_TIME_SCALE (1.5x) because story beats want to be
// brisk; a greeting doesn't. Playing the wave at the reciprocal puts it back
// at the speed it was actually authored.
const PERSONA_WAVE_RATE = 1 / CHAR_TIME_SCALE;
let chatIdleBvh = null, chatTalkBvh = null, chatWaveBvh = null;

async function getChatMotionBvhs() {
    if (!chatIdleBvh) chatIdleBvh = await fetch(CHAT_IDLE_MOTION_URL).then(r => r.text());
    if (!chatTalkBvh) chatTalkBvh = await fetch(CHAT_TALK_MOTION_URL).then(r => r.text());
    // The greeting is a saved motion like the other two — generated once and
    // committed, never a GPU call. If it's missing she simply doesn't wave.
    if (!chatWaveBvh) chatWaveBvh = await fetch(CHAT_WAVE_MOTION_URL).then(r => r.ok ? r.text() : null).catch(() => null);
    return { idleBvh: chatIdleBvh, talkBvh: chatTalkBvh, waveBvh: chatWaveBvh };
}

// Chat framing. The greeting and the conversation captions both sit along the
// bottom of the viewport, so the persona is framed in the upper two-thirds:
// look at knee height and the body reads above the text instead of behind it.
// A demo needs room to move, so it pulls back.
function frameChat(wide) {
    camera.position.set(0, wide ? 190 : 115, wide ? 430 : 205);
    controls.target.set(0, wide ? 90 : 85, 0);
    controls.update();
}

// Make `target` the only one of her three motions carrying any weight,
// ramping the others down over the same window so the mixer blends
// joint-by-joint in between — smooth regardless of where in its own loop
// each clip happens to be. All three actions stay playing at all times;
// only weights move. Having one primitive (rather than a pairwise crossfade
// per pair) is what keeps a hello interrupted by a question from leaving two
// motions stacked on top of each other.
function personaPlayOnly(target, fade = PERSONA_CROSSFADE) {
    let anyLive = false;
    for (const a of [personaIdleAction, personaTalkAction, personaWaveAction]) {
        if (!a || a === target) continue;
        if (a.getEffectiveWeight() > 0.001) { a.fadeOut(fade); anyLive = true; } else a.setEffectiveWeight(0);
    }
    if (!target) return;
    target.enabled = true;
    target.setEffectiveTimeScale(target._personaTimeScale ?? 1);
    if (!target.isRunning()) target.play();
    if (!anyLive) {
        // Nothing else is driving the skeleton, so there's nothing to cross
        // from: take over at full weight this instant. Fading up from zero
        // here means a few frames with no weight at all anywhere, and with
        // no weight the bones sit in their raw rest pose — every bone along
        // +X, i.e. flat — so she'd appear to rise up off the floor.
        target.setEffectiveWeight(1);
        return;
    }
    // Base weight back to 1 BEFORE fading in. fadeIn only ramps a 0->1
    // *factor* that three.js multiplies by action.weight, so fading in an
    // action left at weight 0 (which is how the inactive ones are parked)
    // yields 0 forever — the greeting played silently, every weight sat at
    // 0, and with nothing driving the bones she collapsed into the skeleton's
    // raw rest pose and appeared to slide off across the floor.
    target.setEffectiveWeight(1);
    target.fadeIn(fade);
}

function setPersonaTalking(on) {
    if (on === personaTalking || !personaIdleAction || !personaTalkAction) { personaTalking = on; return; }
    personaTalking = on;
    personaGreetToken++; // a question mid-hello cancels the pending wave->idle hand-off
    if (on) personaTalkAction.time = 0; // she starts the gesture from its beginning each time she speaks
    personaPlayOnly(on ? personaTalkAction : personaIdleAction);
}

// She waves hello on arriving in Chat, then settles into the idle loop. The
// wave is LoopOnce and clamps on its last frame, so the hand-off to idle is
// a weight ramp out of a held pose rather than a snap back to frame 0.
function playPersonaGreeting(entry) {
    if (!personaWaveAction || !personaIdleAction || !entry) return;
    const token = ++personaGreetToken;
    personaTalking = false;
    personaWaveAction.reset();
    personaPlayOnly(personaWaveAction);
    const onFinished = (e) => {
        if (e.action !== personaWaveAction) return;
        entry.mixer.removeEventListener('finished', onFinished);
        if (token !== personaGreetToken) return; // superseded (she started talking, or greeted again)
        personaPlayOnly(personaIdleAction, PERSONA_SETTLE);
    };
    entry.mixer.addEventListener('finished', onFinished);
}

// Put the persona on stage (or return them to idle after a demo).
// keepTalking: the object-path demo re-spawns her mid-sentence, and she
// should carry on gesturing until the narration actually ends.
async function showChatPersona({ keepTalking = false, greet = false } = {}) {
    const { idleBvh, talkBvh, waveBvh } = await getChatMotionBvhs();
    await loadAvatarVRM().catch(err => console.warn('VRM avatar failed to load, falling back to the mannequin:', err));
    // Those awaits (a 14MB VRM on the first one) are long enough to switch
    // tabs during. Without this the persona would finish building into
    // whatever mode is on screen by then — landing her in Create's scene.
    if (document.body.dataset.mode !== 'help') return;
    clearAllCharacters();
    const entry = createCharacter(idleBvh, { label: 'Animo', yaw: CHAT_FACING_YAW, prompt: 'idle', avatar: avatarKey });
    syncPrimaryGlobals();
    frameChat(false);

    // createCharacter() armed entry.action as a LoopOnce clip; both her
    // motions loop continuously instead, smoothed at the seam so the wrap
    // doesn't show.
    loopifyClip(entry.mergedClip, PERSONA_LOOP_BLEND);
    personaIdleAction = entry.action;
    personaIdleAction.setLoop(THREE.LoopRepeat, Infinity);
    personaIdleAction.clampWhenFinished = false;

    const talkClip = loopifyClip(normalizeRootTravel(new BVHLoader().parse(talkBvh).clip), PERSONA_LOOP_BLEND);
    personaTalkAction = entry.mixer.clipAction(talkClip);
    personaTalkAction.setLoop(THREE.LoopRepeat, Infinity);
    personaTalkAction.clampWhenFinished = false;
    personaTalkAction.enabled = true;
    personaTalkAction.play();

    personaWaveAction = null;
    if (waveBvh) {
        const waveClip = normalizeRootTravel(new BVHLoader().parse(waveBvh).clip);
        personaWaveAction = entry.mixer.clipAction(waveClip);
        personaWaveAction.setLoop(THREE.LoopOnce, 1);
        personaWaveAction.clampWhenFinished = true; // hold the last frame so idle can fade in over it
        personaWaveAction.enabled = true;
        personaWaveAction._personaTimeScale = PERSONA_WAVE_RATE;
        personaWaveAction.setEffectiveWeight(0);
        personaWaveAction.play();
    }

    personaTalking = keepTalking;
    personaIdleAction.setEffectiveWeight(keepTalking ? 0 : 1);
    personaTalkAction.setEffectiveWeight(keepTalking ? 1 : 0);

    chatPersonaIdle = true;
    // Greeting comes after the weights above are settled, and never over a
    // demo hand-off (keepTalking) — she's mid-answer there, not arriving.
    if (greet && !keepTalking) playPersonaGreeting(entry);
    isPlaying = true;
    playbackFinished = false;
    recomputePlaybackDuration();
    updatePlayPauseIcon();
}

// Clears whatever the previous Help-mode demo left in the scene (character
// and/or a demoed object) before starting a fresh one.
function clearHelpDemo() {
    activeObjectAnimations = [];
    clearAllCharacters();
    personaIdleAction = null;  // going into a demo — showChatPersona() rebuilds these afterward
    personaTalkAction = null;
    personaWaveAction = null;
    if (helpDemoObject) { scene.remove(helpDemoObject); helpDemoObject = null; }
    document.getElementById('help-replay-btn').style.display = 'none';
}
document.getElementById('help-replay-btn').addEventListener('click', restartPlayback);

const HELP_INPUT_PLACEHOLDER = 'Ask anything, or teach me something...';
const helpInput = document.getElementById('help-input');

async function askHelp(question) {
    if (!question || !question.trim()) return;
    document.getElementById('help-welcome').style.display = 'none';
    addHelpBubble('user', question);
    // Same "disable + swap placeholder" pattern as Create's chat bar — the
    // panel is where the text lives now, so that's where "thinking" shows too.
    helpInput.disabled = true;
    helpInput.placeholder = 'Thinking...';
    warmEleven(); // ready the voice connection while the answer is being written

    // The persona keeps standing there and gesturing while it thinks — no
    // loading curtain for an ordinary reply, only for an actual demo below.
    try {
        const result = await callGeminiHelp(question);
        addHelpBubble('assistant', result.answer);
        speak(result.answer, () => setPersonaTalking(false), () => setPersonaTalking(true));

        if (result.demo_type === 'human_motion' && result.motion_prompt) {
            clearHelpDemo();
            showGenLoadingOverlay('Generating demo...');
            const skillName = question.replace(/^(teach me|how do i|how to|show me)\s*/i, '').trim() || question;
            rememberSkill(skillName).then(renderLearnedList);

            const { bvhText } = await fetchMotionBVH(result.motion_prompt, 6);
            chatPersonaIdle = false;          // a demo plays once, then holds
            frameChat(true);
            loadBVH(bvhText, '', avatarKey); // the demo is her too, not the mannequin; plays once, then pauses — see createPlaybackClock()
            if (characters[0]) characters[0].group.rotation.y = CHAT_FACING_YAW;
            if (mixer) mixer.timeScale = 0.6; // slower, easier to follow
            recomputePlaybackDuration(); // duration changed once timeScale was set above
            document.getElementById('help-replay-btn').style.display = 'flex';
            hideGenLoadingOverlay();

        } else if (result.demo_type === 'object_path' && result.object_keyword && result.path && result.path.length > 0) {
            showGenLoadingOverlay('Generating demo...');
            // Keep the persona on screen presenting it, and put the object beside them.
            await showChatPersona({ keepTalking: true });
            const glbUrl = await findLocalModel(result.object_keyword, null);
            if (glbUrl) {
                helpDemoObject = await loadGLBModel(glbUrl, [130, 0, 0], 50, 0);
                if (helpDemoObject) {
                    activeObjectAnimations = [{
                        object3D: helpDemoObject,
                        path: result.path,
                        duration: result.path[result.path.length - 1].t || 4,
                        basePos: helpDemoObject.position.clone(),
                        baseYaw: helpDemoObject.rotation.y,
                    }];
                    recomputePlaybackDuration();
                    isPlaying = true;
                    playbackFinished = false;
                    updatePlayPauseIcon();
                    document.getElementById('help-replay-btn').style.display = 'flex';
                } else {
                    log(`Couldn't load a model for "${result.object_keyword}"`, 'error');
                }
            } else {
                log(`No model available for "${result.object_keyword}" — answering without a demo`, 'system');
            }
            hideGenLoadingOverlay();

        } else {
            // Ordinary answer: just the persona talking. Costs no GPU call.
            //
            // keepTalking: true matters here. setPersonaTalking(true) already
            // ran above, but right after a demo chatPersonaIdle is false and
            // personaIdleAction/personaTalkAction are null (clearHelpDemo
            // nulled them), so that call was a no-op with nothing to
            // crossfade yet. showChatPersona() rebuilds them from scratch —
            // with keepTalking left at its false default it was resetting
            // personaTalking back to false and pinning her at idle weight 1,
            // which is exactly why she stayed stuck in the idle pose instead
            // of gesturing on the first ordinary question after a demo.
            if (!chatPersonaIdle || characters.length === 0) await showChatPersona({ keepTalking: true });
            hideGenLoadingOverlay();
        }
    } catch (err) {
        addHelpBubble('assistant', `Sorry — ${err.message}`);
        hideGenLoadingOverlay();
    }

    helpInput.disabled = false;
    helpInput.placeholder = HELP_INPUT_PLACEHOLDER;
    helpInput.focus();
}

// Wire up Help mode input — one box now, in the left panel, for both the
// opening question and every follow-up.
helpInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { const v = helpInput.value.trim(); if (v) { helpInput.value = ''; askHelp(v); } }
});
document.querySelectorAll('#create-welcome .w-ex').forEach(card => {
    card.addEventListener('click', () => handleChat(card.dataset.prompt));
});

// ========================================================================
// MODE TOGGLE — Help / Create share the same engine, different chrome.
// ========================================================================
const modeHelpBtn = document.getElementById('mode-help');
const modeCreateBtn = document.getElementById('mode-create');

const tabChat = document.getElementById('tab-chat');
const tabLearned = document.getElementById('tab-learned');
const chatTabContent = document.getElementById('chat-tab-content');
const learnedTabContent = document.getElementById('learned-tab-content');

function switchHelpTab(tab) {
    if (tab === 'chat') {
        tabChat.classList.add('active'); tabLearned.classList.remove('active');
        chatTabContent.style.display = 'flex'; learnedTabContent.style.display = 'none';
    } else {
        tabLearned.classList.add('active'); tabChat.classList.remove('active');
        chatTabContent.style.display = 'none'; learnedTabContent.style.display = 'flex';
        renderLearnedList();
    }
}
tabChat.addEventListener('click', () => switchHelpTab('chat'));
tabLearned.addEventListener('click', () => switchHelpTab('learned'));

// ========================================================================
// PER-MODE SCENE SEPARATION — Create and Chat each keep their own scene
// contents and playback state. Switching modes snapshots+hides whichever
// you're leaving and restores whichever you're entering, so Create's
// scene is never affected by what Chat is doing and vice versa.
// ========================================================================
let createSnapshot = null;
let chatSnapshot = null;

function captureSceneSnapshot() {
    return {
        sceneObjects: sceneObjects.slice(), groundMesh, characterGroup, currentBones,
        bodyMeshes: bodyMeshes.slice(), helpDemoObject,
        mixer, currentAction, currentClip, characters,
        activeObjectAnimations, objectMixer, objectClockAction,
        totalDuration, timelineClips: timelineClips.slice(),
        playbackFinished,
        // The camera is per-mode too. Without it, coming back from Chat left
        // Create looking at wherever frameChat had pointed — the scene was
        // fully restored and visible, just off screen, which read as "my
        // character disappeared" until you hit Reset View.
        cameraPos: camera.position.clone(), cameraTarget: controls.target.clone(),
    };
}

function setSceneVisible(snap, visible) {
    if (!snap) return;
    const toggle = (o) => { if (o) o.visible = visible; };
    snap.sceneObjects.forEach(toggle);
    snap.bodyMeshes.forEach(toggle);
    toggle(snap.groundMesh);
    toggle(snap.characterGroup);
    toggle(snap.helpDemoObject);
    snap.characters.forEach(c => { toggle(c.group); c.bodyMeshesArr.forEach(toggle); });
}

function restoreSceneVars(snap) {
    sceneObjects = snap.sceneObjects; groundMesh = snap.groundMesh;
    characterGroup = snap.characterGroup; currentBones = snap.currentBones;
    bodyMeshes = snap.bodyMeshes;
    helpDemoObject = snap.helpDemoObject; characters = snap.characters;
    mixer = snap.mixer; currentAction = snap.currentAction; currentClip = snap.currentClip;
    activeObjectAnimations = snap.activeObjectAnimations;
    objectMixer = snap.objectMixer; objectClockAction = snap.objectClockAction;
    totalDuration = snap.totalDuration; timelineClips = snap.timelineClips;
    playbackFinished = snap.playbackFinished;
    if (snap.cameraPos) { camera.position.copy(snap.cameraPos); controls.target.copy(snap.cameraTarget); controls.update(); }
    isPlaying = false; // land paused either way; hit play/replay to resume
    updatePlayPauseIcon();
}

// Whether Create has anything on stage. The timeline alone is the wrong test:
// removing the last clip leaves every prop standing (see finishTimelineEdit),
// and a stage full of props is still a scene — showing "Animo 1.0" over it was
// the bug. Ground counts too, since a built scene always has one.
function createHasContent() {
    return timelineClips.length > 0 || sceneObjects.length > 0
        || characters.length > 0 || !!groundMesh;
}

// The welcome overlay is purely an empty-state; every site that used to set
// its display by hand now goes through here so they cannot disagree.
function syncCreateWelcome() {
    const el = document.getElementById('create-welcome');
    if (el) el.style.display = createHasContent() ? 'none' : 'flex';
}

function resetToBlankScene() {
    sceneObjects = []; groundMesh = null; characterGroup = null; currentBones = null;
    bodyMeshes = []; helpDemoObject = null;
    mixer = null; currentAction = null; currentClip = null;
    activeObjectAnimations = []; objectBeats = [];
    if (objectMixer) objectMixer.stopAllAction();
    objectMixer = null; objectClockAction = null;
    totalDuration = 0; timelineClips = [];
    isPlaying = false; playbackFinished = false;
    characters = [];
    currentSceneType = 'outdoor';
    createSceneStarted = false;
    createFog = null;
    scene.fog = null;
    document.getElementById('timeline-bar')?.classList.remove('visible');
    document.getElementById('create-controls')?.classList.remove('visible');
}

function setMode(mode) {
    const leaving = document.body.dataset.mode;
    if (leaving === mode) return;

    // Snap the panel to the new mode's width instead of sliding it there: the
    // slide swept the whole layout sideways every time you switched tabs.
    const panel = document.getElementById('panel');
    panel.classList.add('instant');

    // Snapshot + hide (not remove — cheap to bring back) whichever mode we're leaving.
    if (leaving === 'create') {
        createSnapshot = captureSceneSnapshot();
        setSceneVisible(createSnapshot, false);
    } else if (leaving === 'help') {
        chatSnapshot = captureSceneSnapshot();
        setSceneVisible(chatSnapshot, false);
    }

    document.body.dataset.mode = mode;
    modeHelpBtn.classList.toggle('active', mode === 'help');
    modeCreateBtn.classList.toggle('active', mode === 'create');

    // Restore (or start blank) whichever mode we're entering.
    const entering = mode === 'create' ? createSnapshot : chatSnapshot;
    if (entering) {
        restoreSceneVars(entering);
        setSceneVisible(entering, true);
        if (mode === 'create' && timelineClips.length > 0) { showTimeline(); renderTimelineClips(); }
    } else {
        resetToBlankScene();
    }

    enterMode(mode);
    void panel.offsetWidth; // lay out the new width while transitions are still off
    panel.classList.remove('instant');
}

// The per-mode entry work, split out of setMode because startup needs it too:
// <body> already carries the opening mode, so setMode would early-return
// before ever reaching this.
function enterMode(mode) {
    if (mode === 'help') {
        setCreateLook(false);
        applySkinTone(false);
        // Create's Library pane isn't mode-scoped, so one left open showed
        // through under Chat's own tabs — put Create's panel back first.
        switchTab('activity');
        grid.visible = true; // Chat's neutral stage has no groundMesh of its own — always show the plain floor
        // Chat is never an empty stage: the persona is always standing there,
        // and she waves hello every time you arrive. If she's still standing
        // from last visit, reuse her (re-skinning the VRM is not cheap) and
        // just replay the greeting. Reviving isPlaying matters: restoreSceneVars
        // parks every restored scene paused, which for a looping persona meant
        // she came back frozen — the "stuck coming from Create" bug.
        if (chatPersonaIdle && personaIdleAction && characters.length > 0) {
            isPlaying = true;
            playbackFinished = false;
            updatePlayPauseIcon();
            playPersonaGreeting(characters[0]);
        } else {
            showChatPersona({ greet: true });
        }
        const started = chatHasHistory();
        document.getElementById('help-welcome').style.display = started ? 'none' : 'flex';
        syncChatGreeting();
        // Arriving always lands on the greeting persona, so there's no demo
        // left to replay — a stale Replay button here would restart her loop.
        document.getElementById('help-replay-btn').style.display = 'none';
        switchHelpTab('chat');
        document.getElementById('panel').classList.remove('collapsed');
        stopSpeaking();
    } else {
        setCreateLook(true);
        applySkinTone(true);
        grid.visible = !groundMesh; // Create mode: grid only shows before its own ground is built
        // The panel's chat bar is Create's only input now, so it can't start collapsed.
        document.getElementById('panel').classList.remove('collapsed');
        document.getElementById('panel-reopen').classList.remove('visible');
        switchTab('activity');
        syncCreateWelcome();
    }
}
for (const b of document.querySelectorAll('.avatar-btn')) {
    b.addEventListener('click', () => setAvatarKey(b.dataset.avatar));
}
modeHelpBtn.addEventListener('click', () => setMode('help'));
modeCreateBtn.addEventListener('click', () => setMode('create'));

// Initial paint
renderReferenceLibrary();
renderLearnedList();

// Open on Chat rather than an empty Create stage: the persona is already
// standing there and waves, so there's something on screen before you've
// typed anything.
enterMode('help');
