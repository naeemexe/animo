// Animo — browser configuration.
//
// Copy this file to local-config.js (gitignored) and fill in what you use:
//
//   cp local-config.example.js local-config.js
//
// local-config.js loads before app.js. Everything here is read by the
// browser, so only put keys here that are meant for client-side use.
// Server-side secrets belong in .env (see .env.example).

// ---- Motion server ---------------------------------------------------------
// backend/motion_server.py on a GPU box (RunPod or Vultr). Leave the default
// if you tunnel it to this machine; without a reachable server, Create falls
// back to the bundled sample motions in assets/motions/.
window.ANIMO_API_URL = 'http://localhost:8000';

// ---- Gemini (scene planning, the director, Chat answers) — pick one --------
// A) Your Google Cloud project through backend/gemini_proxy.py (no key in the
//    browser). Used whenever it's set.
window.ANIMO_GEMINI_PROXY_URL = '';          // e.g. 'http://localhost:8010'
// B) An AI Studio API key (https://aistudio.google.com/apikey).
window.ANIMO_GEMINI_KEY = '';

// ---- ElevenLabs (optional) -------------------------------------------------
// Real voices for the Chat characters, one per character. Without a key they
// still speak using the browser's built-in voices.
window.ANIMO_ELEVENLABS_KEY = '';
// Pin exact voices (Voice Library -> a voice -> ID); otherwise defaults are used.
// window.ANIMO_ELEVENLABS_VOICES = { male: 'voice_id', female: 'voice_id' };

// ---- Auth0 (planned — sign-in isn't wired up yet) --------------------------
// Public SPA settings from your Auth0 application. See README -> Integrations.
// window.ANIMO_AUTH0_DOMAIN = 'your-tenant.us.auth0.com';
// window.ANIMO_AUTH0_CLIENT_ID = '';
// window.ANIMO_AUTH0_AUDIENCE = '';
