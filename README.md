# SignSpeak — BSL Practice Guide

A browser-based British Sign Language (BSL) practice tool. SignSpeak uses
[MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic) to
detect full‑body / hand landmarks in real time, then classifies a curated set
of common signs with an embedded geometric / decision‑tree classifier — no
server, no model uploads, no data collection. **Everything runs locally in
your browser.**

## Live demo

Once deployed on Vercel, the app will be available at the project's
`*.vercel.app` URL.

## Detected signs

The live classifier currently recognises:

`hello`, `goodbye`, `yes`, `no`, `please`, `thank you`, `sorry`, `help`,
`eat`, `drink`, `mother`, `father`, `stop`, `want`, `friend`, `work`,
`school`, `family`.

The in‑app **BSL guide** also includes ~100 additional common BSL signs as a
visual reference (not live‑detected — included as a learning aid).

## Running locally

The project is plain HTML/CSS/JS — no build step required. Any static file
server works:

```bash
# Python
python -m http.server 8000

# Node
npx serve .

# Or just open index.html via a local server in your editor
```

Then open <http://localhost:8000>.

> Camera access requires a secure context (HTTPS or `localhost`). Opening the
> HTML file via the `file://` protocol will block the camera.

## How it works

- **MediaPipe Holistic** provides per‑frame landmarks for hands, face, and
  pose.
- A custom classifier in `app.js` (`classifyBSLWordBothHands`) maps the
  landmarks into geometric features (finger extension, pinch distance, wrist
  vs face‑landmark zones) and scores each candidate sign.
- A temporal smoothing pipeline (rolling‑window majority vote +
  gesture‑hold timer + motion‑velocity gate + cooldown) prevents flickery
  detections and double‑commits.

See the comments in `app.js` for the precise thresholds and per‑sign rules.

## Project structure

```
.
├── index.html      # App shell + UI
├── style.css       # All styling
├── app.js          # MediaPipe glue + classifier + UI state
├── vercel.json     # Static‑site config for Vercel
└── README.md
```

## Tech

- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic)
- Vanilla JavaScript, HTML, CSS
- No build tools, no frameworks
