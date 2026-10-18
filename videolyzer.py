#!/usr/bin/env python3
"""Videolyzer - give agents the ability to understand and deep analyze videos.

Usage:
    videolyzer.py <youtube-url-or-file> [--provider gemini] [--out report.md]
        [--model gemini-3.1-flash-lite] [--base-url URL] [--api-key KEY]
        [--prompt "..."] [--no-motion] [--no-cuts] [--cut-threshold 0.4]
        [--keyframes 8]
        [--aux-model NAME] [--aux-base-url URL] [--aux-api-key KEY] [--fast]

Providers:
    gemini  - Google AI Studio. Native video understanding, YouTube URLs
              supported without download. Local files go through the Files
              API (full upload, model samples the timeline itself).
              (default, key via VIDEOLYZER_API_KEY, GEMINI_API_KEY or --api-key)
    openai  - Any OpenAI-compatible API (OpenAI, Groq, Together, etc.).
              Local files only: keyframes are extracted with ffmpeg and sent
              as images through /v1/chat/completions (vision).
              URL input is not supported with this provider.
              (key via VIDEOLYZER_API_KEY, OPENAI_API_KEY or --api-key,
               endpoint via --base-url, default https://api.openai.com/v1)

Auxiliary model (--aux-model):
    Any OpenAI-compatible vision endpoint used as a frame-by-frame analysis
    layer. Roles:
      - Second opinion: cross-check the scenes the primary model reported.
      - Temporal verifier: confirm event order across keyframes.
      - Fallback: automatically used (keyframes) when the primary analysis
        fails, if configured.
    Configured via --aux-model NAME, --aux-base-url URL, --aux-api-key KEY,
    or env vars VIDEOLYZER_AUX_MODEL, VIDEOLYZER_AUX_BASE_URL,
    VIDEOLYZER_AUX_API_KEY. No provider is hardcoded; any endpoint speaking
    the OpenAI chat-completions dialect with vision support works.

Modo Rapido (--fast):
    Skips the full video upload and runs the whole analysis on the
    auxiliary model with evenly spaced keyframes (+ the optical-flow
    subtle-motion hints). Much faster and cheaper than the native path;
    best when only "what appears" matters and fine motion detail is not
    critical. Requires the auxiliary model to be configured.

Modes:
    URL (http, gemini only): the URL is sent straight to the model
        (file_data), no download. Fast. No subtle-motion detection and
        no cut detection (both would require downloading the video).
    Local file: validated with ffprobe, then analyzed natively.
        With --provider gemini the whole file is uploaded and the model
        does its own temporal sampling (no frame extraction by us).
        Before upload, two deterministic passes run locally:
          - cut detection (ffmpeg scene detection, --cut-threshold):
            exact hard-cut timestamps go into the prompt as anchors so
            the model aligns scene boundaries on them instead of guessing;
          - subtle-motion pass (Farneback optical flow): flags segments
            with small-amplitude motion above background noise; the
            timestamps go into the prompt as an explicit hint
            (hybrid two-pass).
        With --provider openai, evenly spaced keyframes are extracted
        and sent as images; both local passes still run and feed the
        same hint sections into the prompt.

Output: Markdown report with detected scenes (timestamps), transcript
aligned per scene, key moments, tone/emotion, and temporal verification.

Zero pip dependencies for the core path (stdlib + ffmpeg/ffprobe).
opencv-python + numpy only needed for the subtle-motion pass.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import subprocess  # nosec B404 - used only for local ffmpeg/ffprobe with fixed args
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
OPENAI_API_BASE = "https://api.openai.com/v1"
MAX_RETRIES = 3
RETRY_DELAYS = [2, 5, 10]
KEYFRAME_COUNT = 8
# Auxiliary model (any OpenAI-compatible vision endpoint). Nothing hardcoded.
AUX_ENV_MODEL = "VIDEOLYZER_AUX_MODEL"
AUX_ENV_BASE_URL = "VIDEOLYZER_AUX_BASE_URL"
AUX_ENV_API_KEY = "VIDEOLYZER_AUX_API_KEY"

# Resolve full binary paths (bandit B607: never exec via partial PATH lookup).
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

ANALYZE_PROMPT = """Analyze this video in depth and return a Markdown report with the following sections:

# Video Analysis

## Summary
2-3 sentences about the overall content of the video.

## Scenes
Table with each detected scene:
| # | Start | End | Visual description | Audio/speech |
Use mm:ss timestamps. Detect scene changes by visual change (place, angle, action).

## Aligned transcript
Faithful transcript of speech, organized by scene/timestamp. If there is no speech, say "no dialogue".

## Key moments
List of the 3-5 most important moments with timestamp and why.

## Tone and emotion
Overall tone, dominant emotions, pacing (slow/fast), and how that serves the video's purpose.

## Temporal verification
For each scene with visible motion, describe the MOTION (action verb: crosses, spins, falls, approaches, etc.), not just the static state. If there is a cause-effect sequence (someone picks up an object AND THEN uses it), confirm the temporal order of events. If some motion cannot be determined, say explicitly "motion undetermined" instead of inventing it.

Return ONLY the Markdown, no extra commentary."""

MOTION_HINT_SECTION = """
## Subtle motion detected (optical flow, automatic pass)
The segments below contain small-amplitude motion CONFIRMED by optical-flow
measurement. Describe EXACTLY what moves in each segment: which object,
direction, approximate amplitude and rhythm. Do NOT say these segments are
static. If the motion is ambiguous, describe what the measurement indicates
without inventing a cause.
{items}
"""

CUT_HINT_SECTION = """
## Detected cuts (deterministic, ffmpeg scene detection)
The hard cuts below were detected by frame-difference analysis and are
EXACT, not approximate. Anchor your scene boundaries on these timestamps:
a new scene starts at each cut timestamp. Do NOT merge scenes across a cut,
and do NOT invent scene boundaries between cuts unless there is a clear
visual change (camera angle, location) without a hard cut.
{items}
"""


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def get_api_key(args) -> str:
    for src in (args.api_key, os.environ.get("VIDEOLYZER_API_KEY"),
                os.environ.get("GEMINI_API_KEY"),
                os.environ.get("OPENAI_API_KEY")):
        if src and src.strip():
            return src.strip()
    print(
        "error: no API key. Use --api-key or set VIDEOLYZER_API_KEY "
        "(or GEMINI_API_KEY / OPENAI_API_KEY).",
        file=sys.stderr,
    )
    sys.exit(3)


def get_aux_config(args) -> dict | None:
    """Auxiliary multimodal model config. None when not configured.

    Accepts any OpenAI-compatible vision endpoint; nothing provider-specific.
    Precedence: flags > VIDEOLYZER_AUX_* env vars.
    """
    model = args.aux_model or os.environ.get(AUX_ENV_MODEL)
    if not model or not model.strip():
        return None
    base_url = (args.aux_base_url or os.environ.get(AUX_ENV_BASE_URL)
                or OPENAI_API_BASE).rstrip("/")
    api_key = args.aux_api_key or os.environ.get(AUX_ENV_API_KEY)
    if not api_key or not api_key.strip():
        print(
            f"error: --aux-model set but no aux API key. Use --aux-api-key "
            f"or set {AUX_ENV_API_KEY}.",
            file=sys.stderr,
        )
        sys.exit(3)
    return {"model": model.strip(), "base_url": base_url,
            "api_key": api_key.strip()}


def fmt_ts(s: float) -> str:
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def http_json(url: str, headers: dict, data: bytes | None = None,
              timeout: int = 600, method: str = "POST") -> dict | None:
    """Request with retry on 429/5xx. Returns parsed JSON or None."""
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        print(f"error: refusing non-HTTP(S) URL scheme: {scheme}",
              file=sys.stderr)
        return None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 - scheme validated to http/https above
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                last_err = e
            else:
                body = e.read().decode(errors="replace")[:300]
                print(f"error: API {e.code}: {body}", file=sys.stderr)
                return None
        except Exception as e:  # noqa: BLE001
            last_err = e
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAYS[attempt])
    print(f"warn: failed after {MAX_RETRIES} tries: {last_err}", file=sys.stderr)
    return None


# ---------------- subtle-motion detection (provider-agnostic) ----------------

def detect_subtle_motion(path: str, sample_fps: float = 2.0) -> list:
    """Pass 1: optical flow (Farneback) flags segments with subtle motion.

    Samples frame pairs, computes mean flow magnitude per cell (3x3 grid)
    and flags segments where the most active cell beats background noise
    (median + 4*1.4826*MAD, robust to outliers, with absolute floor).
    Returns list of {start, end, region, magnitude, axis}. Fails silent.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("note: opencv unavailable, skipping subtle-motion detection",
              file=sys.stderr)
        return []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("note: could not open video for optical flow", file=sys.stderr)
        return []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / sample_fps)))
        W = 320
        samples = []  # (t_s, max_cell_mag, (row, col), mean|vx|, mean|vy|)
        prev_gray = None
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (W, max(1, int(h * W / w))))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                t = idx / fps
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    # cv2 stubs don't cover flow=None (standard allocate-output
                    # idiom); verified correct at runtime
                    flow = cv2.calcOpticalFlowFarneback(  # pyright: ignore[reportCallIssue]
                        prev_gray, gray, None,  # pyright: ignore[reportArgumentType]
                        0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    gh, gw = mag.shape
                    ch, cw = max(1, gh // 3), max(1, gw // 3)
                    best = (0.0, (1, 1), 0.0, 0.0)
                    for r in range(3):
                        for c in range(3):
                            cell = mag[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
                            m = float(cell.mean()) if cell.size else 0.0
                            if m > best[0]:
                                fx = flow[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw, 0]
                                fy = flow[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw, 1]
                                best = (m, (r, c),
                                        float(np.abs(fx).mean()),
                                        float(np.abs(fy).mean()))
                    samples.append((t, best[0], best[1], best[2], best[3]))
                prev_gray = gray
            idx += 1
    finally:
        cap.release()

    if len(samples) < 3:
        return []
    mags = np.array([a[1] for a in samples])
    base = float(np.median(mags))
    # MAD instead of std: std is inflated by the outliers (the signal
    # itself), which would push the threshold above the signal. MAD is robust.
    mad = float(np.median(np.abs(mags - base)))
    threshold = base + max(4 * 1.4826 * mad, 0.25)

    blocks = []
    start_i = None
    last = -10
    for i, (t, m, _cell, _vx, _vy) in enumerate(samples):
        if m >= threshold:
            if start_i is None:
                start_i = i
            elif i - last > 2:
                blocks.append((start_i, last))
                start_i = i
            last = i
    if start_i is not None:
        blocks.append((start_i, last))

    names = {
        (0, 0): "top-left corner", (0, 1): "top-center third",
        (0, 2): "top-right corner", (1, 0): "middle-left third",
        (1, 1): "center", (1, 2): "middle-right third",
        (2, 0): "bottom-left corner", (2, 1): "bottom-center third",
        (2, 2): "bottom-right corner",
    }
    dt = step / fps
    result = []
    for a, b in blocks:
        t0 = max(0.0, samples[a][0] - dt)
        t1 = samples[b][0] + dt
        regs = [samples[i][2] for i in range(a, b + 1)]
        dom = max(set(regs), key=regs.count)
        mag_mean = float(np.mean([samples[i][1] for i in range(a, b + 1)]))
        vxm = float(np.mean([samples[i][3] for i in range(a, b + 1)]))
        vym = float(np.mean([samples[i][4] for i in range(a, b + 1)]))
        if vym > 2 * vxm:
            axis = "vertical"
        elif vxm > 2 * vym:
            axis = "horizontal"
        else:
            axis = "mixed"
        result.append({
            "start": t0, "end": t1,
            "region": names[dom],
            "magnitude": round(mag_mean, 3),
            "axis": axis,
        })
    return result


def build_prompt(segments: list, base_prompt: str, cuts: list | None = None) -> str:
    prompt = base_prompt
    if cuts:
        items = "\n".join(f"- cut at {fmt_ts(c)} ({c:.2f}s)" for c in cuts)
        prompt = prompt + CUT_HINT_SECTION.format(items=items)
    if not segments:
        return prompt
    items = "\n".join(
        f"- {fmt_ts(s['start'])}-{fmt_ts(s['end'])} (region: {s['region']}, "
        f"dominant motion axis: {s['axis']}, "
        f"mean magnitude {s['magnitude']} px/frame)"
        for s in segments
    )
    return prompt + MOTION_HINT_SECTION.format(items=items)


# ---------------- deterministic cut detection (provider-agnostic) ----------------

def detect_cuts(path: str, threshold: float = 0.4) -> list:
    """Detect hard cuts via ffmpeg scene detection.

    Runs the `select` filter with `gt(scene,threshold)` and parses the
    `showinfo` pts_time of each selected frame. Returns sorted timestamps
    in seconds. Fails silent (returns []) on ffmpeg error or if ffmpeg
    is not installed.
    """
    if not FFMPEG:
        return []
    r = subprocess.run(  # nosec B603 - args are fixed strings, no shell
        [FFMPEG, "-v", "info", "-i", path,
         "-filter:v", f"select='gt(scene,{threshold})',showinfo",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    cuts = []
    for line in (r.stderr or "").splitlines():
        if "showinfo" not in line:
            continue
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            cuts.append(float(m.group(1)))
    # Drop the t=0 selection (first frame always "selected") and dedupe.
    cuts = sorted({c for c in cuts if c > 0.05})
    return cuts


# ---------------- Gemini provider ----------------

def gemini_generate(base: str, key: str, model: str, parts: list) -> str | None:
    url = f"{base}/models/{model}:generateContent?key={key}"
    data = http_json(url, {"Content-Type": "application/json"},
                     json.dumps({"contents": [{"parts": parts}]}).encode())
    if not data:
        return None
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        print(f"error: unexpected response: {json.dumps(data)[:300]}",
              file=sys.stderr)
        return None


def gemini_upload(base: str, key: str, path: str, mime: str) -> str | None:
    """Direct upload to the Gemini Files API. Returns the file name."""
    with open(path, "rb") as f:
        body = f.read()
    url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={key}"
    data = http_json(url, {"Content-Type": mime}, body, timeout=600)
    if not data:
        return None
    try:
        return data["file"]["name"]
    except (KeyError, TypeError):
        print(f"error: upload failed: {json.dumps(data)[:300]}",
              file=sys.stderr)
        return None


def gemini_wait_active(base: str, key: str, file_name: str,
                       timeout_s: int = 300) -> str | None:
    """Wait for the file to leave PROCESSING. Returns the file URI."""
    url = f"{base}/{file_name}?key={key}"
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        print(f"error: refusing non-HTTP(S) URL scheme: {scheme}",
              file=sys.stderr)
        return None
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310 - scheme validated above
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            # transient 400 while the file is still being indexed
            print(f"note: GET {e.code}, retrying...", file=sys.stderr)
            time.sleep(10)
            continue
        except Exception as e:  # noqa: BLE001
            print(f"note: {e}, retrying...", file=sys.stderr)
            time.sleep(10)
            continue
        state = data.get("state")
        if state == "ACTIVE":
            return data["uri"]
        if state == "FAILED":
            print(f"error: file processing failed: {json.dumps(data)[:300]}",
                  file=sys.stderr)
            return None
        print(f"note: file {state}, waiting...", file=sys.stderr)
        time.sleep(10)
    print("error: timeout waiting for file processing", file=sys.stderr)
    return None


def validate_video(path: str) -> dict:
    """ffprobe: validate codec, duration, streams, integrity. No frame extraction."""
    if not FFPROBE:
        raise RuntimeError("ffprobe not found in PATH")
    r = subprocess.run(  # nosec B603 - args are fixed strings, no shell
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"ffprobe failed (invalid or corrupt file): {r.stderr[:200]}")
    info = json.loads(r.stdout)
    streams = info.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not video:
        raise RuntimeError("file has no video stream")
    fmt = info.get("format", {})
    return {
        "duration": float(fmt.get("duration", 0)),
        "size": int(fmt.get("size", 0)),
        "video_codec": video[0].get("codec_name"),
        "has_audio": bool(audio),
    }


# ---------------- OpenAI-compatible provider ----------------

def extract_keyframes(path: str, count: int, duration: float) -> list:
    """Extract `count` evenly spaced JPEG keyframes. Returns temp paths."""
    if not FFMPEG:
        raise RuntimeError("ffmpeg not found in PATH")
    paths = []
    for i in range(count):
        t = duration * (i + 0.5) / count if duration > 0 else i
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        r = subprocess.run(  # nosec B603 - args are fixed strings, no shell
            [FFMPEG, "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", path,
             "-frames:v", "1", "-vf", "scale=1024:-1", "-q:v", "4", tmp.name],
            capture_output=True,
        )
        if r.returncode == 0 and os.path.getsize(tmp.name) > 0:
            paths.append(tmp.name)
        else:
            os.unlink(tmp.name)
    return paths


def openai_chat(base: str, key: str, model: str, messages: list) -> str | None:
    url = f"{base.rstrip('/')}/chat/completions"
    payload = {"model": model, "messages": messages, "max_tokens": 4096}
    data = http_json(url,
                     {"Content-Type": "application/json",
                      "Authorization": f"Bearer {key}"},
                     json.dumps(payload).encode())
    if not data:
        return None
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        print(f"error: unexpected response: {json.dumps(data)[:300]}",
              file=sys.stderr)
        return None


# ---------------- analysis flows ----------------

def analyze_url_gemini(base: str, key: str, model: str, url: str,
                       prompt: str) -> str | None:
    # URL mode: no download, no optical-flow pass (documented limitation).
    parts = [{"text": prompt}, {"file_data": {"file_uri": url}}]
    return gemini_generate(base, key, model, parts)


def analyze_file_gemini(base: str, key: str, model: str, path: str,
                        prompt: str, want_motion: bool,
                        cut_threshold: float | None) -> str | None:
    info = validate_video(path)
    print(f"validated: {info['video_codec']}, {info['duration']:.0f}s, "
          f"audio={'yes' if info['has_audio'] else 'no'}", file=sys.stderr)
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "video/mp4"
    print("uploading full video via Files API (native model sampling)...",
          file=sys.stderr)
    file_name = gemini_upload(base, key, path, mime)
    if not file_name:
        return None
    print(f"upload ok: {file_name}, waiting for processing...", file=sys.stderr)
    file_uri = gemini_wait_active(base, key, file_name)
    if not file_uri:
        return None
    cuts = []
    if cut_threshold is not None:
        print("detecting hard cuts (ffmpeg scene detection)...",
              file=sys.stderr)
        cuts = detect_cuts(path, cut_threshold)
        print(f"  {len(cuts)} cuts detected", file=sys.stderr)
    segments = []
    if want_motion:
        print("pass 1: optical flow looking for subtle motion...",
              file=sys.stderr)
        segments = detect_subtle_motion(path)
        for s in segments:
            print(f"  subtle motion: {fmt_ts(s['start'])}-{fmt_ts(s['end'])} "
                  f"({s['region']}, mag {s['magnitude']})", file=sys.stderr)
        if not segments:
            print("  no subtle motion above noise floor", file=sys.stderr)
    parts = [{"text": build_prompt(segments, prompt, cuts)},
             {"file_data": {"file_uri": file_uri, "mime_type": mime}}]
    return gemini_generate(base, key, model, parts)


def analyze_file_openai(base: str, key: str, model: str, path: str,
                        prompt: str, want_motion: bool,
                        keyframes: int,
                        cut_threshold: float | None) -> str | None:
    info = validate_video(path)
    print(f"validated: {info['video_codec']}, {info['duration']:.0f}s, "
          f"audio={'yes' if info['has_audio'] else 'no'}", file=sys.stderr)
    print(f"extracting {keyframes} keyframes...", file=sys.stderr)
    kf_paths = extract_keyframes(path, keyframes, info["duration"])
    if not kf_paths:
        print("error: could not extract keyframes", file=sys.stderr)
        return None
    cuts = []
    if cut_threshold is not None:
        print("detecting hard cuts (ffmpeg scene detection)...",
              file=sys.stderr)
        cuts = detect_cuts(path, cut_threshold)
        print(f"  {len(cuts)} cuts detected", file=sys.stderr)
    segments = []
    if want_motion:
        print("pass 1: optical flow looking for subtle motion...",
              file=sys.stderr)
        segments = detect_subtle_motion(path)
    content: list[dict] = [{"type": "text", "text": build_prompt(segments, prompt, cuts)}]
    try:
        for i, kf in enumerate(kf_paths):
            with open(kf, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            content.append({
                "type": "text",
                "text": f"[frame {i + 1}/{len(kf_paths)}]",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        return openai_chat(base, key, model,
                           [{"role": "user", "content": content}])
    finally:
        for kf in kf_paths:
            try:
                os.unlink(kf)
            except OSError:
                pass


# ---------------- auxiliary model (universal, provider-agnostic) ----------------

AUX_FAST_PREAMBLE = """\
You are analyzing a video through evenly spaced still frames (keyframes). \
Treat the frames as a temporal sequence: describe what CHANGES between them, \
verify the order of events, and call out motion explicitly. This is the \
fast analysis path; be concise but do not skip the temporal verification \
section.\
"""


def analyze_aux_fast(aux: dict, path: str, prompt: str, want_motion: bool,
                     keyframes: int,
                     cut_threshold: float | None) -> str | None:
    """Modo Rapido: whole analysis on the auxiliary model via keyframes.

    No full upload, no native temporal sampling. The optical-flow pass
    still runs so subtle motion gets flagged into the prompt. Works with
    any OpenAI-compatible vision endpoint.
    """
    info = validate_video(path)
    print(f"validated: {info['video_codec']}, {info['duration']:.0f}s, "
          f"audio={'yes' if info['has_audio'] else 'no'}", file=sys.stderr)
    print(f"Modo Rapido: {keyframes} keyframes -> aux model "
          f"'{aux['model']}'...", file=sys.stderr)
    kf_paths = extract_keyframes(path, keyframes, info["duration"])
    if not kf_paths:
        print("error: could not extract keyframes", file=sys.stderr)
        return None
    cuts = []
    if cut_threshold is not None:
        print("detecting hard cuts (ffmpeg scene detection)...",
              file=sys.stderr)
        cuts = detect_cuts(path, cut_threshold)
        print(f"  {len(cuts)} cuts detected", file=sys.stderr)
    segments = []
    if want_motion:
        print("pass 1: optical flow looking for subtle motion...",
              file=sys.stderr)
        segments = detect_subtle_motion(path)
        for s in segments:
            print(f"  subtle motion: {fmt_ts(s['start'])}-{fmt_ts(s['end'])} "
                  f"({s['region']}, mag {s['magnitude']})", file=sys.stderr)
    full_prompt = AUX_FAST_PREAMBLE + build_prompt(segments, prompt, cuts)
    content: list[dict] = [{"type": "text", "text": full_prompt}]
    try:
        for i, kf in enumerate(kf_paths):
            with open(kf, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            content.append({
                "type": "text",
                "text": f"[frame {i + 1}/{len(kf_paths)}]",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        return openai_chat(aux["base_url"], aux["api_key"], aux["model"],
                           [{"role": "user", "content": content}])
    finally:
        for kf in kf_paths:
            try:
                os.unlink(kf)
            except OSError:
                pass


def analyze_aux_verify(aux: dict, path: str, primary_report: str,
                       keyframes: int) -> str | None:
    """Second opinion: the aux model cross-checks the primary report.

    Sends keyframes plus the primary report and asks for agreement /
    disagreement per scene, focusing on temporal claims. Returns the
    verdict text or None.
    """
    info = validate_video(path)
    kf_paths = extract_keyframes(path, keyframes, info["duration"])
    if not kf_paths:
        return None
    verify_prompt = (
        "You are a second-opinion reviewer. Below is a video analysis report "
        "produced by another model from evenly spaced keyframes of the same "
        "video. Review each scene claim against the frames: say which claims "
        "you AGREE with, which you DISAGREE with (and why), and flag any "
        "temporal-order mistake (wrong event sequence). Be terse.\n\n"
        "--- PRIMARY REPORT ---\n" + primary_report
    )
    content: list[dict] = [{"type": "text", "text": verify_prompt}]
    try:
        for i, kf in enumerate(kf_paths):
            with open(kf, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            content.append({"type": "text",
                            "text": f"[frame {i + 1}/{len(kf_paths)}]"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        verdict = openai_chat(aux["base_url"], aux["api_key"], aux["model"],
                              [{"role": "user", "content": content}])
        if verdict:
            return "## Second opinion (auxiliary model)\n\n" + verdict
        return None
    finally:
        for kf in kf_paths:
            try:
                os.unlink(kf)
            except OSError:
                pass


# ---------------- main ----------------

def main() -> int:
    p = argparse.ArgumentParser(description="Videolyzer: deep video analysis.")
    p.add_argument("source", help="YouTube URL (gemini only) or local file path")
    p.add_argument("--provider", default=DEFAULT_PROVIDER,
                   choices=["gemini", "openai"],
                   help="API provider (default: gemini)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Model name (default: {DEFAULT_MODEL})")
    p.add_argument("--base-url", default=None,
                   help="Custom API base URL for OpenAI-compatible providers")
    p.add_argument("--api-key", default=None, help="API key (or env var)")
    p.add_argument("--aux-model", default=None,
                   help="Auxiliary multimodal model name (any OpenAI-compatible "
                        f"vision endpoint; or {AUX_ENV_MODEL})")
    p.add_argument("--aux-base-url", default=None,
                   help="Aux model API base URL "
                        f"(or {AUX_ENV_BASE_URL}; default {OPENAI_API_BASE})")
    p.add_argument("--aux-api-key", default=None,
                   help=f"Aux model API key (or {AUX_ENV_API_KEY})")
    p.add_argument("--fast", action="store_true",
                   help="Modo Rapido: run the whole analysis on the auxiliary "
                        "model via keyframes (fast, no full upload). "
                        "Requires --aux-model.")
    p.add_argument("--verify", action="store_true",
                   help="Second opinion: after the primary analysis, have the "
                        "auxiliary model cross-check the report. "
                        "Requires --aux-model.")
    p.add_argument("--out", default=None, help="Output .md file (default: stdout)")
    p.add_argument("--prompt", default=None,
                   help="Custom analysis prompt (overrides the default)")
    p.add_argument("--no-motion", action="store_true",
                   help="Skip the optical-flow subtle-motion pass")
    p.add_argument("--cut-threshold", type=float, default=0.4,
                   help="Scene-detection threshold for deterministic cut "
                        "detection (ffmpeg). Lower = more sensitive. "
                        "Default: 0.4")
    p.add_argument("--no-cuts", action="store_true",
                   help="Skip deterministic cut detection")
    p.add_argument("--keyframes", type=int, default=KEYFRAME_COUNT,
                   help="Keyframes for the openai provider / aux model (default: 8)")
    args = p.parse_args()

    key = get_api_key(args)
    aux = get_aux_config(args)
    prompt = args.prompt or ANALYZE_PROMPT
    want_motion = not args.no_motion
    cut_threshold = None if args.no_cuts else args.cut_threshold

    if args.fast and not aux:
        print("error: --fast (Modo Rapido) requires an auxiliary model. "
              f"Set --aux-model (or {AUX_ENV_MODEL}) plus "
              f"--aux-api-key (or {AUX_ENV_API_KEY}).", file=sys.stderr)
        return 2
    if args.verify and not aux:
        print("error: --verify requires an auxiliary model. "
              f"Set --aux-model (or {AUX_ENV_MODEL}).", file=sys.stderr)
        return 2
    if is_url(args.source) and (args.fast or args.provider == "openai"):
        print("error: URL input is only supported with "
              "--provider gemini (without --fast)", file=sys.stderr)
        return 2

    try:
        if args.fast:
            # Modo Rapido: everything on the aux model via keyframes.
            if aux is None:  # unreachable: validated above, narrows type
                raise RuntimeError("aux model required for --fast")
            if not os.path.isfile(args.source):
                print(f"error: file not found: {args.source}", file=sys.stderr)
                return 2
            report = analyze_aux_fast(aux, args.source, prompt,
                                      want_motion, args.keyframes,
                                      cut_threshold)
        elif args.provider == "gemini":
            base = GEMINI_API_BASE
            if is_url(args.source):
                print("URL mode: sending straight to the model, no download...",
                      file=sys.stderr)
                print("note: cut detection unavailable in URL mode "
                      "(requires the file)", file=sys.stderr)
                report = analyze_url_gemini(base, key, args.model,
                                            args.source, prompt)
            else:
                if not os.path.isfile(args.source):
                    print(f"error: file not found: {args.source}",
                          file=sys.stderr)
                    return 2
                print("file mode: validation + native upload...",
                      file=sys.stderr)
                report = analyze_file_gemini(base, key, args.model,
                                             args.source, prompt, want_motion,
                                             cut_threshold)
                if not report and aux:
                    # Fallback: aux model with keyframes.
                    print("warn: primary analysis failed, falling back to "
                          "auxiliary model...", file=sys.stderr)
                    report = analyze_aux_fast(aux, args.source, prompt,
                                              want_motion, args.keyframes,
                                              cut_threshold)
        else:  # openai-compatible
            if not os.path.isfile(args.source):
                print(f"error: file not found: {args.source}", file=sys.stderr)
                return 2
            base = args.base_url or OPENAI_API_BASE
            print(f"file mode (openai-compatible): {args.keyframes} keyframes...",
                  file=sys.stderr)
            report = analyze_file_openai(base, key, args.model, args.source,
                                         prompt, want_motion, args.keyframes,
                                         cut_threshold)
            if not report and aux:
                print("warn: primary analysis failed, falling back to "
                      "auxiliary model...", file=sys.stderr)
                report = analyze_aux_fast(aux, args.source, prompt,
                                          want_motion, args.keyframes,
                                          cut_threshold)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 4

    if not report:
        print("error: empty response from the model", file=sys.stderr)
        return 5

    # Second opinion (optional): aux model cross-checks the primary report.
    if args.verify and not args.fast and not is_url(args.source):
        if aux is None:  # unreachable: validated above, narrows type
            raise RuntimeError("aux model required for --verify")
        print("second opinion: aux model reviewing the report...",
              file=sys.stderr)
        verdict = analyze_aux_verify(aux, args.source, report, args.keyframes)
        if verdict:
            report = report.rstrip() + "\n\n" + verdict + "\n"
        else:
            print("note: second opinion unavailable, keeping primary report",
                  file=sys.stderr)

    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"report saved: {args.out}", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
