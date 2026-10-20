#!/usr/bin/env python3
"""Videolyzer - give agents the ability to understand and deep analyze videos.

Usage:
    videolyzer.py <youtube-url-or-file> [--provider gemini] [--out report.md]
        [--model gemini-3.1-flash-lite] [--base-url URL] [--api-key KEY]
        [--prompt "..."] [--no-motion] [--no-cuts] [--cut-threshold 0.4]
        [--keyframes 8]
        [--aux-model NAME] [--aux-base-url URL] [--aux-api-key KEY] [--fast]
        [--strips] [--strip-frames 8] [--max-strips 24]
        [--lang LANG] [--stt-model NAME] [--transcribe]

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
    Frame strips (--strips or VIDEOLYZER_STRIPS=1): instead of evenly spaced
    keyframes, the aux model gets one image per shot (top row spread across
    the shot, bottom row consecutive frames) plus per-shot cadence (ones/
    twos/threes), hold and camera numbers measured locally. Built for
    animation review; batched and merged for endpoints that cap images.
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

# ---------------- local measurement cache ----------------
# Deterministic passes (cut detection, optical flow, per-shot timing) are
# pure functions of the video bytes + params. Cache them under
# ~/.cache/videolyzer (or $VIDEOLYZER_CACHE_DIR) keyed by file hash so
# re-runs skip minutes of recomputation. Disabled with --no-cache.

CACHE_ENV = "VIDEOLYZER_CACHE_DIR"
_NO_CACHE = False


def _cache_dir() -> str | None:
    d = os.environ.get(CACHE_ENV)
    if not d:
        d = os.path.join(os.path.expanduser("~"), ".cache", "videolyzer")
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except OSError:
        return None


def _file_hash(path: str) -> str | None:
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:32]
    except OSError:
        return None


def _cached(kind: str, path: str, params: str, compute):
    """Return cached result for (kind, file, params) or run compute()."""
    cpath = None
    if not _NO_CACHE:
        d = _cache_dir()
        fh = _file_hash(path) if d else None
        if d and fh:
            safe_params = re.sub(r"[^A-Za-z0-9_.-]", "_", params)[:64]
            cpath = os.path.join(d, f"{kind}_{fh}_{safe_params}.json")
            try:
                with open(cpath) as f:
                    return json.load(f)
            except (OSError, ValueError):
                pass
    result = compute()
    if cpath:
        try:
            tmp = cpath + ".tmp"
            with open(tmp, "w") as f:
                json.dump(result, f)
            os.replace(tmp, cpath)
        except OSError:
            pass
    return result

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

TRANSCRIPT_HINT_SECTION = """
## Transcript (measured, dedicated speech-to-text model)
The transcript below was produced by a dedicated transcription model and
is GROUND TRUTH for what is said and when. Do NOT invent dialogue that
contradicts it; if a scene has no transcript lines, say "no dialogue".
Align your "Aligned transcript" section on these timestamps.
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


# Speech-to-text (dedicated transcriber, e.g. whisper large-v3 via any
# OpenAI-compatible /audio/transcriptions endpoint). Takes the audio load
# off the vision model: the transcript becomes ground truth in the prompt.
STT_ENV_MODEL = "VIDEOLYZER_STT_MODEL"
STT_ENV_BASE_URL = "VIDEOLYZER_STT_BASE_URL"
STT_ENV_API_KEY = "VIDEOLYZER_STT_API_KEY"
STT_DEFAULT_MODEL = "whisper-1"


def get_stt_config(args) -> dict | None:
    """STT endpoint config. None when not configured.

    Precedence: flags > VIDEOLYZER_STT_* env vars. Unlike the aux model,
    the base URL and API key fall back to the primary provider's when not
    set (same account often serves both).
    """
    if not args.transcribe:
        return None
    model = (args.stt_model or os.environ.get(STT_ENV_MODEL)
             or STT_DEFAULT_MODEL).strip()
    base_url = (args.stt_base_url or os.environ.get(STT_ENV_BASE_URL)
                or args.base_url or OPENAI_API_BASE).rstrip("/")
    api_key = (args.stt_api_key or os.environ.get(STT_ENV_API_KEY)
               or args.api_key or os.environ.get("VIDEOLYZER_API_KEY")
               or os.environ.get("OPENAI_API_KEY"))
    if not api_key or not api_key.strip():
        print(
            f"error: --transcribe set but no STT API key. Use --stt-api-key "
            f"or set {STT_ENV_API_KEY} (or reuse the primary --api-key).",
            file=sys.stderr,
        )
        sys.exit(3)
    return {"model": model, "base_url": base_url,
            "api_key": api_key.strip()}


def extract_audio(path: str) -> str | None:
    """Extract mono 16 kHz WAV for the transcriber. Returns temp path or None."""
    if not FFMPEG:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    r = subprocess.run(  # nosec B603 - args are fixed strings, no shell
        [FFMPEG, "-y", "-v", "error", "-i", path,
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", tmp.name],
        capture_output=True,
    )
    if r.returncode == 0 and os.path.getsize(tmp.name) > 0:
        return tmp.name
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
    return None


def _multipart_body(fields: dict, file_field: str, file_name: str,
                    file_bytes: bytes) -> tuple[bytes, str]:
    """Minimal multipart/form-data encoder (stdlib only)."""
    import secrets
    boundary = "----videolyzer" + secrets.token_hex(16)
    parts = []
    for k, v in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{file_field}"; '
        f'filename="{file_name}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    )
    head = "".join(parts).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + file_bytes + tail, boundary


def transcribe_audio(cfg: dict, wav_path: str) -> str | None:
    """Transcribe via {base}/audio/transcriptions (OpenAI-compatible).

    Tries verbose_json (segments with timestamps) first, falls back to
    plain json. Returns formatted "[mm:ss] text" lines or None.
    """
    url = f"{cfg['base_url']}/audio/transcriptions"
    if urllib.parse.urlparse(url).scheme.lower() not in ("http", "https"):
        print("error: refusing non-HTTP(S) STT URL scheme", file=sys.stderr)
        return None
    try:
        with open(wav_path, "rb") as f:
            audio = f.read()
    except OSError:
        return None
    size_mb = len(audio) / (1 << 20)
    if size_mb > 25:
        print(f"warn: audio is {size_mb:.0f} MB; many STT endpoints cap "
              f"uploads at 25 MB and may reject it", file=sys.stderr)
    for fmt in ("verbose_json", "json"):
        body, boundary = _multipart_body(
            {"model": cfg["model"], "response_format": fmt,
             "temperature": "0"},
            "file", "audio.wav", audio)
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}",
                   "Authorization": f"Bearer {cfg['api_key']}"}
        data = http_json(url, headers, body, timeout=600)
        if not data:
            continue
        try:
            if fmt == "verbose_json" and "segments" in data:
                lines = []
                for s in data["segments"]:
                    t = s.get("text", "").strip()
                    if t:
                        lines.append(f"[{fmt_ts(float(s.get('start', 0)))}] {t}")
                if lines:
                    return "\n".join(lines)
            elif "text" in data and data["text"].strip():
                return data["text"].strip()
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _whispero_bin() -> str | None:
    """Path of the installed `whispero` CLI, or None."""
    return shutil.which("whispero")


def _stt_http_explicit(args) -> bool:
    """True when the user explicitly configured an HTTP STT endpoint."""
    return bool(args.stt_model or args.stt_base_url or args.stt_api_key
                or os.environ.get(STT_ENV_MODEL)
                or os.environ.get(STT_ENV_BASE_URL)
                or os.environ.get(STT_ENV_API_KEY))


def _srt_to_lines(srt_path: str) -> str | None:
    """Parse whispero's SRT output into '[mm:ss] text' lines."""
    try:
        with open(srt_path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    lines = []
    for block in text.strip().split("\n\n"):
        parts = block.strip().splitlines()
        if len(parts) < 3:
            continue
        m = re.match(r"(\d{2}):(\d{2}):(\d{2})", parts[1])
        if not m:
            continue
        h, mi, s = (int(x) for x in m.groups())
        body = " ".join(p.strip() for p in parts[2:] if p.strip())
        if body:
            lines.append(f"[{fmt_ts(h * 3600 + mi * 60 + s)}] {body}")
    return "\n".join(lines) if lines else None


def transcribe_via_whispero(wav_path: str, timeout: int = 900) -> str | None:
    """Transcribe with the installed `whispero` CLI (local faster-whisper).

    Uses --format srt so segments keep timestamps (plus speaker tags when
    diarization is on). Returns '[mm:ss] text' lines or None.
    """
    exe = _whispero_bin()
    if not exe:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
    tmp.close()
    try:
        r = subprocess.run(  # nosec B603 - fixed args, binary from PATH
            [exe, wav_path, "--format", "srt", "--output", tmp.name],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            print(f"warn: whispero failed: "
                  f"{(r.stderr or '').strip()[-200:]}", file=sys.stderr)
            return None
        return _srt_to_lines(tmp.name)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"warn: whispero backend failed: {e}", file=sys.stderr)
        return None
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def fmt_ts(s: float) -> str:
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def _with_lang(text: str, lang: str | None) -> str:
    """Append a report-language instruction. Universal: any language the
    model understands, e.g. --lang pt, --lang es, --lang 日本語."""
    if lang and lang.strip():
        return text.rstrip() + f"\n\nWrite the entire report in {lang.strip()}."
    return text


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
    Results are cached per video+sample rate.
    """
    return _cached("motion", path, f"fps{sample_fps}",
                   lambda: _detect_subtle_motion_raw(path, sample_fps))


def _detect_subtle_motion_raw(path: str, sample_fps: float = 2.0) -> list:
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


def build_prompt(segments: list, base_prompt: str, cuts: list | None = None,
               transcript: str | None = None) -> str:
    prompt = base_prompt
    if cuts:
        items = "\n".join(f"- cut at {fmt_ts(c)} ({c:.2f}s)" for c in cuts)
        prompt = prompt + CUT_HINT_SECTION.format(items=items)
    if transcript and transcript.strip():
        prompt = prompt + TRANSCRIPT_HINT_SECTION.format(items=transcript.strip())
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
    is not installed. Results are cached per video+threshold.
    """
    return _cached("cuts", path, f"thr{threshold}",
                   lambda: _detect_cuts_raw(path, threshold))


def _detect_cuts_raw(path: str, threshold: float) -> list:
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
        "fps": video[0].get("r_frame_rate", "24/1"),
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


def openai_chat_stream(base: str, key: str, model: str,
                       messages: list) -> str | None:
    """Same call as openai_chat, streamed (SSE). Reasoning models can think
    for minutes before the first content token; streaming keeps proxies and
    gateways with ~100 s idle timeouts from cutting the request."""
    url = f"{base.rstrip('/')}/chat/completions"
    if urllib.parse.urlparse(url).scheme.lower() not in ("http", "https"):
        print("error: refusing non-HTTP(S) URL scheme", file=sys.stderr)
        return None
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": 8192, "stream": True}).encode()
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"})
        try:
            parts = []
            with urllib.request.urlopen(req, timeout=600) as resp:  # nosec B310 - scheme validated above
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    try:
                        choice = json.loads(line[5:])["choices"][0]
                    except (ValueError, KeyError, IndexError):
                        continue
                    parts.append((choice.get("delta") or {}).get("content") or "")
                    if choice.get("finish_reason") == "length":
                        print("warn: aux answer hit max_tokens and was cut",
                              file=sys.stderr)
            text = "".join(parts).strip()
            if text:
                return text
            last_err = RuntimeError("empty streamed response")
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504, 524):
                print(f"error: API {e.code}: "
                      f"{e.read().decode(errors='replace')[:300]}", file=sys.stderr)
                return None
            last_err = e
        except Exception as e:  # noqa: BLE001
            last_err = e
        print(f"warn: aux attempt {attempt + 1} failed: {last_err}",
              file=sys.stderr)
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAYS[attempt])
    print(f"warn: aux stream failed after {MAX_RETRIES} tries: {last_err}",
          file=sys.stderr)
    return None


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
                        cut_threshold: float | None,
                        transcript: str | None = None) -> str | None:
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
    parts = [{"text": build_prompt(segments, prompt, cuts, transcript)},
             {"file_data": {"file_uri": file_uri, "mime_type": mime}}]
    return gemini_generate(base, key, model, parts)


def analyze_file_openai(base: str, key: str, model: str, path: str,
                        prompt: str, want_motion: bool,
                        keyframes: int,
                        cut_threshold: float | None,
                        transcript: str | None = None) -> str | None:
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
    content: list[dict] = [{"type": "text", "text": build_prompt(segments, prompt, cuts, transcript)}]
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
                     cut_threshold: float | None,
                     transcript: str | None = None) -> str | None:
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
    full_prompt = AUX_FAST_PREAMBLE + build_prompt(segments, prompt, cuts, transcript)
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
                       keyframes: int, strips: bool = False,
                       strip_frames: int | None = None,
                       max_strips: int | None = None,
                       cut_threshold: float | None = None,
                       lang: str | None = None) -> str | None:
    """Second opinion: the aux model cross-checks the primary report.

    Sends keyframes plus the primary report and asks for agreement /
    disagreement per scene, focusing on temporal claims. In strips mode
    it sends per-shot frame strips with the measured cadence/camera
    numbers instead of evenly spaced keyframes. Returns the verdict text
    or None.
    """
    # Defaults resolved at call time: the STRIP_* constants are defined
    # further below in this module (after this function).
    if strip_frames is None:
        strip_frames = STRIP_FRAMES
    if max_strips is None:
        max_strips = MAX_STRIPS
    info = validate_video(path)
    content: list[dict] = []
    paths: list[str] = []
    try:
        if strips:
            duration = info["duration"]
            try:
                num, den = info.get("fps", "24/1").split("/")
                fps = float(num) / float(den or 1) or 24.0
            except (ValueError, ZeroDivisionError, AttributeError):
                fps = 24.0
            cuts = detect_cuts(path, min(cut_threshold if cut_threshold
                                         is not None else STRIP_CUT_THRESHOLD,
                                         STRIP_CUT_THRESHOLD))
            shots = shot_list(duration, cuts)
            stats = shot_timing_stats(path, shots) or [
                {"start": s, "end": e, "note": "opencv not installed"}
                for s, e in shots]
            order = list(range(len(shots)))
            if len(order) > max_strips:
                order = [order[round(i * (len(order) - 1) / (max_strips - 1))]
                         for i in range(max_strips)]
            table = "\n".join(fmt_shot(i, stats[i]) for i in order)
            verify_prompt = (
                "You are a second-opinion reviewer. Below is a video analysis "
                "report produced by another model. You get one frame strip "
                "per shot (in order, shot number printed on the image) plus "
                "locally measured cadence/camera numbers, which are ground "
                "truth for timing: trust them over your visual impression. "
                "Review each scene claim against the strips and numbers: say "
                "which claims you AGREE with, which you DISAGREE with (and "
                "why), and flag any temporal-order mistake (wrong event "
                "sequence). Be terse.\n\n"
                f"Per-shot numbers:\n{table}\n\n"
                "--- PRIMARY REPORT ---\n" + primary_report
            )
            content.append({"type": "text",
                            "text": _with_lang(verify_prompt, lang)})
            for i in order:
                sp = extract_strip(path, shots[i][0], shots[i][1],
                                   strip_frames, fps,
                                   f"SHOT {i + 1}  ({shots[i][0]:.1f}s-"
                                   f"{shots[i][1]:.1f}s)")
                if not sp:
                    continue
                paths.append(sp)
                with open(sp, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                content.append({"type": "text",
                                "text": "[" + fmt_shot(i, stats[i]) + "]"})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            if not paths:
                return None
        else:
            kf_paths = extract_keyframes(path, keyframes, info["duration"])
            if not kf_paths:
                return None
            paths = kf_paths
            verify_prompt = (
                "You are a second-opinion reviewer. Below is a video analysis report "
                "produced by another model from evenly spaced keyframes of the same "
                "video. Review each scene claim against the frames: say which claims "
                "you AGREE with, which you DISAGREE with (and why), and flag any "
                "temporal-order mistake (wrong event sequence). Be terse.\n\n"
                "--- PRIMARY REPORT ---\n" + primary_report
            )
            content.append({"type": "text",
                            "text": _with_lang(verify_prompt, lang)})
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
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------- per-shot frame strips (aux fallback for animation) ----------------

STRIP_FRAMES = 8          # frames per row in each shot strip
MAX_STRIPS = 24           # shots sent to the aux model (evenly sampled if more)
STRIPS_PER_REQUEST = 6    # images per aux request (batched, then merged)
STRIP_BATCH_DELAY = 2.0   # seconds between aux requests
STRIP_CUT_THRESHOLD = 0.25  # scene threshold cap in strips mode
STRIPS_ENV = "VIDEOLYZER_STRIPS"

STRIPS_PREAMBLE = """\
You are analyzing a video WITHOUT seeing it play. You get one image per shot, \
in order, its shot number printed at the top. Each image has two rows of {n} \
frames, left to right:
- TOP row: frames spread evenly across the whole shot (staging, acting, \
pose progression).
- BOTTOM row: {n} CONSECUTIVE frames (1/{fps:.0f} s apart) from the middle \
of the shot (timing).
With each shot you also get numbers measured locally on every frame. They are \
ground truth for timing: trust them over your visual impression for cadence \
(ones/twos/threes), holds and camera speed, and do not invent frame-level \
timing the numbers do not support. Use the frames for what numbers cannot \
see: acting and gesture, staging and composition, character on-model \
consistency between shots, line weight, palette, camera language (angle, \
framing, move type) and how each shot flows into the next (match cuts, \
eyelines, action carried across the cut). There is no audio: mark any \
transcript section as unavailable.
Glossary: "ones" = new drawing every frame; "twos" = every 2nd frame; \
"threes" = every 3rd; "held" = no new drawing; "camera" = whole-frame \
pan/zoom measured after removing it from the drawing changes.
"""

STRIPS_FOCUS = """
Besides the sections above, include a section "## Animation and camera" with:
cadence and timing (per shot and overall), consistency (on-model, line \
weight, palette), acting, camera language, and shot-to-shot flow. Quote the \
measured numbers where you use them. Say plainly when a frame is too small \
to judge something.
"""


def shot_list(duration: float, cuts: list, min_len: float = 0.25) -> list:
    """Shots as (start, end) seconds from cut times; merges flash-frame cuts."""
    bounds = [0.0]
    for c in sorted(cuts):
        if c - bounds[-1] >= min_len and duration - c >= min_len:
            bounds.append(c)
    bounds.append(duration)
    return list(zip(bounds[:-1], bounds[1:]))


def shot_timing_stats(path: str, shots: list) -> list:
    """Per-shot cadence and camera numbers from one decode pass (opencv).

    Camera motion is the dominant similarity transform between frames (LK
    features + RANSAC); a second transform takes a second layer (a held cel
    over a scrolling background). A 'new drawing' is >0.4% of pixels still
    changing after both are removed. Returns [] without opencv.
    Results are cached per video+shot list.
    """
    import hashlib
    shots_key = hashlib.sha256(json.dumps(
        [[round(s, 3), round(e, 3)] for s, e in shots]).encode()).hexdigest()[:16]
    return _cached("timing", path, f"shots{shots_key}",
                   lambda: _shot_timing_stats_raw(path, shots))


def _shot_timing_stats_raw(path: str, shots: list) -> list:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return []
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w, h = 240, 135
    inner = np.zeros((h, w), bool)
    inner[6:-6, 6:-6] = True
    per = []  # (t, changed_fraction, camera_px_per_frame_at_1080p)
    prev = None
    k = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (w, h),
                                                     interpolation=cv2.INTER_AREA),
                                          cv2.COLOR_BGR2GRAY), (3, 3), 0)
        if prev is not None:
            cands = [cv2.absdiff(g, prev)]
            cam = 0.0
            p0 = cv2.goodFeaturesToTrack(prev, 400, 0.005, 4)
            if p0 is not None and len(p0) >= 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None,
                                                     winSize=(15, 15), maxLevel=3)
                sel = st.ravel() == 1
                a, b = p0[sel], p1[sel]
                for layer in range(2):
                    if len(a) < 10:
                        break
                    m, inl = cv2.estimateAffinePartial2D(
                        a, b, method=cv2.RANSAC, ransacReprojThreshold=1.0)
                    if m is None or inl.sum() < 8:
                        break
                    if layer == 0:
                        zoom = abs(float(np.hypot(m[0, 0], m[1, 0])) - 1.0)
                        cam = float(np.hypot(m[0, 2], m[1, 2])) * 1080 / h \
                            + zoom * 1000
                    cands.append(cv2.absdiff(cv2.warpAffine(
                        prev, m, (w, h), borderMode=cv2.BORDER_REFLECT), g))
                    keep = inl.ravel() == 0
                    a, b = a[keep], b[keep]
            resid = np.minimum.reduce(cands)
            per.append((k / fps, float((resid[inner] > 14).mean()), cam))
        prev = g
        k += 1
    cap.release()
    stats = []
    for s, e in shots:
        rows = [r for r in per if s + 1.5 / fps < r[0] < e - 0.5 / fps]
        if len(rows) < 3:
            stats.append({"start": s, "end": e, "note": "too short to measure"})
            continue
        events = [i for i, r in enumerate(rows) if r[1] > 0.004]
        gaps = [b - a for a, b in zip(events, events[1:]) if b - a <= 4]
        counts = {n: gaps.count(n) for n in (1, 2, 3, 4)}
        if len(events) < 4:
            cadence = "held (no new drawings)"
        else:
            best = max(counts, key=lambda n: counts[n])
            cadence = {1: "ones", 2: "twos", 3: "threes", 4: "fours"}[best]
            if counts[best] < 0.5 * max(len(gaps), 1):
                cadence += " (mixed)"
        cams = sorted(r[2] for r in rows)
        moving = [c for c in cams if c > 0.3]
        stats.append({
            "start": s, "end": e,
            "cadence": cadence,
            "new_drawing_frames": round(len(events) / len(rows), 2),
            "held_frames": round(sum(1 for r in rows if r[1] < 0.001) / len(rows), 2),
            "camera_moving": round(len(moving) / len(rows), 2),
            "camera_px_per_frame": round(moving[len(moving) // 2], 1) if moving else 0.0,
        })
    return stats


def _ckpt_path(kind: str, path: str, params: str) -> str | None:
    """Checkpoint file path for resumable long runs, or None."""
    d = _cache_dir()
    fh = _file_hash(path) if d else None
    if not (d and fh):
        return None
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", params)[:64]
    return os.path.join(d, f"ckpt_{kind}_{fh}_{safe}.json")


def _ckpt_load(cpath: str | None) -> dict:
    if not cpath:
        return {}
    try:
        with open(cpath) as f:
            d = json.load(f)
            return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _ckpt_save(cpath: str | None, data: dict) -> None:
    if not cpath:
        return
    try:
        tmp = cpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, cpath)
    except OSError:
        pass


def _ckpt_delete(cpath: str | None) -> None:
    if cpath:
        try:
            os.unlink(cpath)
        except OSError:
            pass


def extract_strip(path: str, start: float, end: float, n: int,
                  fps: float, label: str = "") -> str | None:
    """One JPEG per shot: top row spread across the shot, bottom row
    consecutive frames from its middle, the shot label burned in on top
    (models keep shot identity better when it is inside the image).
    Returns a temp path or None."""
    if not FFMPEG:
        return None
    dur = max(end - start, 1.0 / fps)
    mid = max(start, start + dur / 2 - n / (2 * fps))
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    graph = (f"[0:v]fps={n / dur:.4f},scale=320:-2,tile={n}x1[top];"
             f"[1:v]scale=320:-2,tile={n}x1[bot];[top][bot]vstack")
    text = re.sub(r"[^A-Za-z0-9 .()-]", "", label)  # ':' would break drawtext
    tag = (f",pad=iw:ih+44:0:44,drawtext=text='{text}':fontcolor=yellow:"
           "fontsize=32:x=10:y=6") if text else ""
    for g in ([graph + tag, graph] if tag else [graph]):  # no drawtext? no label
        r = subprocess.run(  # nosec B603 - args are fixed strings, no shell
            [FFMPEG, "-y", "-v", "error",
             "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", path,
             "-ss", f"{mid:.3f}", "-i", path,
             "-filter_complex", g, "-frames:v", "1", "-q:v", "4", tmp.name],
            capture_output=True,
        )
        if r.returncode == 0 and os.path.getsize(tmp.name) > 0:
            return tmp.name
    os.unlink(tmp.name)
    return None


def fmt_shot(i: int, st: dict) -> str:
    head = (f"Shot {i + 1}: {fmt_ts(st['start'])}-{fmt_ts(st['end'])} "
            f"({st['end'] - st['start']:.2f}s)")
    if "cadence" not in st:
        return head + " - " + st.get("note", "no timing numbers")
    return (head + f" - cadence {st['cadence']}, new drawing on "
            f"{st['new_drawing_frames']:.0%} of frames, drawing held on "
            f"{st['held_frames']:.0%}, camera moving on {st['camera_moving']:.0%} "
            f"(median {st['camera_px_per_frame']} px/frame at 1080p)")


def analyze_aux_strips(aux: dict, path: str, prompt: str, want_motion: bool,
                       cut_threshold: float | None,
                       frames: int = STRIP_FRAMES,
                       max_strips: int = MAX_STRIPS,
                       resume: bool = True,
                       lang: str | None = None,
                       transcript: str | None = None) -> str | None:
    """Aux analysis built for animation: one ordered frame strip per shot
    plus locally measured cadence/camera numbers. Batched for endpoints that
    take a limited number of images per request; batches are then merged.

    Long runs checkpoint per-batch notes under the cache dir: if the run
    dies mid-way, the next run resumes from the first missing batch
    (disable with resume=False / --no-resume). The checkpoint is deleted
    after a successful merge.
    """
    info = validate_video(path)
    duration = info["duration"]
    fps = 24.0
    try:
        num, den = info.get("fps", "24/1").split("/")
        fps = float(num) / float(den or 1) or 24.0
    except (ValueError, ZeroDivisionError, AttributeError):
        pass
    # Dark or low-contrast cuts (common in animation) score low: cap at 0.25.
    cuts = detect_cuts(path, min(cut_threshold if cut_threshold is not None
                                 else STRIP_CUT_THRESHOLD, STRIP_CUT_THRESHOLD))
    shots = shot_list(duration, cuts)
    cuts = [s for s, _ in shots[1:]]  # same numbering in prompt and strips
    print(f"strips: {len(shots)} shots from {len(cuts)} cuts", file=sys.stderr)
    stats = shot_timing_stats(path, shots) or \
        [{"start": s, "end": e, "note": "opencv not installed"} for s, e in shots]
    order = list(range(len(shots)))
    if len(order) > max_strips:  # keep order, sample evenly
        order = [order[round(i * (len(order) - 1) / (max_strips - 1))]
                 for i in range(max_strips)]
    segments = detect_subtle_motion(path) if want_motion else []
    table = "\n".join(fmt_shot(i, st) for i, st in enumerate(stats))
    lengths = sorted(e - s for s, e in shots)
    summary = (f"Measured: {len(shots)} shots in {duration:.1f}s, median shot "
               f"{lengths[len(lengths) // 2]:.2f}s, mean {duration / len(shots):.2f}s.\n"
               f"Per-shot numbers (all shots):\n{table}\n")
    preamble = STRIPS_PREAMBLE.format(n=frames, fps=fps)
    batches = [order[i:i + STRIPS_PER_REQUEST]
               for i in range(0, len(order), STRIPS_PER_REQUEST)]
    # Checkpoint: per-batch notes keyed by video + strip config.
    cpath = _ckpt_path("strips", path, f"f{frames}_m{max_strips}") \
        if resume else None
    notes_by_batch: dict[int, str] = {}
    if cpath:
        saved = _ckpt_load(cpath)
        if saved.get("order") == order and isinstance(saved.get("notes"), dict):
            notes_by_batch = {int(k): v for k, v in saved["notes"].items()}
            if notes_by_batch:
                print(f"strips: resuming, {len(notes_by_batch)}/"
                      f"{len(batches)} batches already done",
                      file=sys.stderr)
    first_call = True
    for bi, batch in enumerate(batches):
        if bi in notes_by_batch:
            continue  # checkpointed in a previous run
        if not first_call:
            time.sleep(STRIP_BATCH_DELAY)  # sequential and gentle on rate limits
        first_call = False
        content: list[dict] = []
        paths = []
        try:
            for i in batch:
                sp = extract_strip(path, shots[i][0], shots[i][1], frames, fps,
                                   f"SHOT {i + 1}  ({shots[i][0]:.1f}s-{shots[i][1]:.1f}s)")
                if not sp:
                    continue
                paths.append(sp)
                with open(sp, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                content.append({"type": "text", "text": "[" + fmt_shot(i, stats[i]) + "]"})
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            if not paths:
                continue
            if len(batches) == 1:
                ask = _with_lang(preamble + summary
                                 + build_prompt(segments, prompt, cuts, transcript)
                                 + STRIPS_FOCUS, lang)
            else:
                ask = _with_lang(
                    preamble + summary +
                    f"\nThis is batch {bi + 1}/{len(batches)}. For each image, "
                    "in order, start with the SHOT number printed on that "
                    "image, then write 3-6 terse lines: what happens, acting, "
                    "camera, cadence (from the numbers), consistency issues, "
                    "and how it connects to the previous shot. Describe only "
                    "what is in that image.", lang)
            print(f"strips: batch {bi + 1}/{len(batches)} ({len(paths)} shots) "
                  f"-> aux model '{aux['model']}'...", file=sys.stderr)
            out = openai_chat_stream(aux["base_url"], aux["api_key"], aux["model"],
                              [{"role": "user",
                                "content": [{"type": "text", "text": ask}] + content}])
        finally:
            for sp in paths:
                try:
                    os.unlink(sp)
                except OSError:
                    pass
        if out is None:
            return None  # checkpoint kept: next run resumes here
        if len(batches) == 1:
            return out
        notes_by_batch[bi] = out
        _ckpt_save(cpath, {"order": order,
                           "notes": {str(k): v
                                     for k, v in notes_by_batch.items()}})
    notes = [notes_by_batch[bi] for bi in range(len(batches))
             if bi in notes_by_batch]
    if not notes:
        return None
    group = ("" if len(shots) <= 40 else
             f"\nThere are {len(shots)} shots: in the Scenes table group "
             "consecutive shots into scenes (about 30 rows at most) and cite "
             "the shot numbers each row covers. The per-shot notes are "
             "appended to your report automatically; do not repeat them.\n")
    merge = _with_lang(
        preamble + summary + "\nPer-shot notes from the frame strips, in "
        "order:\n\n" + "\n\n".join(notes) + "\n\nWrite the final report "
        "from these notes and numbers.\n" + group +
        build_prompt(segments, prompt, cuts, transcript) + STRIPS_FOCUS, lang)
    report = openai_chat_stream(aux["base_url"], aux["api_key"], aux["model"],
                                [{"role": "user", "content": merge}])
    if report is None:
        return None  # checkpoint kept: next run resumes here
    _ckpt_delete(cpath)  # full success: checkpoint no longer needed
    return (report.rstrip() + "\n\n## Per-shot notes (aux model, from frame strips)\n\n"
            + "\n\n".join(notes) + "\n")


def run_aux(aux: dict, args, path: str, prompt: str, want_motion: bool,
            cut_threshold: float | None,
            transcript: str | None = None) -> str | None:
    """Aux analysis in the configured mode: frame strips or keyframes."""
    if args.strips:
        return analyze_aux_strips(aux, path, prompt, want_motion,
                                  cut_threshold, args.strip_frames,
                                  args.max_strips,
                                  resume=not args.no_resume,
                                  lang=args.lang,
                                  transcript=transcript)
    return analyze_aux_fast(aux, path, prompt, want_motion, args.keyframes,
                            cut_threshold, transcript=transcript)


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
    p.add_argument("--strips", action="store_true",
                   default=os.environ.get(STRIPS_ENV, "") not in ("", "0"),
                   help="Aux model (--fast and the fallback) gets one ordered "
                        "frame strip per shot plus measured cadence/camera "
                        f"numbers instead of evenly spaced keyframes (or {STRIPS_ENV}=1)")
    p.add_argument("--strip-frames", type=int, default=STRIP_FRAMES,
                   help=f"Frames per strip row (default: {STRIP_FRAMES})")
    p.add_argument("--max-strips", type=int, default=MAX_STRIPS,
                   help=f"Max shots sent as strips (default: {MAX_STRIPS})")
    p.add_argument("--out", default=None, help="Output .md file (default: stdout)")
    p.add_argument("--prompt", default=None,
                   help="Custom analysis prompt (overrides the default)")
    p.add_argument("--lang", default=None,
                   help="Report language (any language name or code, e.g. "
                        "pt, es, Japanese). The whole report is written in "
                        "it; nothing is hardcoded.")
    p.add_argument("--transcribe", action="store_true",
                   help="Transcribe the audio with a dedicated speech-to-text "
                        "model (any OpenAI-compatible /audio/transcriptions "
                        "endpoint, e.g. whisper large-v3) and inject it as "
                        "ground truth, taking the audio load off the vision "
                        "model.")
    p.add_argument("--stt-model", default=None,
                   help="STT model name (or VIDEOLYZER_STT_MODEL; "
                        "default: whisper-1)")
    p.add_argument("--stt-backend", default="auto",
                   choices=["auto", "whispero", "http"],
                   help="Transcriber backend: auto (default) uses the "
                        "installed whispero CLI when present and no HTTP "
                        "STT endpoint is configured; whispero forces it; "
                        "http forces the /audio/transcriptions endpoint")
    p.add_argument("--stt-base-url", default=None,
                   help="STT API base URL (or VIDEOLYZER_STT_BASE_URL; "
                        "defaults to the primary base URL)")
    p.add_argument("--stt-api-key", default=None,
                   help="STT API key (or VIDEOLYZER_STT_API_KEY; defaults "
                        "to the primary API key)")
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
    p.add_argument("--no-cache", action="store_true",
                   help="Skip the local measurement cache (cut detection, "
                        "optical flow, per-shot timing are cached per video "
                        "under ~/.cache/videolyzer)")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore any strips checkpoint and start all batches "
                        "from scratch")
    args = p.parse_args()

    global _NO_CACHE
    _NO_CACHE = args.no_cache

    key = get_api_key(args)
    aux = get_aux_config(args)
    prompt = _with_lang(args.prompt or ANALYZE_PROMPT, args.lang)
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
    if args.strips and aux is None:
        print("warn: --strips is set but no auxiliary model is configured; "
              "strips only applies to the aux path (--fast or the fallback) "
              "and will be ignored.", file=sys.stderr)

    # Transcriber backend: auto -> local whispero when installed and no
    # HTTP STT endpoint was explicitly configured, else the HTTP endpoint.
    stt = None
    stt_backend = None
    if args.transcribe:
        want = args.stt_backend
        if want == "auto":
            want = "http" if _stt_http_explicit(args) else (
                "whispero" if _whispero_bin() else "http")
        if want == "whispero" and not _whispero_bin():
            print("error: --stt-backend whispero but no `whispero` CLI "
                  "found on PATH.", file=sys.stderr)
            return 2
        stt_backend = want
        if want == "http":
            stt = get_stt_config(args)

    try:
        transcript = None
        if stt_backend and not is_url(args.source):
            if not os.path.isfile(args.source):
                print(f"error: file not found: {args.source}", file=sys.stderr)
                return 2
            if stt_backend == "whispero":
                print("transcribing audio with local whispero...",
                      file=sys.stderr)
            else:
                print("transcribing audio with dedicated STT model "
                      f"'{stt['model'] if stt else '?'}'...",
                      file=sys.stderr)
            wav = extract_audio(args.source)
            if wav:
                try:
                    if stt_backend == "whispero":
                        transcript = transcribe_via_whispero(wav)
                    elif stt is not None:
                        transcript = transcribe_audio(stt, wav)
                finally:
                    try:
                        os.unlink(wav)
                    except OSError:
                        pass
            if transcript:
                n = len(transcript.splitlines())
                print(f"  transcript ok: {n} lines", file=sys.stderr)
            else:
                print("warn: transcription failed, continuing without it",
                      file=sys.stderr)
        if args.fast:
            # Modo Rapido: everything on the aux model via keyframes.
            if aux is None:  # unreachable: validated above, narrows type
                raise RuntimeError("aux model required for --fast")
            if not os.path.isfile(args.source):
                print(f"error: file not found: {args.source}", file=sys.stderr)
                return 2
            report = run_aux(aux, args, args.source, prompt,
                                  want_motion, cut_threshold,
                                  transcript=transcript)
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
                                             cut_threshold,
                                             transcript=transcript)
                if not report and aux:
                    # Fallback: aux model with keyframes.
                    print("warn: primary analysis failed, falling back to "
                          "auxiliary model...", file=sys.stderr)
                    report = run_aux(aux, args, args.source, prompt,
                                  want_motion, cut_threshold,
                                  transcript=transcript)
        else:  # openai-compatible
            if not os.path.isfile(args.source):
                print(f"error: file not found: {args.source}", file=sys.stderr)
                return 2
            base = args.base_url or OPENAI_API_BASE
            print(f"file mode (openai-compatible): {args.keyframes} keyframes...",
                  file=sys.stderr)
            report = analyze_file_openai(base, key, args.model, args.source,
                                         prompt, want_motion, args.keyframes,
                                         cut_threshold,
                                         transcript=transcript)
            if not report and aux:
                print("warn: primary analysis failed, falling back to "
                      "auxiliary model...", file=sys.stderr)
                report = run_aux(aux, args, args.source, prompt,
                                  want_motion, cut_threshold,
                                  transcript=transcript)
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
        verdict = analyze_aux_verify(aux, args.source, report, args.keyframes,
                                      args.strips, args.strip_frames,
                                      args.max_strips, cut_threshold,
                                      lang=args.lang)
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
