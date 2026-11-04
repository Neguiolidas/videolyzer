# Videolyzer

**Give agents the ability to understand and deep analyze videos.**

A single CLI that turns a video (YouTube URL or local file) into a structured Markdown report: detected scenes with timestamps, transcript aligned per scene, key moments, tone/emotion, and temporal verification. Built for agents that need to *watch* video, not just transcribe it.

**Agents:** read [`AGENTS.md`](AGENTS.md) — mode picker, `--json` envelope, exit codes, trust levels per field, and the repo map.

## Quick start

```bash
pip install videolyzer          # installs the `videolyzer` command

# Free, no API key: exact cuts, shots, motion segments, video facts (JSON)
videolyzer ./my-video.mp4 --measure-only

export VIDEOLYZER_API_KEY="your-key"

# URL mode (gemini only, no download)
videolyzer "https://www.youtube.com/watch?v=..." --out report.md

# Local file mode (native video understanding)
videolyzer ./my-video.mp4 --out report.md

# Machine-readable: report + measurements + provenance in one JSON object
videolyzer ./my-video.mp4 --json --out report.json

# Fast action / subtle motion: let Gemini sample 4 fps instead of 1
videolyzer ./my-video.mp4 --fps 4 --out report.md

# Speech as ground truth, transcribed ON THIS MACHINE (pip install "whispero[local]")
videolyzer ./my-video.mp4 --transcribe --out report.md

# Offline: nothing leaves the machine (local vision server, e.g. Ollama)
videolyzer ./my-video.mp4 --provider local --model qwen2.5vl:7b --transcribe
# Offline with no vision model at all: deterministic measured report
videolyzer ./my-video.mp4 --provider local --transcribe --hear

# Any OpenAI-compatible provider (Groq, Together, OpenAI, ...)
videolyzer ./my-video.mp4 --provider openai --model gpt-4o \
    --base-url https://api.groq.com/openai/v1 --out report.md

# Auxiliary model (any OpenAI-compatible vision endpoint)
export VIDEOLYZER_AUX_MODEL="your-vision-model"
export VIDEOLYZER_AUX_BASE_URL="https://your-endpoint/v1"
export VIDEOLYZER_AUX_API_KEY="your-key"

# Modo Rapido: fast keyframe analysis on the aux model (no full upload)
videolyzer ./my-video.mp4 --fast --out report.md

# Second opinion: primary analysis + aux model cross-check
videolyzer ./my-video.mp4 --verify --out report.md
```

Running from a checkout works the same with `python videolyzer.py ...`.

## Modes: URL vs local file

| | URL mode | Local file mode |
|---|---|---|
| Providers | `gemini` only | `gemini`, `openai` and `local` (offline) |
| Download | None. The URL goes straight to the model (`file_data`) | None for gemini (full upload via Files API, streamed from disk, overlapped with the local passes); keyframes extracted locally for openai |
| Temporal understanding | Native: the model samples the timeline itself | Native (gemini) / keyframe-based (openai) |
| Subtle-motion detection | **Not available** (would require downloading the video) | Available: optical-flow pass flags small-amplitude motion segments (frame pairs across a hard cut are ignored) and injects them into the prompt as an explicit hint |
| Cut detection | **Not available** (would require downloading the video) | Available: ffmpeg scene detection finds exact hard-cut timestamps (`--cut-threshold`, default 0.4); they go into the prompt as scene-boundary anchors |
| Speed | Fastest | Slower (upload + optional motion/cut passes) |

### URL mode (gemini only)

No download, no frame extraction. The video URL is passed to the model and it does its own temporal sampling. Best for quick overviews of online videos. Limitation: subtle motion (small-amplitude movement) can be missed, because the pre-analysis optical-flow pass cannot run without the pixels.

### Local file mode

The file is validated with `ffprobe` (codec, duration, integrity — no frames extracted), then two deterministic passes run locally before the model sees anything:

- **Cut detection** (ffmpeg scene detection, `--cut-threshold` default 0.4, skip with `--no-cuts`): finds exact hard-cut timestamps. They go into the prompt as scene-boundary anchors ("a new scene starts at each cut timestamp — do NOT merge scenes across a cut"), so the model aligns on measured cuts instead of guessing approximate boundaries.
- **Subtle-motion pass** (Farneback optical flow, 2 fps sampling, 3×3 grid): measures motion magnitude per segment and flags anything above background noise (median + 4·1.4826·MAD, robust to outliers). Frame pairs that straddle a detected cut are dropped, so a cut is never reported as "motion". The flagged timestamps, region and dominant axis go into the prompt as an explicit hint ("do NOT say these segments are static").
- **Video facts** (ffprobe): exact duration, resolution, frame rate and audio presence go into the prompt, so the model has a hard bound on its timestamps.

Then:

- **gemini:** the whole file is uploaded through the Files API (in parallel with the local passes); the model samples the timeline itself — 1 fps by default, `--fps N` to raise it, `--media-resolution low|medium|high` to trade detail for tokens.
  Containers Gemini does not document (`.mkv`, `.ts`, `.ogv`...) are remuxed to MP4 first (stream copy, no re-encode). The uploaded copy is **deleted after the analysis** (the Files API would otherwise keep it 48 h); `--keep-upload` keeps it and reuses it on the next run of the same video with the same key (no re-upload).
- **openai (any OpenAI-compatible API):** `--keyframes N` (default 8) frames are extracted with ffmpeg and sent as images through `/v1/chat/completions` (vision). Placement is cut-aware — when the shots fit in the budget every shot gets a frame and spare frames go to the longest shots — and every frame is labelled with its exact timestamp. URL input is not supported with this provider.

### Offline mode (`--provider local`)

Every request stays on this machine; this is enforced, not just a default: any non-loopback host (primary, aux or STT endpoint) is refused, redirects are never followed, and the local transcriber gets no cloud credentials. `--allow-lan` also admits private-network hosts (RFC 1918, CGNAT/Tailscale, IPv6 ULA) such as a GPU box on the LAN.

- **With a local vision model** (`--model`, or `VIDEOLYZER_LOCAL_MODEL`): any OpenAI-compatible server — Ollama (default `http://127.0.0.1:11434/v1`), LM Studio, llama.cpp, vLLM (`--base-url`). Same cut-aware, timestamp-labelled keyframes as the openai provider (`--strips` works too). No key needed. Ollama note: its default context window is small for several images; raise it (e.g. `OLLAMA_CONTEXT_LENGTH=16384`) or the prompt is silently truncated.
- **Without a model** (or when the local server is down): a deterministic **measured report** with the same section headings — shots table, measured motion, transcript aligned per shot, pacing, audio measurements. It says *when* things change and *where* motion is, never *what* is on screen.
- Transcription uses local whispero; there is no cloud fallback.

## Auxiliary model & Modo Rapido

The auxiliary model is a second, independent multimodal model — any OpenAI-compatible vision endpoint (no provider hardcoded). It plays three roles:

| Role | How |
|---|---|
| **Second opinion** (`--verify`) | After the primary analysis, the aux model reviews the report (against frame strips when `--strips` is on, keyframes otherwise): agrees/disagrees per scene, flags temporal-order mistakes. Appended as a "Second opinion" section. |
| **Temporal verifier** | Built into every aux pass: the prompt forces motion verbs and event-order checks across keyframes. |
| **Fallback** | If the primary analysis fails and an aux model is configured, Videolyzer automatically retries with the aux model on keyframes (or on per-shot frame strips with `--strips`). |

**Dedicated transcriber** (`--transcribe`) offloads the audio: the transcript goes into the prompt as ground truth (clearly delimited and marked as data, so speech in the video cannot act as instructions), so the vision model spends its budget on what it sees. Two backends (`--stt-backend`):

- `whispero` — the installed [whispero](https://pypi.org/project/whispero/) CLI, always run as `whispero <audio> --provider local --diarize --format json`: faster-whisper plus local speaker diarization on this machine. Both pins matter: whispero's own default provider is cloud Gemini and its default language is Portuguese. `--diarize` is what makes whispero return timestamped segments; speaker labels come along. Needs `pip install "whispero[local]"` (models download once to `~/.cache/huggingface`). `--stt-speakers N` passes a known speaker count; `--stt-backend whispero --stt-model large-v3` picks the model.
- `http` — any OpenAI-compatible `/audio/transcriptions` endpoint (e.g. whisper large-v3). Audio is sent as 32 kbps MP3 (about 14 MB per hour, under the usual 25 MB cap). Config via `--stt-model` / `--stt-base-url` / `--stt-api-key` (or `VIDEOLYZER_STT_*`).

With `auto` (default), whispero is picked when installed and no HTTP STT setting is given. `--stt-lang` sets the spoken language (default: auto-detect; `--lang` is the *report* language). Transcripts are cached per video and settings. The two tools stay separate packages — this is integration, not a merger.

**Audio measurements** (`--hear`, local, needs `pip install "whispero[hear]"`): `whispero hear --music --beats --cuts <measured cuts>` measures the music bed and ducking, tempo/beat grid, and classifies **every video cut** as on-beat / in-pause / mid-voiced. The numbers go into the prompt and the JSON (`measurements.audio`). Caveat: without a music bed the beat grid can follow the rhythm of speech, and the prompt says so.

**Modo Rapido** (`--fast`) skips the full video upload entirely and runs the whole analysis on the auxiliary model with cut-aware, timestamp-labelled keyframes (+ the optical-flow subtle-motion hints). Much faster and cheaper than the native path — best when only "what appears" matters and fine motion detail is not critical. Requires the aux model to be configured (the primary API key is not needed).

Configuration (flags take precedence over env vars):

| Flag | Env var | Default |
|---|---|---|
| `--aux-model` | `VIDEOLYZER_AUX_MODEL` | — (required to enable) |
| `--aux-base-url` | `VIDEOLYZER_AUX_BASE_URL` | `https://api.openai.com/v1` |
| `--aux-api-key` | `VIDEOLYZER_AUX_API_KEY` | — (required) |

```bash
# One-off
videolyzer video.mp4 --fast \
    --aux-model my-vision-model \
    --aux-base-url https://my-endpoint/v1 \
    --aux-api-key sk-...

# Or via env vars (then just --fast / --verify)
export VIDEOLYZER_AUX_MODEL="my-vision-model"
export VIDEOLYZER_AUX_BASE_URL="https://my-endpoint/v1"
export VIDEOLYZER_AUX_API_KEY="sk-..."
```

## CLI reference

```
videolyzer <source> [--provider gemini|openai|local] [--model NAME]
    [--base-url URL] [--api-key KEY] [--allow-lan]
    [--fps N] [--media-resolution low|medium|high] [--keep-upload]
    [--max-tokens N]
    [--aux-model NAME] [--aux-base-url URL] [--aux-api-key KEY]
    [--fast] [--verify]
    [--out FILE] [--json] [--measure-only]
    [--prompt "..."] [--no-motion] [--no-cuts]
    [--cut-threshold 0.4] [--keyframes 8]
    [--strips] [--strip-frames 8] [--max-strips 24]
    [--lang LANG] [--transcribe] [--stt-backend auto|whispero|http]
    [--stt-lang CODE] [--stt-speakers N] [--hear]
    [--stt-model NAME] [--stt-base-url URL] [--stt-api-key KEY]
    [--no-cache] [--no-resume] [--version]
```

| Flag | Default | Description |
|---|---|---|
| `source` | — | YouTube URL (gemini only) or local file path |
| `--provider` | `gemini` | `gemini` (Google AI Studio), `openai` (any OpenAI-compatible API) or `local` (offline mode, see above) |
| `--model` | `gemini-3.1-flash-lite` | Model name. Required for `openai`: a vision-capable model of your provider. Optional for `local` (or `VIDEOLYZER_LOCAL_MODEL`) |
| `--base-url` | — | API base URL. openai default `https://api.openai.com/v1`; local default `http://127.0.0.1:11434/v1` (or `VIDEOLYZER_LOCAL_BASE_URL`) |
| `--allow-lan` | off | `--provider local`: also admit private-network hosts, not only loopback |
| `--api-key` | — | API key. Or `VIDEOLYZER_API_KEY`, then the provider's own var: `GEMINI_API_KEY`/`GOOGLE_API_KEY` (gemini) or `OPENAI_API_KEY` (openai). Keys are never sent to another provider |
| `--fps` | model default (1) | Gemini only: frames per second the model samples (`video_metadata.fps`). Raise for fast action / subtle motion; tokens scale with it |
| `--media-resolution` | model default | Gemini only: `low` (cheaper, longer videos), `medium`, `high` (small text/details) |
| `--keep-upload` | off | Gemini file mode: keep the uploaded copy (48 h) and reuse it on the next run instead of deleting it right away |
| `--max-tokens` | 4096 / 8192 | Output cap for OpenAI-compatible calls (plain / streamed aux). Models that require `max_completion_tokens` are handled automatically |
| `--json` | off | Emit one JSON object: `ok`, `mode`, `engine`, `fallback`, `measurements` (with per-pass status), `transcript` (structured segments), `report_markdown`, `second_opinion_markdown` (see AGENTS.md) |
| `--measure-only` | off | Local files: only the deterministic passes (facts, cuts, shots, motion; `--strips` adds per-shot timing; `--hear` adds audio; `--transcribe` adds the transcript) as JSON. No vision model, no key |
| `--version` | — | Print the version |
| `--aux-model` | — | Aux multimodal model name (any OpenAI-compatible vision endpoint; or `VIDEOLYZER_AUX_MODEL`) |
| `--aux-base-url` | `https://api.openai.com/v1` | Aux model endpoint (or `VIDEOLYZER_AUX_BASE_URL`) |
| `--aux-api-key` | — | Aux model key (or `VIDEOLYZER_AUX_API_KEY`) |
| `--fast` | off | **Modo Rapido**: full analysis on the aux model via keyframes (no upload). Requires `--aux-model` |
| `--verify` | off | Second opinion: aux model cross-checks the primary report (frame strips when `--strips` is on, timestamped keyframes otherwise) and ends with an overall reliability verdict. Requires `--aux-model` |
| `--out` | stdout | Output file (UTF-8) |
| `--prompt` | built-in | Custom analysis prompt (overrides the default) |
| `--no-motion` | off | Skip the optical-flow subtle-motion pass |
| `--cut-threshold` | 0.4 | Scene-detection threshold for deterministic cut detection. Lower = more sensitive |
| `--no-cuts` | off | Skip deterministic cut detection |
| `--keyframes` | 8 | Keyframes for the openai provider / aux model (cut-aware, timestamp-labelled) |
| `--strips` | off | Aux model (`--fast` and the fallback) gets one image per shot (top row spread across the shot, bottom row consecutive frames) plus locally measured cadence (ones/twos/threes), holds and camera speed. Built for animation review; also `VIDEOLYZER_STRIPS=1`. Warns when set without an aux model |
| `--strip-frames` | 8 | Frames per strip row |
| `--max-strips` | 24 | Max shots sent as strips (evenly sampled, batched 6 per request, then merged) |
| `--lang` | — | Report language, any language name or code (`pt`, `es`, `Japanese`...). Nothing hardcoded |
| `--transcribe` | off | Transcribe audio with a dedicated transcriber and inject it as transcript ground truth, taking the audio load off the vision model |
| `--stt-backend` | `auto` | `auto`: local `whispero` CLI when installed and no HTTP STT setting is given, else HTTP. `whispero`: force the local CLI (faster-whisper + diarization on this machine, timestamped segments with speakers). `http`: force the `/audio/transcriptions` endpoint |
| `--stt-lang` | auto-detect | Language spoken in the video (`en`, `pt`, `pt-BR`...). Not the report language |
| `--stt-speakers` | estimated | whispero: known number of speakers |
| `--hear` | off | Audio measurements via local `whispero hear`: music bed/ducking, tempo, every cut classified on-beat / in-pause / mid-voiced |
| `--stt-model` | `whisper-1` | STT model name (or `VIDEOLYZER_STT_MODEL`). Selects the http backend in auto mode; for whispero use `--stt-backend whispero --stt-model large-v3` |
| `--stt-base-url` | see desc. | STT endpoint base URL (or `VIDEOLYZER_STT_BASE_URL`). With `--provider openai` falls back to the primary base URL, else `https://api.openai.com/v1` |
| `--stt-api-key` | see desc. | STT API key (or `VIDEOLYZER_STT_API_KEY`). The primary key is reused only when the STT endpoint *is* the primary endpoint; `OPENAI_API_KEY` only for api.openai.com |
| `--no-cache` | off | Skip the local cache (cuts, optical flow, per-shot timing, audio measurements and transcripts, per video hash, under `~/.cache/videolyzer` or `$VIDEOLYZER_CACHE_DIR`; created private: dir 0700, files 0600). Failed passes are never cached |
| `--no-resume` | off | Ignore any strips checkpoint and run all batches from scratch (checkpoints live in the cache dir and are deleted after a successful merge) |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | Usage / input error (bad flag value, missing file, unsupported combination) |
| 3 | Missing API key |
| 4 | Runtime error |
| 5 | The model returned nothing usable |

Progress and diagnostics go to stderr; stdout carries only the result. Markdown reports end with an invisible provenance comment (`<!-- videolyzer … | mode=… | engine=… | fallback=… -->`).

## Requirements

- Python 3.10+
- `ffmpeg` / `ffprobe` on PATH (system package)
- `opencv-python` + `numpy` (installed with the package) for the subtle-motion and per-shot timing passes. The core path only needs the stdlib + ffmpeg: if opencv is missing, those passes are skipped and reported as `unavailable`.
- Optional: `pip install "videolyzer[transcribe]"` (= `whispero[local]`, local transcription), `"videolyzer[hear]"` (= `whispero[hear]`, audio measurements); a local OpenAI-compatible vision server for `--provider local --model ...`.

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q    # synthetic ffmpeg videos + a fake local API; no network
ruff check .
```

## Report format

```markdown
# Video Analysis
## Summary
## Scenes            (| # | Start | End | Visual description | Audio/speech |)
## Aligned transcript
## Key moments
## Tone and emotion
## Temporal verification   (motion described with action verbs, cause-effect order checked)
```

## How it differs from frame extraction

Classic pipelines extract N frames and ask a vision model about each still image. That loses motion: a bouncing ball becomes "ball at left" / "ball at right", and anything between sampled frames is invisible (or hallucinated).

Videolyzer instead hands the *video* to the model:

1. **Native temporal sampling** — the model, not our code, decides what to look at along the timeline.
2. **Temporal verification** — the prompt forces motion verbs per scene and cause-effect order checks, plus an explicit "motion undetermined" escape hatch instead of invention.
3. **Hybrid subtle-motion pass** (local files) — classical CV finds what the model's sampling might skip, and the model is told exactly where to look.

See `docs/` for the comparative tests that motivated this design.

## License

Apache 2.0 — see [LICENSE](LICENSE).
