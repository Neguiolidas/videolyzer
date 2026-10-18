# Videolyzer

**Give agents the ability to understand and deep analyze videos.**

A single CLI that turns a video (YouTube URL or local file) into a structured Markdown report: detected scenes with timestamps, transcript aligned per scene, key moments, tone/emotion, and temporal verification. Built for agents that need to *watch* video, not just transcribe it.

## Quick start

```bash
pip install -r requirements.txt   # opencv + numpy (only needed for the subtle-motion pass)

export VIDEOLYZER_API_KEY="your-key"

# URL mode (gemini only, no download)
python videolyzer.py "https://www.youtube.com/watch?v=..." --out report.md

# Local file mode
python videolyzer.py ./my-video.mp4 --out report.md

# Any OpenAI-compatible provider (Groq, Together, OpenAI, ...)
python videolyzer.py ./my-video.mp4 --provider openai --model gpt-4o \
    --base-url https://api.groq.com/openai/v1 --out report.md

# Auxiliary model (any OpenAI-compatible vision endpoint)
export VIDEOLYZER_AUX_MODEL="your-vision-model"
export VIDEOLYZER_AUX_BASE_URL="https://your-endpoint/v1"
export VIDEOLYZER_AUX_API_KEY="your-key"

# Modo Rapido: fast keyframe analysis on the aux model (no full upload)
python videolyzer.py ./my-video.mp4 --fast --out report.md

# Second opinion: primary analysis + aux model cross-check
python videolyzer.py ./my-video.mp4 --verify --out report.md
```

## Modes: URL vs local file

| | URL mode | Local file mode |
|---|---|---|
| Providers | `gemini` only | `gemini` and `openai` |
| Download | None. The URL goes straight to the model (`file_data`) | None for gemini (full upload via Files API); keyframes extracted locally for openai |
| Temporal understanding | Native: the model samples the timeline itself | Native (gemini) / keyframe-based (openai) |
| Subtle-motion detection | **Not available** (would require downloading the video) | Available: optical-flow pass flags small-amplitude motion segments and injects them into the prompt as an explicit hint |
| Speed | Fastest | Slower (upload + optional motion pass) |

### URL mode (gemini only)

No download, no frame extraction. The video URL is passed to the model and it does its own temporal sampling. Best for quick overviews of online videos. Limitation: subtle motion (small-amplitude movement) can be missed, because the pre-analysis optical-flow pass cannot run without the pixels.

### Local file mode

The file is validated with `ffprobe` (codec, duration, integrity — no frames extracted), then:

- **gemini:** the whole file is uploaded through the Files API; the model samples the timeline itself. Before the upload, an OpenCV pass (Farneback optical flow, 2 fps sampling, 3×3 grid) measures motion magnitude per segment and flags anything above background noise (median + 4·1.4826·MAD, robust to outliers). The flagged timestamps, region and dominant axis go into the prompt as an explicit hint ("do NOT say these segments are static").
- **openai (any OpenAI-compatible API):** `N` evenly spaced keyframes (`--keyframes`, default 8) are extracted with ffmpeg and sent as images through `/v1/chat/completions` (vision). The same optical-flow pass runs and feeds the same hint section into the prompt. URL input is not supported with this provider.

## Auxiliary model & Modo Rapido

The auxiliary model is a second, independent multimodal model — any OpenAI-compatible vision endpoint (no provider hardcoded). It plays three roles:

| Role | How |
|---|---|
| **Second opinion** (`--verify`) | After the primary analysis, the aux model reviews the report against keyframes: agrees/disagrees per scene, flags temporal-order mistakes. Appended as a "Second opinion" section. |
| **Temporal verifier** | Built into every aux pass: the prompt forces motion verbs and event-order checks across keyframes. |
| **Fallback** | If the primary analysis fails and an aux model is configured, Videolyzer automatically retries with the aux model on keyframes. |

**Modo Rapido** (`--fast`) skips the full video upload entirely and runs the whole analysis on the auxiliary model with evenly spaced keyframes (+ the optical-flow subtle-motion hints). Much faster and cheaper than the native path — best when only "what appears" matters and fine motion detail is not critical. Requires the aux model to be configured.

Configuration (flags take precedence over env vars):

| Flag | Env var | Default |
|---|---|---|
| `--aux-model` | `VIDEOLYZER_AUX_MODEL` | — (required to enable) |
| `--aux-base-url` | `VIDEOLYZER_AUX_BASE_URL` | `https://api.openai.com/v1` |
| `--aux-api-key` | `VIDEOLYZER_AUX_API_KEY` | — (required) |

```bash
# One-off
python videolyzer.py video.mp4 --fast \
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
videolyzer.py <source> [--provider gemini|openai] [--model NAME]
    [--base-url URL] [--api-key KEY]
    [--aux-model NAME] [--aux-base-url URL] [--aux-api-key KEY]
    [--fast] [--verify]
    [--out report.md] [--prompt "..."] [--no-motion] [--keyframes 8]
```

| Flag | Default | Description |
|---|---|---|
| `source` | — | YouTube URL (gemini only) or local file path |
| `--provider` | `gemini` | `gemini` (Google AI Studio) or `openai` (any OpenAI-compatible API) |
| `--model` | `gemini-3.1-flash-lite` | Model name. For `openai`, pass a vision-capable model of your provider |
| `--base-url` | — | Custom API base URL (openai provider). Default `https://api.openai.com/v1` |
| `--api-key` | — | API key. Or set `VIDEOLYZER_API_KEY` (`GEMINI_API_KEY` / `OPENAI_API_KEY` also work) |
| `--aux-model` | — | Aux multimodal model name (any OpenAI-compatible vision endpoint; or `VIDEOLYZER_AUX_MODEL`) |
| `--aux-base-url` | `https://api.openai.com/v1` | Aux model endpoint (or `VIDEOLYZER_AUX_BASE_URL`) |
| `--aux-api-key` | — | Aux model key (or `VIDEOLYZER_AUX_API_KEY`) |
| `--fast` | off | **Modo Rapido**: full analysis on the aux model via keyframes (no upload). Requires `--aux-model` |
| `--verify` | off | Second opinion: aux model cross-checks the primary report. Requires `--aux-model` |
| `--out` | stdout | Output `.md` file |
| `--prompt` | built-in | Custom analysis prompt (overrides the default) |
| `--no-motion` | off | Skip the optical-flow subtle-motion pass |
| `--keyframes` | 8 | Keyframes for the openai provider / aux model |

## Requirements

- Python 3.10+
- `ffmpeg` / `ffprobe` on PATH (system package)
- `pip install -r requirements.txt` → `opencv-python`, `numpy` (only needed for the subtle-motion pass; the tool degrades gracefully without them)

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
