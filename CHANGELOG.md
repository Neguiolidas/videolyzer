# Changelog

## 0.3.0 (unreleased)

Audit release: security, correctness, a working Whispero integration, an
offline mode, better "vision" for the models and a machine-readable
interface for agents. Full findings: `docs/audit-2026-11.md`.

Security / privacy
- Gemini API key moved from the URL query string (`?key=`) to the
  `x-goog-api-key` header: it no longer leaks into proxy logs or errors.
- Provider-scoped key lookup: `GEMINI_API_KEY`/`GOOGLE_API_KEY` are never
  sent to an OpenAI-compatible endpoint and `OPENAI_API_KEY` never to
  Gemini. `--transcribe` reuses the primary key only for the primary
  endpoint (previously a Gemini key could be sent to api.openai.com).
- HTTP redirects are refused (urllib re-sent `Authorization` /
  `x-goog-api-key` to whatever host a 301/302/303 pointed at).
- Warning when a key would travel over plain `http://` to a remote host.
- The uploaded video is deleted from the Gemini Files API after the
  analysis (it stayed 48 h); `--keep-upload` keeps and reuses it.
- Cache dir created 0700, entries 0600 (it now holds transcripts).
- whispero runs without cloud credentials in its environment.
- Video-derived text (transcript, primary report under review) is delimited
  and marked as data in prompts (prompt-injection hygiene).
- Inputs sanitized: absolute source paths for ffmpeg/ffprobe/opencv (a file
  named `-i.mp4` or `concat:x` is never an option/protocol), Gemini model
  names and API-returned file names validated before entering URLs,
  multipart values stripped of CR/LF, `--out` refuses to overwrite the input.

Whispero integration (was non-functional)
- Before: `whispero <wav> --format srt` without `--diarize` made whispero
  write plain text (SRT needs `--diarize`), the SRT parser found nothing and
  the transcript was always dropped; without `--provider local` whispero
  used its default cloud Gemini provider; without `--lang` it assumed
  Portuguese.
- Now: `whispero <audio> --provider local --diarize --format json --lang ""`
  (auto-detect, or `--stt-lang`): timestamped segments with speaker labels,
  fully local. `--stt-speakers`, model via `--stt-backend whispero
  --stt-model ...`. Live progress relay, clear hints for a missing
  `whispero[local]` / too-old whispero / no speech, timeout scaled to the
  video length, transcripts cached.
- `--hear`: `whispero hear --music --beats --cuts <measured cuts>`: music
  bed/ducking, tempo, every cut classified on-beat / in-pause / mid-voiced.
- Transcript exposed as structured JSON (`segments` with start/end/speaker).
- HTTP STT: audio sent as 32 kbps MP3 (fits the usual 25 MB cap for about
  1 h 45 min instead of ~13 min of WAV); `language` passed when known.

Offline mode (`--provider local`)
- Every request must target a loopback host (or a private network with
  `--allow-lan`); enforced in the single HTTP gate, aux and STT included.
- Local OpenAI-compatible vision server (Ollama default URL, LM Studio,
  llama.cpp, vLLM), keyframes or strips, no key needed.
- Without a model, or when the local server is down: deterministic
  measured report with the same section headings.

Fixes
- Motion pass: frame pairs straddling a hard cut are dropped and end the
  current segment. Before, every cut produced a bogus motion segment that
  was injected into the prompt as confirmed motion.
- Cut timestamps are relative to the first frame (`setpts=PTS-STARTPTS`):
  MPEG-TS files starting at 1.4 s+ no longer shift every cut.
- MPEG-TS frames: `-ss` lands on the next keyframe there (frames off by a
  whole GOP); frames now come from a one-time stream-copy MP4.
- Gemini: all non-thought text parts joined (thinking models could lose the
  report); `MAX_TOKENS`/`SAFETY` and blocked prompts reported; containers it
  does not document (mkv, ts, ogv) remuxed to MP4; `.ts`/`.3gp` no longer
  sent with a wrong MIME type.
- OpenAI-compatible: automatic retry with `max_completion_tokens`;
  truncation warned; list-shaped `content` handled.
- HTTP: retries cover 408/504/524 and honour `Retry-After`; a response that
  arrived (even non-JSON) is never re-sent (no double billing / re-upload).
- Gemini upload streamed from disk instead of loaded in RAM.
- Cache: failures (tool missing, decode error) are no longer cached as empty
  results; keys are collision-free; JSON written as UTF-8 (non-ASCII strip
  checkpoints crashed on Windows); entries versioned.
- ffprobe output decoded as UTF-8 with replacement (non-UTF-8 metadata
  crashed the run); `N/A` durations and bogus VFR frame rates handled.
- `--max-strips 1` no longer divides by zero; strip checkpoints keyed by
  model/lang/prompt/transcript.
- Timestamps `h:mm:ss` from one hour on (was `75:00`).
- Reports written as UTF-8; an unwritable `--out` falls back to stdout.

Vision
- Keyframes labelled with exact timestamps and placed cut-aware (every
  shot gets a frame when they fit; never on a cut); long side capped at
  1024 px, never upscaled.
- Video facts (duration, resolution, fps, audio) in every file-mode prompt.
- `--fps N` (Gemini `video_metadata.fps`) and `--media-resolution`; video
  part before the text prompt, as Google advises.
- Gemini upload/processing overlaps with the local passes.
- `--verify` sees the measured cuts, ends with an overall verdict, and is
  skipped when the aux model wrote the report itself.

Agent interface
- `--json` envelope (ok, mode, engine, fallback, measurements with per-pass
  status, structured transcript, report, second opinion); JSON on failure.
- `--measure-only`: deterministic measurements as JSON, no model, no key.
- Stable exit codes (0/2/3/4/5/130), `--version`, provenance comment.
- `AGENTS.md` (+ `CLAUDE.md`), 114 tests (synthetic videos, fake API, fake
  whispero), CI on Python 3.10 and 3.13; the release workflow runs them.

## 0.2.0

- Frame strips mode (`--strips`): the aux model gets one ordered frame strip
  per shot (top row spread across the shot, bottom row consecutive frames)
  plus locally measured cadence (ones/twos/threes), holds and camera
  numbers, instead of evenly spaced keyframes. Batched and merged, with
  per-batch checkpointing and resume.
- `--verify` uses frame strips when `--strips` is active.
- Local measurement cache: cut detection, optical-flow and per-shot timing
  results are cached per video under `~/.cache/videolyzer`
  (`--no-cache` to skip).
- `--lang`: report language, any language name or code, nothing hardcoded.
- `--transcribe`: dedicated transcriber injects the transcript as ground
  truth, taking the audio load off the vision model. Backends
  (`--stt-backend auto|whispero|http`): the locally installed `whispero`
  CLI, or any OpenAI-compatible `/audio/transcriptions` endpoint.
- Warns when `--strips` is set without an auxiliary model configured.

## 0.1.0

Initial release: native Gemini video analysis (URL and file modes),
OpenAI-compatible keyframe analysis, deterministic cut detection,
optical-flow subtle-motion hints, auxiliary model (fast mode, second
opinion, fallback).
