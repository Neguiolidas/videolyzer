# Changelog

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
