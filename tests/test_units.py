"""Pure-function tests (no ffmpeg, no network)."""
from __future__ import annotations

import argparse
import os
import re

import pytest

import videolyzer as v


def test_version_matches_pyproject():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "pyproject.toml"), encoding="utf-8") as f:
        m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
    assert m and m.group(1) == v.__version__


@pytest.mark.parametrize("secs,expected", [(0, "00:00"), (59.9, "00:59"),
                                           (61, "01:01"), (3599, "59:59"),
                                           (3600, "1:00:00"), (3725, "1:02:05"),
                                           (-3, "00:00"), (float("nan"), "00:00")])
def test_fmt_ts(secs, expected):
    assert v.fmt_ts(secs) == expected


@pytest.mark.parametrize("rate,expected", [("24/1", 24.0), ("30000/1001", 29.97),
                                           ("25", 25.0), ("0/0", 24.0),
                                           (None, 24.0), ("bad", 24.0)])
def test_parse_fps(rate, expected):
    assert v._parse_fps(rate) == pytest.approx(expected, abs=0.01)


def test_shot_list_merges_flash_frames():
    assert v.shot_list(10.0, [2.0, 2.1, 5.0, 9.9]) == [(0.0, 2.0), (2.0, 5.0),
                                                      (5.0, 10.0)]
    assert v.shot_list(4.0, []) == [(0.0, 4.0)]


def test_keyframe_times_cover_every_shot():
    cuts = [1.0, 1.5, 9.0]  # a 0.5 s shot that even spacing would skip
    times = v.keyframe_times(10.0, 6, cuts)
    assert len(times) == 6 and times == sorted(times)
    for s, e in v.shot_list(10.0, cuts):
        assert any(s <= t < e for t in times), (s, e, times)
    # The long middle shot (1.5-9.0) gets the spare frames.
    assert sum(1 for t in times if 1.5 <= t < 9.0) == 3


def test_keyframe_times_even_and_nudged_off_cuts():
    assert v.keyframe_times(8.0, 4) == [1.0, 3.0, 5.0, 7.0]
    times = v.keyframe_times(8.0, 3, [2.0, 4.0, 6.0])  # more shots than frames
    assert all(min(abs(t - c) for c in (2.0, 4.0, 6.0)) >= 0.1 for t in times)


def test_keyframe_times_degenerate():
    assert v.keyframe_times(0, 3) == [0.0, 1.0, 2.0]
    assert v.keyframe_times(5.0, 0) == [2.5]


def test_sample_order():
    assert v._sample_order(5, 10) == [0, 1, 2, 3, 4]
    assert v._sample_order(10, 3) == [0, 4, 9]  # used to divide by k-1
    assert v._sample_order(10, 1) == [5]


def _samples(mags, dt=0.5):
    return [[i * dt, (i + 1) * dt, m, 1, 1, m, 0.0] for i, m in enumerate(mags)]


def test_segments_from_samples_flags_signal_and_axis():
    mags = [0.1] * 10 + [3.0, 3.2, 3.1] + [0.1] * 10
    segs = v.segments_from_samples(_samples(mags), 0.5)
    assert len(segs) == 1
    assert segs[0]["axis"] == "horizontal" and segs[0]["region"] == "center"
    assert segs[0]["start"] == pytest.approx(5.0) and segs[0]["end"] == pytest.approx(7.0)


def test_segments_from_samples_drops_pairs_across_cuts():
    mags = [0.1] * 10 + [9.0] + [0.1] * 10  # spike = picture change at a cut
    raw = _samples(mags)
    assert v.segments_from_samples(raw, 0.5)  # without cuts: bogus segment
    cut = raw[10][1]  # pair (5.0, 5.5]: new shot starts at 5.5
    assert v.segments_from_samples(raw, 0.5, [cut]) == []


def test_build_prompt_sections_and_facts():
    info = {"duration": 83.4, "width": 1920, "height": 1080, "fps": "24/1",
            "has_audio": True}
    p = v.build_prompt([{"start": 1, "end": 2, "region": "center",
                         "axis": "vertical", "magnitude": 0.5}],
                       "BASE", cuts=[2.5], transcript="[00:01] hi", video=info)
    assert p.startswith("BASE")
    for needle in ("01:23 (83.40s)", "1920x1080", "cut at 00:02 (2.50s)",
                   "[00:01] hi", "region: center"):
        assert needle in p
    assert v.build_prompt([], "BASE") == "BASE"


def test_with_lang():
    assert v._with_lang("x", None) == "x"
    assert v._with_lang("x", " pt ").endswith("Write the entire report in pt.")


def test_multipart_body_roundtrip():
    body, boundary = v._multipart_body({"model": "m"}, "file", "a.wav", b"RIFF")
    assert body.startswith(f"--{boundary}".encode())
    assert b'name="model"\r\n\r\nm\r\n' in body and body.endswith(
        f"\r\n--{boundary}--\r\n".encode())


def test_parse_gemini_response_joins_parts_and_skips_thoughts():
    data = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
        {"text": "thinking...", "thought": True},
        {"text": "# Video "}, {"text": "Analysis"}]}}]}
    assert v.parse_gemini_response(data) == "# Video Analysis"


def test_parse_gemini_response_blocked(capsys):
    assert v.parse_gemini_response({"promptFeedback": {"blockReason": "SAFETY"}}) is None
    assert "blocked: SAFETY" in capsys.readouterr().err


def test_gemini_video_part():
    assert v.gemini_video_part("u") == {"file_data": {"file_uri": "u"}}
    assert v.gemini_video_part("u", "video/mp4", 4) == {
        "file_data": {"file_uri": "u", "mime_type": "video/mp4"},
        "video_metadata": {"fps": 4}}


def _args(**kw):
    base = dict(provider="gemini", api_key=None, base_url=None, transcribe=True,
                stt_model=None, stt_base_url=None, stt_api_key=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_primary_key_is_scoped_per_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    assert v.find_api_key(_args(provider="gemini")) == "g-key"
    # A Google key must never be picked up for an OpenAI-compatible endpoint.
    assert v.find_api_key(_args(provider="openai")) is None
    monkeypatch.setenv("OPENAI_API_KEY", "o-key")
    assert v.find_api_key(_args(provider="openai")) == "o-key"


def test_stt_never_reuses_gemini_key_for_openai(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    with pytest.raises(SystemExit) as e:
        v.get_stt_config(_args(provider="gemini", api_key="g-key"))
    assert e.value.code == v.EXIT_AUTH


def test_stt_reuses_primary_key_only_for_primary_endpoint(monkeypatch):
    cfg = v.get_stt_config(_args(provider="openai", api_key="groq",
                                 base_url="https://api.groq.com/openai/v1"))
    assert cfg == {"model": "whisper-1", "api_key": "groq",
                   "base_url": "https://api.groq.com/openai/v1"}
    with pytest.raises(SystemExit):  # different STT host: no key reuse
        v.get_stt_config(_args(provider="openai", api_key="groq",
                               base_url="https://api.groq.com/openai/v1",
                               stt_base_url="https://stt.example/v1"))


def test_retry_delay_honours_retry_after():
    class E(Exception):
        headers = {"Retry-After": "7"}
    assert v._retry_delay(0, E()) == 7
    E.headers = {"Retry-After": "9999"}
    assert v._retry_delay(0, E()) == v.RETRY_AFTER_CAP
    assert v._retry_delay(0, None) == v.RETRY_DELAYS[0]


def test_http_request_refuses_non_http():
    data, status, err = v.http_request("file:///etc/passwd", {})
    assert data is None and status is None and "refusing" in err


def test_validate_args():
    ns = v.build_parser().parse_args(["x.mp4", "--keyframes", "0"])
    assert "keyframes" in v._validate_args(ns)
    ns = v.build_parser().parse_args(["x.mp4", "--provider", "openai"])
    assert "--model" in v._validate_args(ns)
    ns = v.build_parser().parse_args(["x.mp4", "--cut-threshold", "2"])
    assert "cut-threshold" in v._validate_args(ns)
    assert v._validate_args(v.build_parser().parse_args(["x.mp4"])) is None


def test_file_hash_memoized(tmp_path, monkeypatch):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    h1 = v._file_hash(str(f))
    opened = []
    real_open = open
    monkeypatch.setattr("builtins.open",
                        lambda *a, **k: opened.append(a) or real_open(*a, **k))
    assert v._file_hash(str(f)) == h1 and not opened


def test_motion_segment_never_spans_a_cut():
    # Motion right before AND right after a cut: two segments, not one.
    mags = [0.1] * 10 + [3.0, 3.0, 9.0, 3.0, 3.0] + [0.1] * 10
    raw = _samples(mags)
    cut = raw[12][1]
    segs = v.segments_from_samples(raw, 0.5, [cut])
    assert len(segs) == 2
    assert segs[0]["end"] <= cut + 0.5 and segs[1]["start"] >= cut - 0.5


def test_multipart_values_cannot_inject_headers():
    body, _ = v._multipart_body({"model": "m\r\nX-Evil: 1"}, "f", "a.wav", b"")
    assert b"\r\nX-Evil" not in body


def test_is_url_case_insensitive():
    assert v.is_url("HTTPS://youtu.be/x") and not v.is_url("./https.mp4")


def test_aux_key_optional_only_for_local_endpoints(monkeypatch):
    ns = v.build_parser().parse_args(["x.mp4", "--aux-model", "a",
                                      "--aux-base-url", "http://127.0.0.1:1234/v1"])
    assert v.get_aux_config(ns)["api_key"] == "local"
    ns = v.build_parser().parse_args(["x.mp4", "--aux-model", "a",
                                      "--aux-base-url", "https://203.0.113.9/v1"])
    with pytest.raises(SystemExit):
        v.get_aux_config(ns)


def test_plaintext_key_warning_once(capsys, monkeypatch):
    monkeypatch.setattr(v, "_WARNED_PLAINTEXT", set())
    for _ in range(2):
        v._warn_plaintext_credentials("http://203.0.113.9/v1", {"Authorization": "k"})
    v._warn_plaintext_credentials("http://127.0.0.1/v1", {"Authorization": "k"})
    v._warn_plaintext_credentials("https://203.0.113.9/v1", {"Authorization": "k"})
    assert capsys.readouterr().err.count("plain http://") == 1


def test_cache_files_are_private(tmp_path, monkeypatch):
    import stat
    d = tmp_path / "fresh-cache"
    monkeypatch.setenv(v.CACHE_ENV, str(d))
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    v._cached("k", str(f), "p", lambda: {"x": 1})
    (entry,) = list(d.iterdir())
    assert stat.S_IMODE(d.stat().st_mode) == 0o700
    assert stat.S_IMODE(entry.stat().st_mode) == 0o600
