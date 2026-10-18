# Cut Detection Test

## Method
`detect_cuts()` in `videolyzer.py` uses ffmpeg scene detection:
```
ffmpeg -i input.mp4 -filter:v "select='gt(scene,0.4)',showinfo" -f null -
```
Parses `pts_time` from showinfo output. Threshold configurable via `--cut-threshold` (default 0.4), skip with `--no-cuts`.

## Test
Synthetic video: 5 segments × 2s each (test pattern / black / test pattern / white / test pattern), 10s total. Ground truth cuts at 2.0, 4.0, 6.0, 8.0.

| Threshold | Detected | Result |
|-----------|----------|--------|
| 0.4 (default) | 2.00s, 4.00s, 6.00s, 8.00s | ✅ 4/4 exact |

## Known limitation
Solid flat colors (e.g. pure red → pure green with no texture) may not trigger the scene detector at default threshold — ffmpeg's scene metric needs pixel variance. Real footage (the actual use case) works correctly. Lower `--cut-threshold` if needed for synthetic/minimal content.

## URL mode
Cut detection is unavailable in URL mode (requires the file for ffmpeg). Documented in README and printed as a note at runtime.
