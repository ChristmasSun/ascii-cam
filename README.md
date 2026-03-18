# ASCII Webcam Renderer

Live ASCII terminal renderer of Webcam. Transforms webcam feed into live ASCII art. 13 rendering modes, color support, and a bunch of other effects.

## Quick Start

```bash
uv run ascii_cam.py
```

Requires Python 3.12+ and a webcam. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

## Rendering Modes

Cycle through modes with `m`, `←`, `→`.

| Mode | Description |
|------|-------------|
| **ascii** | Classic ASCII character ramp (` .:-=+*#%@`) |
| **block** | Unicode block characters (`░▒▓█`) |
| **braille** | Braille dot patterns — 8x resolution per character cell |
| **outline** | Pure edge detection, clean line-art contours |
| **dither** | Bayer ordered dithering — halftone newspaper look |
| **matrix** | Green katakana rain shaped by your silhouette |
| **emoji** | Colored emoji squares mapped to brightness |
| **stipple** | Pointillism with dots and spaces |
| **pixel** | Solid color/grayscale blocks — low-res pixel display |
| **thermal** | False color heatmap (blue → cyan → green → yellow → red → white) |
| **halfblock** | Half-block characters (`▀`) for 2x vertical resolution |
| **mirror** | ASCII rendering with mirror/kaleidoscope effects |
| **hd** | 24-bit true color with `▀` — highest quality, closest to actual webcam |

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `m` / `→` | Next mode |
| `←` | Previous mode |
| `c` | Toggle color on/off |
| `t` | Cycle color tint (full / green / amber / blue / sepia) |
| `i` | Invert brightness |
| `e` | Cycle edge detection strength (off → 0.3 → 0.6 → 1.0 → off) |
| `a` | Toggle adaptive character ramp |
| `b` | Toggle background subtraction |
| `k` | Cycle mirror/kaleidoscope (off → horizontal → vertical → quad) |
| `[` / `]` | Decrease / increase brightness |
| `{` / `}` | Decrease / increase contrast |
| `0` | Reset brightness and contrast |

## Features

- **13 rendering modes** from classic ASCII to true-color HD
- **256-color and 24-bit true color** support
- **Color tints** — full color, green terminal, amber CRT, blue cyberpunk, sepia
- **Edge detection** blending for sharper facial features
- **Temporal smoothing** to reduce flickering noise
- **Background subtraction** to isolate your silhouette
- **Mirror/kaleidoscope** effects (horizontal, vertical, 4-way)
- **Brightness/contrast controls** adjustable on the fly
- **Auto-detects dark/light terminal** background
- **Live status bar** showing all current settings and FPS
- **CLAHE contrast enhancement** for better detail in varying lighting

## Dependencies

- [OpenCV](https://opencv.org/) — webcam capture and image processing
- [NumPy](https://numpy.org/) — fast array operations
