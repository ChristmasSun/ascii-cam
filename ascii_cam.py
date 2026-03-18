#!/usr/bin/env python3
"""Live ASCII art webcam renderer for terminal."""

import curses
import os
import random
import sys
import time

import cv2
import numpy as np

# Character ramps (dark-to-bright for dark backgrounds, reversed for light)
SHORT_CHARS_DARK = " .:-=+*#%@"
SHORT_CHARS_LIGHT = "@%#*+=:-. "
BLOCK_CHARS = " ░▒▓█"
EMOJI_CHARS = ["  ", "🟫", "🟥", "🟧", "🟨", "🟩", "🟦", "🟪", "⬜"]
BRAILLE_BASE = 0x2800
MATRIX_CHARS = list("ｦｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾗﾘﾙﾚﾛﾜﾝ0123456789")

COLOR_TINTS = ["full", "green", "amber", "blue", "sepia"]

TARGET_FPS = 30

MODES = [
    "ascii", "block", "braille", "outline", "dither", "matrix",
    "emoji", "stipple", "pixel", "thermal", "halfblock", "mirror", "hd",
]

# Mirror sub-modes
MIRROR_MODES = [None, "h", "v", "quad"]
MIRROR_LABELS = ["off", "horiz", "vert", "quad"]


def detect_dark_background():
    """Detect if terminal has a dark background."""
    colorfgbg = os.environ.get("COLORFGBG", "")
    if colorfgbg:
        parts = colorfgbg.split(";")
        try:
            return int(parts[-1]) < 8
        except (ValueError, IndexError):
            pass
    return True


def measure_char_density():
    """Measure actual pixel density of printable ASCII chars using OpenCV."""
    chars_with_density = []
    for c in range(32, 127):
        ch = chr(c)
        img = np.zeros((20, 14), dtype=np.uint8)
        cv2.putText(img, ch, (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        density = np.sum(img > 0) / img.size
        chars_with_density.append((ch, density))
    chars_with_density.sort(key=lambda x: x[1])
    return np.array([c for c, _ in chars_with_density])


def init_all_color_pairs():
    """Initialize all color pairs we need."""
    # 1-216: 6x6x6 color cube (fg on default bg)
    for i in range(1, 217):
        curses.init_pair(i, 15 + i, -1)
    # 217-240: grayscale fg=bg pairs for pixel mode
    for i in range(24):
        curses.init_pair(217 + i, 232 + i, 232 + i)
    # 241-250: special pairs
    curses.init_pair(241, curses.COLOR_GREEN, -1)   # matrix green
    curses.init_pair(242, curses.COLOR_WHITE, -1)    # matrix white
    curses.init_pair(243, curses.COLOR_BLUE, -1)     # thermal/tint blue
    curses.init_pair(244, curses.COLOR_CYAN, -1)     # thermal cyan
    curses.init_pair(245, curses.COLOR_GREEN, -1)    # thermal green
    curses.init_pair(246, curses.COLOR_YELLOW, -1)   # thermal yellow
    curses.init_pair(247, curses.COLOR_RED, -1)      # thermal red
    curses.init_pair(248, curses.COLOR_WHITE, -1)    # thermal white
    curses.init_pair(249, 208, -1)                   # amber tint
    curses.init_pair(250, 180, -1)                   # sepia tint


def rgb_to_256(r, g, b):
    """Map RGB 0-255 to 256-color terminal index."""
    ri = min(5, int(r) * 6 // 256)
    gi = min(5, int(g) * 6 // 256)
    bi = min(5, int(b) * 6 // 256)
    return 16 + 36 * ri + 6 * gi + bi


def rgb_to_color_pair_vectorized(r_arr, g_arr, b_arr):
    """Vectorized RGB to color pair for entire frame."""
    ri = np.minimum(5, r_arr.astype(np.int16) * 6 // 256)
    gi = np.minimum(5, g_arr.astype(np.int16) * 6 // 256)
    bi = np.minimum(5, b_arr.astype(np.int16) * 6 // 256)
    pairs = 16 + 36 * ri + 6 * gi + bi - 15
    return np.clip(pairs, 1, 216).astype(np.int16)


def brightness_to_grayscale_pair(val):
    """Map brightness 0-255 to grayscale color pair (217-240)."""
    return 217 + int(val) * 23 // 255


def brightness_to_thermal_pair(val):
    """Map brightness to thermal color (blue->cyan->green->yellow->red->white)."""
    v = int(val)
    if v < 43:
        return 243
    elif v < 86:
        return 244
    elif v < 128:
        return 245
    elif v < 170:
        return 246
    elif v < 213:
        return 247
    return 248


TINT_PAIRS = {"green": 241, "amber": 249, "blue": 243, "sepia": 250}


def render_braille(gray, render_w, render_h, threshold=128):
    """Render grayscale as braille unicode characters."""
    bw, bh = render_w * 2, render_h * 4
    resized = cv2.resize(gray, (bw, bh))
    binary = resized > threshold
    rows = []
    for cy in range(render_h):
        row = []
        py = cy * 4
        for cx in range(render_w):
            px = cx * 2
            code = BRAILLE_BASE
            if binary[py, px]:     code |= 0x01
            if binary[py+1, px]:   code |= 0x02
            if binary[py+2, px]:   code |= 0x04
            if binary[py+3, px]:   code |= 0x40
            if binary[py, px+1]:   code |= 0x08
            if binary[py+1, px+1]: code |= 0x10
            if binary[py+2, px+1]: code |= 0x20
            if binary[py+3, px+1]: code |= 0x80
            row.append(chr(code))
        rows.append("".join(row))
    return rows


def ordered_dither(gray, gamma=1.8):
    """8x8 Bayer ordered dithering with gamma correction."""
    bayer8 = np.array([
        [ 0,32, 8,40, 2,34,10,42], [48,16,56,24,50,18,58,26],
        [12,44, 4,36,14,46, 6,38], [60,28,52,20,62,30,54,22],
        [ 3,35,11,43, 1,33, 9,41], [51,19,59,27,49,17,57,25],
        [15,47, 7,39,13,45, 5,37], [63,31,55,23,61,29,53,21],
    ], dtype=np.float32) / 64.0
    h, w = gray.shape
    threshold = np.tile(bayer8, (h // 8 + 1, w // 8 + 1))[:h, :w]
    normalized = gray.astype(np.float32) / 255.0
    return np.power(normalized, gamma) > threshold


def apply_mirror(img, mirror_mode):
    """Apply mirror/kaleidoscope to an image."""
    h, w = img.shape[:2]
    out = img.copy()
    if mirror_mode == "h":
        left = out[:, :w // 2]
        out[:, w // 2:w // 2 + left.shape[1]] = cv2.flip(left, 1)
    elif mirror_mode == "v":
        top = out[:h // 2, :]
        out[h // 2:h // 2 + top.shape[0], :] = cv2.flip(top, 0)
    elif mirror_mode == "quad":
        q = out[:h // 2, :w // 2]
        out[:h // 2, w // 2:w // 2 + q.shape[1]] = cv2.flip(q, 1)
        top_half = out[:h // 2, :]
        out[h // 2:h // 2 + top_half.shape[0], :] = cv2.flip(top_half, 0)
    return out


def get_color_attr(y, x, color_pairs, color_enabled, tint_index):
    """Get curses color attribute for a position."""
    if not color_enabled:
        return 0
    if tint_index == 0 and color_pairs is not None:
        return curses.color_pair(int(color_pairs[y, x]))
    elif tint_index > 0:
        tint_name = COLOR_TINTS[tint_index]
        return curses.color_pair(TINT_PAIRS.get(tint_name, 0))
    return 0


def render_row_with_color(stdscr, y, row_str, color_enabled, color_pairs, tint_index, render_w):
    """Render a row, optionally with color."""
    if color_enabled:
        for x in range(min(len(row_str), render_w)):
            attr = get_color_attr(y, x, color_pairs, color_enabled, tint_index)
            stdscr.addstr(y, x, row_str[x], attr)
    else:
        stdscr.addstr(y, 0, row_str)


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    curses.start_color()
    curses.use_default_colors()

    dark_bg = detect_dark_background()

    color_enabled = False
    color_available = False
    try:
        init_all_color_pairs()
        color_available = True
    except curses.error:
        pass

    ascii_chars = np.array(list(SHORT_CHARS_DARK if dark_bg else SHORT_CHARS_LIGHT))
    num_chars = len(ascii_chars)
    use_adaptive = False

    mode_index = 0
    tint_index = 0
    brightness_offset = 0
    contrast_factor = 1.0
    mirror_index = 0
    inverted = False
    edge_blend = 0.0
    bg_sub_enabled = False
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=120, varThreshold=50, detectShadows=False
    )

    fps_display = 0.0
    fps_update_timer = 0.0
    prev_gray = None
    matrix_grid = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        curses.endwin()
        print("ERROR: Could not open webcam.")
        print("Go to System Settings > Privacy & Security > Camera")
        print("and allow your terminal app access.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    frame_time = 1.0 / TARGET_FPS

    try:
        while True:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                continue

            h, w = stdscr.getmaxyx()
            render_h = h - 2  # leave room for status bar at bottom
            render_w = w

            if render_h < 1 or render_w < 1:
                continue

            frame = cv2.flip(frame, 1)

            if bg_sub_enabled:
                fg_mask = bg_subtractor.apply(frame)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.GaussianBlur(fg_mask, (7, 7), 0)
                frame = cv2.bitwise_and(frame, frame, mask=fg_mask)

            mm = MIRROR_MODES[mirror_index]
            if mm:
                frame = apply_mirror(frame, mm)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if brightness_offset != 0 or contrast_factor != 1.0:
                gf = gray.astype(np.float32)
                gf = contrast_factor * (gf - 128) + 128 + brightness_offset
                gray = np.clip(gf, 0, 255).astype(np.uint8)

            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            if edge_blend > 0:
                edges = cv2.Canny(gray, 50, 150)
                gray = cv2.addWeighted(gray, 1.0, edges, edge_blend, 0)

            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            gray_f = gray.astype(np.float32)
            if prev_gray is not None and prev_gray.shape == gray_f.shape:
                gray_f = 0.65 * gray_f + 0.35 * prev_gray
            prev_gray = gray_f
            gray = gray_f.astype(np.uint8)

            current_mode = MODES[mode_index]

            # Prepare shared color data
            color_pairs = None
            if color_enabled and tint_index == 0 and current_mode not in ("matrix", "pixel", "emoji", "thermal", "halfblock", "hd"):
                fs = cv2.resize(frame, (render_w, render_h * 2))
                fs = fs[::2, :]
                frgb = cv2.cvtColor(fs, cv2.COLOR_BGR2RGB)
                color_pairs = rgb_to_color_pair_vectorized(frgb[:,:,0], frgb[:,:,1], frgb[:,:,2])

            try:
                if current_mode == "braille":
                    g = (255 - gray) if inverted else gray
                    rows = render_braille(g, render_w, render_h)
                    for y in range(render_h):
                        render_row_with_color(stdscr, y, rows[y], color_enabled, color_pairs, tint_index, render_w)

                elif current_mode == "outline":
                    resized = cv2.resize(gray, (render_w, render_h * 2))[::2, :]
                    ei = cv2.Canny(resized, 80, 160)
                    if inverted: ei = 255 - ei
                    for y in range(render_h):
                        row = "".join("#" if ei[y, x] > 0 else " " for x in range(render_w))
                        render_row_with_color(stdscr, y, row, color_enabled, color_pairs, tint_index, render_w)

                elif current_mode == "dither":
                    resized = cv2.resize(gray, (render_w, render_h * 2))[::2, :]
                    d = ordered_dither(resized)
                    if inverted: d = ~d
                    for y in range(render_h):
                        row = "".join("@" if d[y, x] else " " for x in range(render_w))
                        render_row_with_color(stdscr, y, row, color_enabled, color_pairs, tint_index, render_w)

                elif current_mode == "matrix":
                    resized = cv2.resize(gray, (render_w, render_h * 2))[::2, :]
                    norm = resized / 255.0
                    if matrix_grid is None or matrix_grid.shape != (render_h, render_w):
                        matrix_grid = np.array([[random.choice(MATRIX_CHARS) for _ in range(render_w)] for _ in range(render_h)])
                    mask = np.random.random((render_h, render_w)) < 0.15
                    for my, mx in zip(*np.where(mask)):
                        matrix_grid[my, mx] = random.choice(MATRIX_CHARS)
                    for y in range(render_h):
                        for x in range(render_w):
                            b = norm[y, x]
                            if b < 0.15:
                                stdscr.addstr(y, x, " ")
                            elif b > 0.8:
                                stdscr.addstr(y, x, matrix_grid[y, x], curses.color_pair(242) | curses.A_BOLD)
                            else:
                                stdscr.addstr(y, x, matrix_grid[y, x], curses.color_pair(241))

                elif current_mode == "emoji":
                    ew = render_w // 2
                    resized = cv2.resize(gray, (ew, render_h * 2))[::2, :]
                    norm = resized / 255.0
                    if inverted: norm = 1.0 - norm
                    ne = len(EMOJI_CHARS)
                    idx = (norm * (ne - 1)).astype(int)
                    for y in range(render_h):
                        stdscr.addstr(y, 0, "".join(EMOJI_CHARS[idx[y, x]] for x in range(ew)))

                elif current_mode == "stipple":
                    resized = cv2.resize(gray, (render_w, render_h * 2))[::2, :]
                    d = ordered_dither(resized, gamma=2.2)
                    if inverted: d = ~d
                    for y in range(render_h):
                        row = "".join("." if d[y, x] else " " for x in range(render_w))
                        render_row_with_color(stdscr, y, row, color_enabled, color_pairs, tint_index, render_w)

                elif current_mode == "pixel":
                    resized = cv2.resize(gray, (render_w, render_h * 2))[::2, :]
                    if inverted: resized = 255 - resized
                    if color_enabled:
                        fs = cv2.resize(frame, (render_w, render_h * 2))[::2, :]
                        frgb = cv2.cvtColor(fs, cv2.COLOR_BGR2RGB)
                        pp = rgb_to_color_pair_vectorized(frgb[:,:,0], frgb[:,:,1], frgb[:,:,2])
                        for y in range(render_h):
                            for x in range(render_w):
                                stdscr.addstr(y, x, "█", curses.color_pair(int(pp[y, x])))
                    else:
                        for y in range(render_h):
                            for x in range(render_w):
                                stdscr.addstr(y, x, " ", curses.color_pair(brightness_to_grayscale_pair(resized[y, x])))

                elif current_mode == "thermal":
                    resized = cv2.resize(gray, (render_w, render_h * 2))[::2, :]
                    if inverted: resized = 255 - resized
                    chars = np.array(list(SHORT_CHARS_DARK))
                    nc = len(chars)
                    idx = (resized / 255.0 * (nc - 1)).astype(int)
                    cg = chars[idx]
                    for y in range(render_h):
                        for x in range(render_w):
                            stdscr.addstr(y, x, cg[y][x], curses.color_pair(brightness_to_thermal_pair(resized[y, x])) | curses.A_BOLD)

                elif current_mode == "halfblock":
                    # ▀ with fg color = top pixel, uses 1 color per cell
                    resized = cv2.resize(gray, (render_w, render_h * 2))
                    if inverted: resized = 255 - resized
                    if color_enabled:
                        fs = cv2.resize(frame, (render_w, render_h * 2))
                        frgb = cv2.cvtColor(fs, cv2.COLOR_BGR2RGB)
                        pp = rgb_to_color_pair_vectorized(frgb[:,:,0], frgb[:,:,1], frgb[:,:,2])
                        for y in range(render_h):
                            for x in range(render_w):
                                stdscr.addstr(y, x, "▀", curses.color_pair(int(pp[y * 2, x])))
                    else:
                        for y in range(render_h):
                            for x in range(render_w):
                                avg = (int(resized[y*2, x]) + int(resized[y*2+1, x])) // 2
                                stdscr.addstr(y, x, "▀", curses.color_pair(brightness_to_grayscale_pair(avg)))

                elif current_mode == "mirror":
                    # Renders as ascii — mirror effect is applied to source frame
                    chars = ascii_chars
                    nc = num_chars
                    resized = cv2.resize(gray, (render_w, render_h * 2))[::2, :]
                    norm = resized / 255.0
                    if inverted: norm = 1.0 - norm
                    idx = (norm * (nc - 1)).astype(int)
                    cg = chars[idx]
                    for y in range(render_h):
                        render_row_with_color(stdscr, y, "".join(cg[y]), color_enabled, color_pairs, tint_index, render_w)

                elif current_mode == "hd":
                    # HD mode: ▀ with 24-bit true color via ANSI escapes
                    # Bypasses curses color pair limit for maximum quality
                    # Each cell = 2 vertical pixels (fg=top, bg=bottom)
                    fs = cv2.resize(frame, (render_w, render_h * 2))
                    frgb = cv2.cvtColor(fs, cv2.COLOR_BGR2RGB)
                    if inverted:
                        frgb = 255 - frgb

                    buf = []
                    for y in range(render_h):
                        buf.append(f"\033[{y + 1};1H")  # move cursor
                        for x in range(render_w):
                            rt, gt, bt = int(frgb[y*2, x, 0]), int(frgb[y*2, x, 1]), int(frgb[y*2, x, 2])
                            rb, gb, bb = int(frgb[y*2+1, x, 0]), int(frgb[y*2+1, x, 1]), int(frgb[y*2+1, x, 2])
                            buf.append(f"\033[38;2;{rt};{gt};{bt}m\033[48;2;{rb};{gb};{bb}m▀")
                        buf.append("\033[0m")
                    sys.stdout.write("".join(buf))
                    sys.stdout.flush()

                else:
                    # ASCII or Block
                    if current_mode == "block":
                        chars = np.array(list(BLOCK_CHARS))
                        nc = len(chars)
                    else:
                        chars = ascii_chars
                        nc = num_chars
                    resized = cv2.resize(gray, (render_w, render_h * 2))[::2, :]
                    norm = resized / 255.0
                    if inverted: norm = 1.0 - norm
                    idx = (norm * (nc - 1)).astype(int)
                    cg = chars[idx]
                    for y in range(render_h):
                        render_row_with_color(stdscr, y, "".join(cg[y]), color_enabled, color_pairs, tint_index, render_w)

                stdscr.refresh()
            except curses.error:
                pass

            # ── Status bar (always visible, bottom row) ──
            elapsed = time.time() - t_start
            fps_update_timer += elapsed
            if fps_update_timer >= 0.5:
                fps_display = 1.0 / elapsed if elapsed > 0 else 0
                fps_update_timer = 0.0

            # Build always-visible status
            parts = []
            parts.append(f"mode:{current_mode}")
            parts.append(f"color:{'on' if color_enabled else 'off'}")
            if color_enabled:
                parts.append(f"tint:{COLOR_TINTS[tint_index]}")
            parts.append(f"inv:{'on' if inverted else 'off'}")
            parts.append(f"edge:{edge_blend:.1f}")
            parts.append(f"bgsub:{'on' if bg_sub_enabled else 'off'}")
            parts.append(f"mirror:{MIRROR_LABELS[mirror_index]}")
            parts.append(f"brt:{brightness_offset:+d}")
            parts.append(f"con:{contrast_factor:.1f}")
            if use_adaptive:
                parts.append("adaptive")
            parts.append(f"bg:{'dark' if dark_bg else 'light'}")

            status = " | ".join(parts)
            fps_str = f" {fps_display:.0f}fps "
            # Pad status to fill the row
            bar_row = h - 1
            try:
                bar = f" {status} "
                # Truncate if too long
                avail = w - len(fps_str)
                if len(bar) > avail:
                    bar = bar[:avail]
                padding = w - len(bar) - len(fps_str)
                full_bar = bar + " " * max(0, padding) + fps_str
                stdscr.addstr(bar_row, 0, full_bar[:w-1], curses.A_REVERSE)
                stdscr.refresh()
            except curses.error:
                pass

            # ── Input ──
            key = stdscr.getch()
            if key == ord("q"):
                break
            elif key == ord("i"):
                inverted = not inverted
            elif key == ord("c") and color_available:
                color_enabled = not color_enabled
            elif key == ord("t") and color_available:
                tint_index = (tint_index + 1) % len(COLOR_TINTS)
                if not color_enabled:
                    color_enabled = True
            elif key == ord("e"):
                if edge_blend == 0: edge_blend = 0.3
                elif edge_blend < 0.5: edge_blend = 0.6
                elif edge_blend < 0.9: edge_blend = 1.0
                else: edge_blend = 0.0
            elif key == ord("a"):
                use_adaptive = not use_adaptive
                if use_adaptive:
                    ascii_chars = measure_char_density()
                else:
                    ascii_chars = np.array(list(SHORT_CHARS_DARK if dark_bg else SHORT_CHARS_LIGHT))
                num_chars = len(ascii_chars)
            elif key == ord("m") or key == curses.KEY_RIGHT:
                mode_index = (mode_index + 1) % len(MODES)
                stdscr.erase()
            elif key == curses.KEY_LEFT:
                mode_index = (mode_index - 1) % len(MODES)
                stdscr.erase()
            elif key == ord("b"):
                bg_sub_enabled = not bg_sub_enabled
                if bg_sub_enabled:
                    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=50, detectShadows=False)
            elif key == ord("k"):
                mirror_index = (mirror_index + 1) % len(MIRROR_MODES)
            elif key == ord("["):
                brightness_offset = max(-100, brightness_offset - 10)
            elif key == ord("]"):
                brightness_offset = min(100, brightness_offset + 10)
            elif key == ord("{"):
                contrast_factor = max(0.5, round(contrast_factor - 0.1, 1))
            elif key == ord("}"):
                contrast_factor = min(2.0, round(contrast_factor + 0.1, 1))
            elif key == ord("0"):
                brightness_offset = 0
                contrast_factor = 1.0

            remaining = frame_time - (time.time() - t_start)
            if remaining > 0:
                time.sleep(remaining)

    finally:
        cap.release()


if __name__ == "__main__":
    print("ASCII Webcam Renderer")
    print("Controls:")
    print("  q=quit  i=invert  c=color  t=tint  e=edge  a=adaptive")
    print("  m=mode  b=background sub  k=mirror/kaleidoscope")
    print("  [/]=brightness  {/}=contrast  0=reset")
    print(f"Modes: {', '.join(MODES)}")
    print("Starting...")
    curses.wrapper(main)
