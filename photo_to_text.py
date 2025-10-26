#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Dependencies: pip install pillow numpy

from pathlib import Path
from typing import Optional
from PIL import Image, ImageOps, features
import io, lzma, argparse, math
import numpy as np

# ========= Global parameters =========
MAX_CHARS_HARD = 20000        # Hard cap for output text length
TARGET_PSNR0   = 32.0         # Initial PSNR threshold in dB
DEFAULT_INPUT  = "example.jpg"  # Safer default for GitHub (relative path)

# ========= Base91 (ASCII-only) implementation =========
_B91_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "!#$%&()*+,./:;<=>?@[]^_`{|}~\""
)  # length 91
_B91_D = {c: i for i, c in enumerate(_B91_ALPHABET)}

def b91_encode(data: bytes) -> str:
    v = 0
    b = 0
    out = []
    for octet in data:
        v |= octet << b
        b += 8
        if b > 13:
            x = v & 8191
            if x > 88:
                v >>= 13
                b -= 13
            else:
                x = v & 16383
                v >>= 14
                b -= 14
            out.append(_B91_ALPHABET[x % 91])
            out.append(_B91_ALPHABET[x // 91])
    if b:
        out.append(_B91_ALPHABET[v % 91])
        if b > 7 or v > 90:
            out.append(_B91_ALPHABET[v // 91])
    return "".join(out)

# ========= Utilities =========
def resize_keep_aspect(img: Image.Image, long_side: int) -> Image.Image:
    """Resize while preserving aspect ratio so that max(w, h) == long_side (or smaller)."""
    w, h = img.size
    s = min(1.0, long_side / max(w, h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return img if (nw, nh) == (w, h) else img.resize((nw, nh), Image.LANCZOS)

def psnr(ref: Image.Image, test: Image.Image) -> float:
    """Compute PSNR between images (after converting to RGB and aligning size)."""
    r = ref.convert("RGB")
    t = test.convert("RGB").resize(r.size, Image.BILINEAR)
    a = np.asarray(r, dtype=np.float32)
    b = np.asarray(t, dtype=np.float32)
    mse = ((a - b) ** 2).mean()
    if mse <= 1e-10:
        return 100.0
    return 10.0 * math.log10((255.0 ** 2) / mse)

def enc_webp(img: Image.Image, lossless: bool, q: int) -> bytes:
    """Encode to WEBP (lossless if requested)."""
    bio = io.BytesIO()
    if lossless:
        img.save(bio, format="WEBP", lossless=True, quality=100, method=6)
    else:
        img.save(bio, format="WEBP", quality=int(q), method=6)
    return bio.getvalue()

def enc_jpeg(img: Image.Image, q: int) -> bytes:
    """Encode to JPEG with standard settings."""
    bio = io.BytesIO()
    img.save(
        bio,
        format="JPEG",
        quality=int(q),
        optimize=True,
        progressive=True,
        subsampling="4:2:0",
    )
    return bio.getvalue()

def try_variant(
    baseline: Image.Image,
    img_s: Image.Image,
    codec: str,
    q: Optional[int],
    max_chars: int,
    need_psnr: float,
):
    """Try one codec/quality variant; return (text, None) if it fits constraints."""
    try:
        if codec == "WEBP_LOSSLESS":
            data = enc_webp(img_s, True, 100)
        elif codec == "WEBP_Q":
            data = enc_webp(img_s, False, int(q))
        elif codec == "JPEG_Q":
            data = enc_jpeg(img_s, int(q))
        else:
            return (None, None)
    except Exception:
        return (None, None)

    packed = lzma.compress(data, preset=9)
    text = b91_encode(packed)
    if len(text) > max_chars:
        return (None, None)

    # Validate quality by decoding from the already-packed bytes (no extra cost).
    dec_img = Image.open(io.BytesIO(lzma.decompress(packed)))
    if psnr(baseline, dec_img) < need_psnr:
        return (None, None)
    return (text, None)

def search_best(
    input_path: Path,
    max_chars: int = MAX_CHARS_HARD,
    target_psnr: float = TARGET_PSNR0,
    allow_gray: bool = True,
) -> str:
    """Search size/codec settings to meet length and PSNR constraints."""
    base = ImageOps.exif_transpose(Image.open(input_path)).convert("RGB")
    has_webp = features.check("webp")

    long_sides = [224, 192, 176, 160, 144, 128, 112, 96]  # progressively smaller
    webp_qs    = [85, 80, 75, 70, 65, 60, 55, 50]
    jpeg_qs    = [85, 80, 75, 70, 65, 60, 55, 50]

    # Start strict and gradually relax PSNR.
    for need_psnr in [target_psnr, 31.0, 30.0, 29.0, 28.0]:
        for mode in (["RGB", "L"] if allow_gray else ["RGB"]):
            for ls in long_sides:
                img_s = resize_keep_aspect(base, ls).convert(mode)
                baseline = img_s.convert("RGB")

                if has_webp:
                    ok = try_variant(baseline, img_s, "WEBP_LOSSLESS", None, max_chars, need_psnr)
                    if ok[0]:
                        return ok[0]
                    for q in webp_qs:
                        ok = try_variant(baseline, img_s, "WEBP_Q", q, max_chars, need_psnr)
                        if ok[0]:
                            return ok[0]
                for q in jpeg_qs:
                    ok = try_variant(baseline, img_s, "JPEG_Q", q, max_chars, need_psnr)
                    if ok[0]:
                        return ok[0]

    raise RuntimeError(
        "Cannot compress within 20,000 characters. Lower PSNR or resolution and try again."
    )

def main():
    parser = argparse.ArgumentParser(
        description="Image -> ASCII single-line text (<=20,000 chars). Output is a .txt file."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Input image path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--psnr",
        type=float,
        default=TARGET_PSNR0,
        help=f"PSNR threshold in dB (default {TARGET_PSNR0})",
    )
    parser.add_argument(
        "--no-gray",
        action="store_true",
        help="Disable grayscale candidates (may produce longer text).",
    )
    args = parser.parse_args()

    text = search_best(
        Path(args.input),
        max_chars=MAX_CHARS_HARD,
        target_psnr=args.psnr,
        allow_gray=not args.no_gray,
    )

    inp = Path(args.input)
    out_txt = inp.with_name(inp.stem + "_chat_ascii.txt")
    out_txt.write_text(text, encoding="ascii")

    print(f"[OK] Text length: {len(text)} / {MAX_CHARS_HARD}  →  {out_txt}")
    print("[OK] Completed under length/quality constraints. Only .txt is generated (no image).")

if __name__ == "__main__":
    main()
