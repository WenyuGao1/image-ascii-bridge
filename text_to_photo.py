#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Dependencies: pip install pillow

from pathlib import Path
from PIL import Image
import io, lzma, argparse, sys

# ===== Base91 (decode only) =====
_B91_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "!#$%&()*+,./:;<=>?@[]^_`{|}~\""
)
_B91_D = {c: i for i, c in enumerate(_B91_ALPHABET)}

def b91_decode(s: str) -> bytes:
    """Decode Base91 ASCII string to raw bytes (ignores non-alphabet chars)."""
    v = -1
    b = 0
    n = 0
    out = bytearray()
    for ch in s:
        c = _B91_D.get(ch)
        if c is None:
            continue
        if v < 0:
            v = c
        else:
            v += c * 91
            n |= v << b
            if (v & 8191) > 88:
                b += 13
            else:
                b += 14
            while True:
                out.append(n & 255)
                n >>= 8
                b -= 8
                if b <= 7:
                    break
            v = -1
    if v + 1:
        out.append((n | (v << b)) & 255)
    return bytes(out)

def main():
    parser = argparse.ArgumentParser(
        description="ASCII single-line text -> image (decoded from Base91 + LZMA)."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="example_chat_ascii.txt",
        help="Input *_chat_ascii.txt path (default: example_chat_ascii.txt)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output image path. Default: <input_stem_without_suffix>_recovered.png",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"[ERR] Input not found: {in_path}")

    # Read ASCII text
    try:
        txt = in_path.read_text(encoding="ascii").strip()
    except UnicodeDecodeError:
        sys.exit("[ERR] Input file must be ASCII-only text.")

    # Decode → decompress → open as image
    try:
        data = lzma.decompress(b91_decode(txt))
    except Exception as e:
        sys.exit(f"[ERR] Failed to decode/decompress: {e}")

    try:
        img = Image.open(io.BytesIO(data))
    except Exception as e:
        sys.exit(f"[ERR] Decoded bytes are not a valid image: {e}")

    # Decide output path
    if args.output:
        out_path = Path(args.output)
        if out_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            # Default to PNG if no/unsupported suffix
            out_path = out_path.with_suffix(".png")
    else:
        # strip the trailing "_chat_ascii" if present
        stem = in_path.stem.replace("_chat_ascii", "")
        out_path = in_path.with_name(stem + "_recovered.png")

    # Save as PNG by default (lossless, safe)
    try:
        img.save(out_path, format=out_path.suffix.replace(".", "").upper() or "PNG", optimize=True)
    except Exception:
        # Fallback to PNG if the requested format fails
        out_path = out_path.with_suffix(".png")
        img.save(out_path, format="PNG", optimize=True)

    print(f"[OK] Recovered: {out_path}  size: {img.size}  mode: {img.mode}")

if __name__ == "__main__":
    main()
