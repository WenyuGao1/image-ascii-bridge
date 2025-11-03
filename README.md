# Image–ASCII Bridge

Encode images into single-line ASCII text (Base91 + LZMA) within a hard length cap, and decode back to images.

## Features
- Base91 ASCII text (safe to paste into most text boxes)
- LZMA compression
- PSNR-based quality gate for encoding
- WEBP/JPEG candidates, auto search for size/quality balance

## Install
```bash
pip install -r requirements.txt
```

## Quick Start
```bash
# Encode → produces example_chat_ascii.txt
python photo_to_text.py example.jpg
```

## Decode
```bash
python text_to_photo.py example_chat_ascii.txt -o recovered.png
```

## Notes
```bash
Default input names are relative (e.g., example.jpg), no machine-specific paths.
You can adjust PSNR or disable grayscale candidates in the encoder:
python photo_to_text.py example.jpg --psnr 32 --no-gray
```

## Files
```bash
photo_to_text.py — Image → ASCII (Base91+LZMA)
text_to_photo.py — ASCII → Image


