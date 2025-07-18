# Mapping-a-Pattern-onto-a-Curved-Surface


## ✨ Features

- 🎯 Automatic flag detection using HSV color segmentation
- 🌀 Perspective-aware warping that matches original orientation
- 💧 Adjustable transparency control (`0.0` to `1.0`)
- 🖱️ Interactive corner selection if auto-detection fails
- 🖼️ Background-preserving overlay
- 📁 Batch processing support for multiple images

---

# 🇺🇸 Flag Replacement Using OpenCV

This project blends a new flag onto a red/white cloth using computer vision techniques in OpenCV.

## Features
- Automatic flag detection (red and white color ranges)
- Corner detection and perspective transformation
- Realistic blending to simulate printed fabric

## 📸 Input
- `flag1.png`: Original background
- `amerFlag.jpg`: New flag to overlay

## 🧪 Output
- `final_result.jpg`: Blended flag image

## 🔧 Setup

```bash
pip install opencv-python numpy
