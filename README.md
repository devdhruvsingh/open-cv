# 🌾 Rice Grain Counter

A computer vision-based Rice Grain Counter built with **Python and OpenCV** that detects and counts individual rice grains from images and live camera input.

The project started as a simple contour-based object detection system and was progressively improved with preprocessing, watershed-based separation, synthetic dataset generation, testing, and live camera support.

---

# 📌 Project Overview

The goal of this project is to automatically count rice grains using computer vision.

The system supports:

- Static image-based rice grain counting
- Synthetic rice image generation
- Normal and touching/overlapping rice grains
- Contour-based object detection
- Watershed-based separation of touching grains
- Live camera input
- Android phone camera through DroidCam
- OBS virtual camera support
- Bounding boxes around detected grains
- Detection visualization
- Automated testing
- Multiple rice varieties from the dataset

The project is currently based on traditional computer vision techniques rather than a deep learning model.

---

# 🧠 How It Works

The basic processing pipeline is:

Camera/Image
↓
Image Capture
↓
Preprocessing
↓
Grayscale Conversion
↓
Noise Reduction
↓
Thresholding
↓
Morphological Processing
↓
Contour Detection
↓
Object Separation
↓
Watershed Processing
↓
Bounding Boxes
↓
Rice Grain Count

For separated grains, contour detection is usually sufficient.

When multiple grains touch each other, the system uses additional segmentation techniques such as watershed processing to attempt to separate them.

---

# 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Git
- GitHub
- DroidCam
- OBS Studio
- AVFoundation on macOS

---

# 📁 Project Structure

```text
open-cv/
│
├── .venv/
│
├── Rice_Image_Dataset/
│   ├── Arborio/
│   ├── Basmati/
│   ├── Ipsala/
│   ├── Jasmine/
│   └── Karacadag/
│
├── data/
│   └── generated/
│       ├── rice_5_1.png
│       ├── rice_10_1.png
│       ├── rice_20_1.png
│       ├── rice_30_1.png
│       ├── rice_50_1.png
│       ├── touching_5_1.png
│       ├── touching_10_1.png
│       ├── touching_20_1.png
│       ├── touching_30_1.png
│       └── touching_50_1.png
│
├── src/
│   ├── preprocessing.py
│   ├── counter.py
│   └── dataset_generator.py
│
├── main.py
├── count_image.py
├── test_counter.py
├── live_counter.py
├── obs_camera.py
├── camera_diagnostic.py
│
├── README.md
├── .gitignore
└── requirements.txt


---

# 2. `docs/SETUP.md`

This contains the **actual installation/setup process**.

```markdown
# Setup Guide

## Requirements

Recommended environment:

- macOS
- Python 3.10+
- OpenCV
- NumPy
- Git
- Virtual environment

## Clone the Repository

```bash
git clone https://github.com/devdhruvsingh/open-cv.git
cd open-cv

