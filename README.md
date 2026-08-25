# Rice Grain Counter

A computer vision system built with Python and OpenCV that detects and counts rice grains from images and live camera input.

The project started as a basic contour-based object counter and was progressively developed into a more robust rice grain detection system using image preprocessing, contour detection, watershed segmentation, synthetic dataset generation, and live Android camera streaming through DroidCam and OBS Virtual Camera.

---

## Features

- Rice grain detection using OpenCV
- Image preprocessing and binary segmentation
- Contour-based object detection
- Watershed segmentation for touching grains
- Automatic grain counting
- Bounding boxes around detected grains
- Synthetic rice image dataset generation
- Separate datasets for normal and touching grains
- Automated detection testing
- Live camera input
- Android phone camera support
- DroidCam + OBS Virtual Camera integration
- Command-line image counting
- Live grain counting interface
- Modular project architecture

---

## Project Architecture

The project is divided into several components:

```text
Android Phone
     │
     ▼
  DroidCam
     │
     ▼
OBS Virtual Camera
     │
     ▼
OpenCV
     │
     ├── Image Preprocessing
     │
     ├── Segmentation
     │
     ├── Contour Detection
     │
     ├── Watershed Segmentation
     │
     └── Grain Counting