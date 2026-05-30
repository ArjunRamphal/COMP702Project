# South African Bank Note Recognition System

**Course:** COMP702 - Image Processing and Computer Vision

**Authors:** Arjun Ramphal, Darian Robert, Lokadi Naicker, Keolin Naicker

## Overview

This repository contains a robust Computer Vision pipeline for identifying South African bank notes (R10, R20, R50, R100, R200). The system is designed to be invariant to scale, rotation, and facial side (obverse/reverse), and is specifically optimized to perform accurately against high-frequency textured backgrounds.

To fulfill the comparative requirements of the Honours project, this system implements a **Sequential Optimization Study**.

1. **Phase 1: Preprocessing Evaluation:** Tests contrast enhancement and noise reduction techniques (CLAHE, Bilateral, Gaussian, Equalization) to find the optimal structural clarifier.
2. **Phase 2: Segmentation & Geometric Normalization:** Tests boundary isolation methods (Perspective Warping, Exact Masking, Padded Bounding Boxes, Adaptive Thresholding). 
3. **Phase 3: Feature Extraction & Template Matching:** Evaluates the final isolated Regions of Interest (ROI) across four distinct classification architectures:
* **SIFT** (Floating-point geometry matched via FLANN KD-Tree)
* **ORB** (Binary geometry matched via BF Hamming + RANSAC)
* **AKAZE** (Non-linear scale space matched via BF Hamming + RANSAC)
* **Color Histograms** (3D HSV color mass matched via Histogram Intersection)


## Installation & Setup

1. Ensure Python 3.10+ is installed on your system.
2. Clone this repository to your local machine.
3. Install the required dependencies:

```bash
pip install -r requirements.txt

```

### Dependencies (`requirements.txt`)

```text
opencv-python>=4.5.0
numpy
scikit-learn
```

## Dataset Configuration

The system requires a strict directory structure to automatically label and process the images. Ensure your repository is structured as follows before execution:

```text
COMP702_BankNote_Recognition/
│
├── README.md                 
├── requirements.txt          
├── main.py                   
│
├── reference_templates/      # Directory for baseline white-background template notes
│   ├── R10/
│   ├── R20/
│   ├── R50/
│   ├── R100/
│   └── R200/
│
└── test_images/              # Directory for evaluation notes on textured backgrounds
    ├── R10/
    ├── R20/
    ├── R50/
    ├── R100/
    └── R200/

```

## Execution

Run the system pipeline via the terminal from the root directory:

```bash
python main.py

```

### Expected Output

1. **Phase 1 & 2 Diagnostics:** The system will output real-time Scikit-Learn accuracy metrics for every preprocessing and segmentation technique, automatically locking in the mathematical winner for the next stage.
2. **Phase 3 Predictions:** The system will print individual prediction results for every image in the test set across the four distinct extraction architectures (SIFT, ORB, AKAZE, Hist).
3. **Snapshot Generation:** A `report_snapshots/` directory will be automatically generated, saving visual evidence of the preprocessing and segmentation results for use in the final academic report.
4. **Final Pipeline Configuration Report:** Upon completion, a comprehensive summary block will output the absolute best combination of Preprocessing, Segmentation, and Feature Extraction, alongside the final achieved accuracy.