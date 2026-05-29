# South African Bank Note Recognition System

**Course:** COMP702 - Image Processing and Computer Vision

**Authors:** Arjun Ramphal, Darian Robert, Lokadi Naicker, Keolin Naicker

## Overview

This repository contains a robust, dual-strategy Computer Vision pipeline for identifying South African bank notes (R10, R20, R50, R100, R200). The system is designed to be invariant to scale, rotation, and facial side (obverse/reverse), and is specifically optimized to perform accurately against high-frequency textured backgrounds.

To fulfill the comparative requirements of the project, the system implements two decoupled architectural pipelines:

* **Strategy A (Structural):** Utilizes Perspective Warping, SIFT feature extraction, and USAC_MAGSAC Homography to match local geometric landmarks.
* **Strategy B (Statistical):** Utilizes Exact Contour Masking, V-Channel Illumination Normalization, 3D HSV Color Histograms, and Intersection metrics to match global color mass.

## Installation & Setup

1. Ensure Python 3.10+ is installed on your system.
2. Clone this repository to your local machine.
3. Install the required dependencies:
```bash
pip install -r requirements.txt

```



```

### Dependencies (`requirements.txt`)
```text
opencv-python==4.9.0.80
opencv-contrib-python==4.9.0.80
numpy==1.26.4
scikit-learn==1.4.1.post1

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

1. Place clear, un-occluded baseline images of the bank notes in the `reference_templates/` directory, organized by class folder. Ensure you have templates for both old and new notes, as well as front and back sides.
2. Place your evaluation images (notes on textured backgrounds with rotation/scale variations) into the `test_images/` directory, accurately organized by their ground-truth class folder.

## Execution

Run the system pipeline via the terminal from the root directory:

```bash
python main.py

```

### Expected Output

1. **Template Loading:** The system will output a real-time console log tracking the successful loading and feature extraction of your reference templates.
2. **Testing Phase:** The system will print the individual prediction results for every image in the test set, displaying the ground truth alongside Strategy A's prediction and Strategy B's prediction.
3. **Accuracy Report:** Upon completion, a comprehensive Scikit-Learn classification report will be printed to the console, detailing the Precision, Recall, F1-Score, and total Accuracy for both Strategy A and Strategy B.