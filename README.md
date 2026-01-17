# CropVision AI - Crop Leaf Disease Diagnosis and Severity Assessment

A computer vision pipeline combining Deep Learning classification, visual explainability, and semantic segmentation for plant pathology diagnosis.

## Project Overview
Plant diseases cause significant yield losses in agriculture if not identified early. Traditional visual inspection by agronomists is time-consuming and subjective.

CropVision AI automates the diagnosis workflow:
1. Quality Check: Validates leaf presence and image sharpness using OpenCV Laplacian variance and HSV color segmentation.
2. Classification: Identifies the disease using transfer learning with an EfficientNet-B0 backbone.
3. Visual Explainability: Produces attention heatmaps via Grad-CAM to highlight which regions influenced the classification.
4. Lesion Segmentation: Measures disease severity using a U-Net architecture to compute infected-to-healthy leaf pixel ratios.
5. Historical Audit Trail: Automatically logs diagnosis results, confidence levels, and severity stages into a SQLite database.

## Tech Stack
- PyTorch & Torchvision
- Timm (EfficientNet-B0)
- Segmentation Models PyTorch (U-Net with ResNet18 encoder)
- PyTorch Grad-CAM
- OpenCV (Headless)
- Streamlit
- SQLite3
- FastAPI / Uvicorn

## Setup and Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
