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

## Features
- Dynamic Image Upload: Accepts real-world leaf photographs (JPG/PNG) and evaluates them on demand.
- Multi-Model Pipeline: Integrates EfficientNet-B0, U-Net, and Grad-CAM in a unified architecture.
- Severity Staging: Categorizes severity into Mild, Moderate, and Severe based on infected pixel percentage.
- Benchmark Comparison: Presents transfer learning comparison data across EfficientNet-B0, MobileNetV3, and ResNet50.
- SQLite Prediction History: Preserves an audit log of past leaf predictions.

## Tech Stack
- PyTorch & Torchvision
- Timm (EfficientNet-B0)
- Segmentation Models PyTorch (U-Net with ResNet18 encoder)
- PyTorch Grad-CAM
- OpenCV (Headless)
- Streamlit
- SQLite3
- FastAPI / Uvicorn

## Project Structure
```text
cropvision-ai/
├── app.py                  # Streamlit web dashboard
├── engine.py               # Core inference pipeline and database functions
├── api.py                  # Optional FastAPI REST endpoint
├── cnn_benchmark.csv       # Architectural benchmark metrics
├── cropvision_effnet.pth   # Trained EfficientNet-B0 weights
├── cropvision_unet.pth     # Trained U-Net segmentation weights
├── requirements.txt        # Python package dependencies
├── Dockerfile              # Container deployment file
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/HarshwardhanSingh145/cropvision-ai.git
cd cropvision-ai
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

### Option A: Streamlit Web Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Option B: FastAPI Backend
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
Interactive API documentation will be available at `http://localhost:8000/docs`.

## Docker Deployment
```bash
docker build -t cropvision-ai .
docker run -p 8501:8501 cropvision-ai
```

