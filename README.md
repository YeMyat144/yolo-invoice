# YOLO Thai Invoices Detection System

A computer vision system for detecting and extracting key information from Thai invoices and receipts using YOLO (You Only Look Once) object detection.

## 📋 Project Overview

This project implements an automated invoice processing system that can detect and extract various fields from Thai invoices and receipts, including:

- **Seller Information**: Name (Thai/English), VAT number, address, phone, branch
- **Document Details**: Date, product descriptions, quantities, prices
- **Financial Information**: Subtotal, VAT amount, discounts, total amounts
- **Buyer Information**: Name (English)
- **Tax Information**: VAT percentages, withholding tax details

## 🏗️ Project Structure

```
yolo-invoice/
├── data/                          # Dataset and configuration files
│   ├── dataset.yaml              # YOLO dataset configuration
│   ├── invoicedata/              # Raw invoice data
│   └── YOLODataset/              # Processed YOLO dataset
├── models/                        # Trained YOLO models
│   ├── best.pt                   # Best performing model
│   ├── last.pt                   # Latest model checkpoint
│   └── yolo11s.pt                # YOLO11s pretrained model
├── results/                       # Training results and predictions
│   ├── testing/                  # Test predictions
│   ├── F1_curve.png             # Performance metrics
│   └── Invoice Partial Receipt YOLO Training Report.pdf
├── scripts/                       # Python scripts
│   ├── convert.py                # Dataset conversion utility
│   ├── predict.py                # Inference script
│   ├── requirement.txt           # Python dependencies
│   └── yolov11_sec1.ipynb        # Training notebook
└── ENV/                          # Python virtual environment
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for training)
- Windows 10/11

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd yolo-invoice
   ```

2. **Set up virtual environment**
   ```bash
   # Create virtual environment
   python -m venv ENV
   
   # Activate virtual environment
   # Windows:
   ENV\Scripts\activate
   # Linux/Mac:
   source ENV/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r scripts/requirement.txt
   ```

### Usage

#### 1. Dataset Preparation

If you have new invoice data to process:

```bash
python scripts/convert.py
```

This script reorganizes the dataset into the proper YOLO format.

#### 2. Training

Use the Jupyter notebook for training:

```bash
jupyter notebook scripts/yolov11_sec1.ipynb
```

#### 3. Inference

To run predictions on new images:

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("models/best.pt")

# Run inference on an image
results = model("path/to/your/invoice.jpg")

# Display and save results
for result in results:
    result.show()
    result.save(filename="prediction_result.jpg")
```

Or use the provided script:

```bash
python scripts/predict.py
```

## 📊 Model Performance

The system is trained to detect 22 different invoice fields:

| Class ID | Field Name | Description |
|----------|------------|-------------|
| 0 | seller_name_eng | Seller name (English) |
| 1 | seller_vat_number | Seller VAT number |
| 2 | Doc_date | Document date |
| 3 | product_description | Product description |
| 4 | quantity_item | Item quantity |
| 5 | price | Item price |
| 6 | sub_total | Subtotal before tax |
| 7 | vat_amount | VAT amount |
| 8 | total_due_amount | Total amount due |
| 9 | seller_address_thai | Seller address (Thai) |
| 10 | seller_name_thai | Seller name (Thai) |
| 11 | seller_phone | Seller phone number |
| 12 | seller_branch | Seller branch |
| 13 | discount_amount | Discount amount |
| 14 | seller_address_eng | Seller address (English) |
| 15 | Table | Table structure |
| 16 | vat_% | VAT percentage |
| 17 | amount_after_discount | Amount after discount |
| 18 | buyer_name_eng | Buyer name (English) |
| 19 | vat_% | VAT percentage (duplicate) |
| 20 | WHT_% | Withholding tax percentage |
| 21 | WHT_amount | Withholding tax amount |

## 🔧 Configuration

### Dataset Configuration

The `data/dataset.yaml` file contains:
- Training, validation, and test data paths
- Class names and their corresponding IDs
- Dataset-specific configurations

### Model Configuration

- **Model Architecture**: YOLO11s
- **Input Resolution**: Configurable (typically 640x640)
- **Classes**: 22 invoice field classes
- **Training**: Transfer learning from pretrained weights

## 📈 Results

Training results and performance metrics are available in the `results/` directory:
- **F1_curve.png**: F1-score performance curve
- **Training Report**: Detailed training analysis and metrics
- **Test Predictions**: Sample inference results
