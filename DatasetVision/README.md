# Vision Dataset Generator

Interactive tool for generating computer vision datasets using pre-trained models with Streamlit interface.

## Features

- Multiple vision tasks supported:
  - Image Classification
  - Object Detection
  - Image Segmentation
  - Depth Estimation
  - Keypoint Detection

- Interactive interface:
  - Real-time image preview
  - Batch processing
  - Progress tracking
  - Results visualization

- Multiple pre-trained models supported
- Configurable task parameters
- Multiple export formats (JSON, CSV, JSONL)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DatasetVision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit interface:
```bash
streamlit run streamlit_app.py
```

2. Using the interface:
   - Select vision task
   - Choose model
   - Configure task parameters
   - Upload images
   - Process images
   - Review results
   - Export dataset

## Supported Tasks

### Image Classification
- Models: ResNet-50, ViT, DeiT
- Parameters:
  - Top K predictions

### Object Detection
- Models: DETR, Faster R-CNN
- Parameters:
  - Confidence threshold

### Image Segmentation
- Models: Mask2Former, SegFormer
- Parameters:
  - Mask threshold

### Depth Estimation
- Models: DPT, DINO
- Parameters:
  - Minimum depth

### Keypoint Detection
- Models: HRNet, MobileNetV3
- Parameters:
  - Keypoint confidence threshold

## Output Formats

### JSON
```json
{
  "task": "image_classification",
  "model": "microsoft/resnet-50",
  "config": {
    "top_k": 5
  },
  "result": {
    "predictions": [
      {
        "label": "cat",
        "confidence": 0.95
      }
    ]
  }
}
```

### CSV
Contains flattened version of JSON output for easy analysis.

### JSONL
Line-by-line JSON format for large datasets.

## Configuration

Edit `config.py` to modify:
- Model settings
- Task parameters
- Cache settings
- Output paths
- Visualization options

## Directory Structure

```
DatasetVision/
├── streamlit_app.py    # Streamlit interface
├── vision_utils.py     # Vision processing utilities
├── config.py          # Configuration settings
├── requirements.txt   # Dependencies
├── output/           # Generated datasets
├── .model_cache/     # Cached models
└── logs/            # Application logs
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU support)
- See requirements.txt for complete list

## GPU Support

Enable GPU usage by setting environment variable:
```bash
export USE_GPU=1
```

## Contributing

Feel free to submit issues and pull requests for:
- New vision tasks
- Additional models
- Interface improvements
- Bug fixes

## License

MIT License