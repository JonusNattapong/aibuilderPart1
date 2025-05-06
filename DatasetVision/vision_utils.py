"""
Utility functions for vision dataset generation.
"""
import os
import json
import torch
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
from torchvision import transforms
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from typing import List, Dict, Any

class VisionTaskManager:
    """Manages vision tasks and their processing."""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def process_batch(self, 
                     model: torch.nn.Module,
                     images: List[Image.Image],
                     config: Dict[str, Any]) -> List[Dict]:
        """Process a batch of images for the specified task."""
        
        if self.task_name == "image_classification":
            return self._process_classification(model, images, config)
        elif self.task_name == "object_detection":
            return self._process_detection(model, images, config)
        elif self.task_name == "image_segmentation":
            return self._process_segmentation(model, images, config)
        elif self.task_name == "depth_estimation":
            return self._process_depth(model, images, config)
        elif self.task_name == "keypoint_detection":
            return self._process_keypoints(model, images, config)
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")

    def _process_classification(self, model, images, config):
        """Process images for classification."""
        results = []
        for image in images:
            # Prepare image
            img_tensor = self.transform(image).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs.logits, dim=1)
                
                # Get top K predictions
                top_k = min(config.get('top_k', 5), probs.shape[1])
                values, indices = torch.topk(probs, top_k)
                
                predictions = [
                    {
                        "label": model.config.id2label[idx.item()],
                        "confidence": val.item()
                    }
                    for val, idx in zip(values[0], indices[0])
                ]
                
            results.append({
                "task": "classification",
                "predictions": predictions
            })
        
        return results

    def _process_detection(self, model, images, config):
        """Process images for object detection."""
        results = []
        for image in images:
            # Prepare image
            img_tensor = self.transform(image).unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(img_tensor)
                
                boxes = outputs.pred_boxes[0].cpu().numpy()
                scores = outputs.scores[0].cpu().numpy()
                labels = outputs.labels[0].cpu().numpy()
                
                # Filter by confidence threshold
                conf_threshold = config.get('confidence_threshold', 0.5)
                mask = scores > conf_threshold
                
                detections = [
                    {
                        "bbox": box.tolist(),
                        "label": model.config.id2label[label],
                        "confidence": score
                    }
                    for box, label, score in zip(boxes[mask], labels[mask], scores[mask])
                ]
                
            results.append({
                "task": "detection",
                "detections": detections
            })
        
        return results

    def _process_segmentation(self, model, images, config):
        """Process images for segmentation."""
        results = []
        for image in images:
            # Prepare image
            img_tensor = self.transform(image).unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(img_tensor)
                masks = outputs.pred_masks.squeeze().cpu().numpy()
                
                # Convert masks to RLE format
                segmentation = []
                for i, mask in enumerate(masks):
                    rle = self._mask_to_rle(mask > config.get('mask_threshold', 0.5))
                    segmentation.append({
                        "label": model.config.id2label[i],
                        "rle": rle
                    })
                
            results.append({
                "task": "segmentation",
                "segmentation": segmentation
            })
        
        return results

    def _process_depth(self, model, images, config):
        """Process images for depth estimation."""
        results = []
        for image in images:
            # Prepare image
            img_tensor = self.transform(image).unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(img_tensor)
                depth_map = outputs.pred_depth.squeeze().cpu().numpy()
                
                # Convert depth map to more efficient format
                depth_data = {
                    "min_depth": float(depth_map.min()),
                    "max_depth": float(depth_map.max()),
                    "mean_depth": float(depth_map.mean()),
                    "depth_map": self._compress_depth_map(depth_map)
                }
                
            results.append({
                "task": "depth_estimation",
                "depth": depth_data
            })
        
        return results

    def _process_keypoints(self, model, images, config):
        """Process images for keypoint detection."""
        results = []
        for image in images:
            # Prepare image
            img_tensor = self.transform(image).unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(img_tensor)
                keypoints = outputs.pred_keypoints[0].cpu().numpy()
                scores = outputs.pred_keypoint_scores[0].cpu().numpy()
                
                # Filter by confidence threshold
                conf_threshold = config.get('keypoint_threshold', 0.3)
                mask = scores > conf_threshold
                
                points = [
                    {
                        "coordinate": point.tolist(),
                        "confidence": float(score),
                        "label": f"keypoint_{i}"
                    }
                    for i, (point, score) in enumerate(zip(keypoints[mask], scores[mask]))
                ]
                
            results.append({
                "task": "keypoint_detection",
                "keypoints": points
            })
        
        return results

    def _mask_to_rle(self, mask):
        """Convert binary mask to run-length encoding."""
        pixels = mask.flatten()
        runs = []
        run = 0
        prev = False
        for pixel in pixels:
            if pixel != prev:
                runs.append(run)
                run = 1
                prev = pixel
            else:
                run += 1
        runs.append(run)
        return runs

    def _compress_depth_map(self, depth_map, compression_factor=4):
        """Compress depth map by downsampling."""
        h, w = depth_map.shape
        new_h, new_w = h // compression_factor, w // compression_factor
        compressed = np.mean(depth_map.reshape(new_h, compression_factor, 
                                             new_w, compression_factor), axis=(1,3))
        return compressed.tolist()

def load_model(task: str, model_name: str) -> torch.nn.Module:
    """Load the specified model for a task."""
    try:
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_name} for task {task}: {str(e)}")

def process_image(image: Image.Image, task: str) -> Image.Image:
    """Pre-process image according to task requirements."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if needed
    if task == "depth_estimation":
        image = image.resize((384, 384))  # Standard size for depth estimation
    else:
        image = image.resize((224, 224))  # Standard size for other tasks
    
    return image

def save_dataset(data: List[Dict],
                filename: str,
                format_type: str,
                output_dir: str) -> str:
    """Save dataset in specified format."""
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{base_filename}.{format_type.lower()}")
    
    try:
        if format_type.upper() == "CSV":
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        elif format_type.upper() == "JSON":
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format_type.upper() == "JSONL":
            with open(output_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error saving dataset: {str(e)}")

def setup_device() -> torch.device:
    """Set up compute device (CPU/GPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
