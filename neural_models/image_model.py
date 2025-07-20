import torch
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import base64
import io
from typing import Dict, List, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ImageClassifier:
    """
    Neural image classifier using pre-trained vision transformers for pattern recognition.
    Converts raw images into classified labels with confidence scores.
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        """
        Initialize the image classifier

        Args:
            model_name: HuggingFace model name for image classification
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            # Define image transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            logger.info(f"ImageClassifier initialized with model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize ImageClassifier: {str(e)}")
            raise e

    async def predict(self, image_data: str) -> Dict[str, Any]:
        """
        Predict classification for input image

        Args:
            image_data: Base64 encoded image or image path

        Returns:
            Dictionary containing prediction results
        """
        try:
            # Load and preprocess image
            image = self._load_image(image_data)

            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get top predictions
            top_predictions = self._get_top_predictions(predictions)

            # Map to domain-specific categories
            domain_label = self._map_to_domain(top_predictions[0]['label'])

            # Extract visual features
            visual_features = self._extract_visual_features(image)

            return {
                "image_processed": True,
                "label": domain_label,
                "confidence": float(top_predictions[0]['score']),
                "top_predictions": top_predictions,
                "visual_features": visual_features,
                "model_used": self.model_name
            }

        except Exception as e:
            logger.error(f"Image prediction failed: {str(e)}")
            raise e

    def _load_image(self, image_data: str) -> Image.Image:
        """
        Load image from base64 string or file path

        Args:
            image_data: Base64 encoded image or file path

        Returns:
            PIL Image object
        """
        try:
            # Try to decode as base64 first
            if image_data.startswith('data:image'):
                # Remove data:image/jpeg;base64, prefix
                image_data = image_data.split(',')[1]

            try:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            except:
                # If base64 decoding fails, try loading as file path
                image = Image.open(image_data)

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return image

        except Exception as e:
            logger.error(f"Failed to load image: {str(e)}")
            raise e

    def _get_top_predictions(self, predictions: torch.Tensor, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get top-k predictions from model output

        Args:
            predictions: Model prediction tensor
            top_k: Number of top predictions to return

        Returns:
            List of top predictions with labels and scores
        """
        # Get top-k indices and scores
        top_scores, top_indices = torch.topk(predictions[0], top_k)

        results = []
        for score, idx in zip(top_scores, top_indices):
            # Map index to label (simplified - in practice you'd use model.config.id2label)
            label = f"CLASS_{idx.item()}"
            results.append({
                "label": label,
                "score": float(score),
                "index": int(idx)
            })

        return results

    def _map_to_domain(self, raw_label: str) -> str:
        """
        Map generic model labels to domain-specific categories

        Args:
            raw_label: Original model prediction label

        Returns:
            Domain-specific label
        """
        # Example domain mapping for visual recognition
        # In a real application, this would be more sophisticated
        domain_mappings = {
            "CLASS_0": "MOLECULAR_STRUCTURE",
            "CLASS_1": "CHEMICAL_COMPOUND",
            "CLASS_2": "BIOLOGICAL_ENTITY",
            "CLASS_3": "MEDICAL_IMAGE",
            "CLASS_4": "SAFETY_HAZARD",
        }

        return domain_mappings.get(raw_label, "UNKNOWN_VISUAL")

    def _extract_visual_features(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract basic visual features from the image

        Args:
            image: PIL Image object

        Returns:
            Dictionary of visual features
        """
        # Convert to numpy array for analysis
        img_array = np.array(image)

        # Basic feature extraction
        features = {
            "dimensions": {
                "width": image.width,
                "height": image.height,
                "channels": len(img_array.shape)
            },
            "color_analysis": {
                "mean_brightness": float(np.mean(img_array)),
                "color_variance": float(np.var(img_array)),
                "dominant_colors": self._get_dominant_colors(img_array)
            },
            "complexity": {
                "edge_density": self._calculate_edge_density(img_array),
                "texture_measure": self._calculate_texture(img_array)
            }
        }

        return features

    def _get_dominant_colors(self, img_array: np.ndarray, n_colors: int = 3) -> List[List[int]]:
        """Extract dominant colors from image"""
        # Reshape image to be a list of pixels
        pixels = img_array.reshape(-1, img_array.shape[-1])

        # Simple approach: find most common colors
        # In practice, you'd use clustering (k-means)
        from collections import Counter

        # Convert to tuples for counting
        pixel_tuples = [tuple(pixel) for pixel in pixels[::100]]  # Sample every 100th pixel
        color_counts = Counter(pixel_tuples)

        # Get top colors
        dominant = [list(color) for color, _ in color_counts.most_common(n_colors)]
        return dominant

    def _calculate_edge_density(self, img_array: np.ndarray) -> float:
        """Calculate edge density as a measure of image complexity"""
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Simple edge detection (Sobel-like)
        grad_x = np.abs(np.diff(gray, axis=1))
        grad_y = np.abs(np.diff(gray, axis=0))

        edge_density = (np.mean(grad_x) + np.mean(grad_y)) / 2
        return float(edge_density)

    def _calculate_texture(self, img_array: np.ndarray) -> float:
        """Calculate texture measure"""
        # Simple texture measure based on local variance
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Calculate local variance in 3x3 windows
        texture = 0.0
        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                window = gray[i-1:i+2, j-1:j+2]
                texture += np.var(window)

        return float(texture / ((gray.shape[0] - 2) * (gray.shape[1] - 2)))

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "model_type": "image_classification",
            "input_size": "224x224"
        }
