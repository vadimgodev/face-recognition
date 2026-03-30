"""Anti-spoofing prediction using MiniFASNet models."""

from __future__ import annotations

import logging
import math
import os
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.antispoof.data_io import transform as trans
from src.antispoof.model_lib.MiniFASNet import (
    MiniFASNetV1,
    MiniFASNetV1SE,
    MiniFASNetV2,
    MiniFASNetV2SE,
)
from src.antispoof.utility import get_kernel, parse_model_name

logger = logging.getLogger(__name__)

MODEL_MAPPING = {
    "MiniFASNetV1": MiniFASNetV1,
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE,
}


class FaceDetector:
    """Face detector using InsightFace RetinaFace (fallback to OpenCV DNN)."""

    def __init__(self, model_path: str = "./models", confidence_threshold: float = 0.6):
        """
        Initialize face detector.

        Args:
            model_path: Path to model directory (unused, kept for compatibility)
            confidence_threshold: Detection confidence threshold
        """
        self.confidence_threshold = confidence_threshold
        self._lock = threading.Lock()  # Thread safety
        self._insightface_detector = None

        # Try to use InsightFace's built-in detector (more reliable)
        try:
            import warnings

            import insightface

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._insightface_detector = insightface.app.FaceAnalysis(
                    name="buffalo_s",  # Lightweight model for detection only
                    providers=["CPUExecutionProvider"],
                )
                self._insightface_detector.prepare(ctx_id=-1, det_size=(320, 320))

            logger.info("Using InsightFace built-in face detector for liveness")
            return
        except Exception as e:
            logger.warning(f"Could not load InsightFace detector: {e}, falling back to OpenCV DNN")

        # Fallback to OpenCV DNN (legacy)
        caffemodel = os.path.join(model_path, "Widerface-RetinaFace.caffemodel")
        deploy = os.path.join(model_path, "deploy.prototxt")

        if not os.path.exists(caffemodel) or not os.path.exists(deploy):
            raise FileNotFoundError(
                f"Detection model files not found in {model_path}. "
                "Please ensure Widerface-RetinaFace.caffemodel and deploy.prototxt exist."
            )

        # Validate model files exist and have non-zero size
        caffemodel_size = os.path.getsize(caffemodel)
        deploy_size = os.path.getsize(deploy)

        if caffemodel_size == 0:
            raise RuntimeError(f"Caffemodel file is empty: {caffemodel}")
        if deploy_size == 0:
            raise RuntimeError(f"Deploy prototxt file is empty: {deploy}")

        # Load model with comprehensive error handling
        try:
            self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)

            # Validate model loaded correctly by testing with a dummy input
            # This will fail if the model architecture doesn't match weights
            test_blob = cv2.dnn.blobFromImage(
                np.zeros((100, 100, 3), dtype=np.uint8), 1.0, mean=(104, 117, 123)
            )
            self.detector.setInput(test_blob, "data")
            _ = self.detector.forward("detection_out")

        except cv2.error as e:
            raise RuntimeError(
                f"Failed to load face detector models from {model_path}. "
                f"The caffemodel file ({caffemodel_size} bytes) may be corrupted, incomplete, "
                f"or incompatible with the prototxt architecture. "
                f"OpenCV error: {e}\n"
                f"Troubleshooting:\n"
                f"1. Re-download Widerface-RetinaFace.caffemodel (expected size: ~5-10MB)\n"
                f"2. Verify deploy.prototxt matches the model architecture\n"
                f"3. Check file permissions and disk space"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading face detector: {e}") from e

        self.confidence_threshold = confidence_threshold
        # Thread safety: OpenCV DNN has global state and is NOT thread-safe
        # This lock protects detector inference operations
        # Note: Callers should also use external locks (see silent_face_liveness.py)
        self._lock = threading.Lock()

    def get_bbox(self, img: np.ndarray) -> list[int]:
        """
        Detect face and return bounding box.

        Args:
            img: Input image (BGR format)

        Returns:
            Bounding box [x, y, width, height]

        Raises:
            ValueError: If no face detected
        """
        # Use InsightFace detector if available (preferred)
        if self._insightface_detector is not None:
            with self._lock:
                # Convert BGR to RGB for InsightFace
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = self._insightface_detector.get(rgb_img)

            if not faces:
                raise ValueError("No face detected in image")

            # Get the face with highest confidence
            face = max(faces, key=lambda f: f.det_score if hasattr(f, "det_score") else 1.0)

            # Extract bbox [x, y, width, height]
            bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            bbox = [x1, y1, x2 - x1, y2 - y1]
            return bbox

        # Fallback to OpenCV DNN (legacy)
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height

        # Resize for detection if image is large
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img_resized = cv2.resize(
                img,
                (
                    int(192 * math.sqrt(aspect_ratio)),
                    int(192 / math.sqrt(aspect_ratio)),
                ),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            img_resized = img

        # Detect faces - OpenCV DNN is not thread-safe, use lock
        with self._lock:
            blob = cv2.dnn.blobFromImage(img_resized, 1, mean=(104, 117, 123))
            self.detector.setInput(blob, "data")
            out = self.detector.forward("detection_out").squeeze()

        # Get highest confidence detection
        if len(out.shape) == 1:
            # No faces detected
            raise ValueError("No face detected in image")

        max_conf_index = np.argmax(out[:, 2])
        confidence = out[max_conf_index, 2]

        if confidence < self.confidence_threshold:
            raise ValueError(f"No face detected with sufficient confidence (max: {confidence:.2f})")

        # Extract bbox coordinates
        left = out[max_conf_index, 3] * width
        top = out[max_conf_index, 4] * height
        right = out[max_conf_index, 5] * width
        bottom = out[max_conf_index, 6] * height

        bbox = [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]
        return bbox


class AntiSpoofPredictor:
    """Anti-spoofing predictor using MiniFASNet models."""

    def __init__(
        self,
        device_id: int = -1,
        model_dir: str = "./models/anti_spoof",
        detector_path: str = "./models",
    ):
        """
        Initialize anti-spoofing predictor.

        Args:
            device_id: GPU device ID (-1 for CPU, 0+ for GPU)
            model_dir: Directory containing anti-spoofing models
            detector_path: Path to face detector models
        """
        self.device = (
            torch.device(f"cuda:{device_id}")
            if device_id >= 0 and torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model_dir = model_dir
        self.face_detector = FaceDetector(detector_path)
        self._loaded_models = {}  # Cache for loaded models

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load anti-spoofing model.

        Args:
            model_path: Path to model file

        Returns:
            Loaded PyTorch model
        """
        # Check cache
        if model_path in self._loaded_models:
            return self._loaded_models[model_path]

        # Parse model parameters from filename
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        kernel_size = get_kernel(h_input, w_input)

        # Create model
        model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)

        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = next(keys)

        # Handle DataParallel models
        if first_layer_name.find("module.") >= 0:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        model.eval()
        self._loaded_models[model_path] = model
        return model

    def predict_single(self, img: np.ndarray, model_path: str) -> tuple[float, float]:
        """
        Run prediction with a single model.

        Args:
            img: Preprocessed image (numpy array)
            model_path: Path to model file

        Returns:
            Tuple of (fake_score, real_score)
        """
        # Transform image to tensor
        test_transform = trans.Compose([trans.ToTensor()])
        img_tensor = test_transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Load model and predict
        model = self._load_model(model_path)

        with torch.no_grad():
            result = model.forward(img_tensor)
            result = F.softmax(result, dim=1).cpu().numpy()[0]

        # result[0] = fake, result[1] = real
        return float(result[0]), float(result[1])

    def predict(
        self, image_bytes: bytes, return_bbox: bool = False
    ) -> tuple[float, list[int] | None]:
        """
        Predict if image is real or fake using model ensemble.

        Args:
            image_bytes: Image data as bytes
            return_bbox: Whether to return detected face bounding box

        Returns:
            Tuple of (real_score, bbox) where:
            - real_score: 0-1 score (higher = more likely real)
            - bbox: Bounding box [x, y, width, height] (if return_bbox=True)

        Raises:
            ValueError: If no face detected or image invalid
        """
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image data")

        # Detect face
        bbox = self.face_detector.get_bbox(image)

        # Get list of models
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith(".pth")]

        if not model_files:
            raise RuntimeError(f"No anti-spoofing models found in {self.model_dir}")

        # Import crop utility
        from src.antispoof.crop_image import CropImage

        image_cropper = CropImage()

        # Aggregate predictions from all models
        total_real_score = 0.0
        num_models = 0

        for model_name in model_files:
            h_input, w_input, model_type, scale = parse_model_name(model_name)

            # Prepare crop parameters
            param = {
                "org_img": image,
                "bbox": bbox,
                "scale": scale if scale is not None else 1.0,
                "out_w": w_input,
                "out_h": h_input,
                "crop": scale is not None,
            }

            # Crop image
            img_cropped = image_cropper.crop(**param)

            # Predict
            model_path = os.path.join(self.model_dir, model_name)
            fake_score, real_score = self.predict_single(img_cropped, model_path)

            total_real_score += real_score
            num_models += 1

        # Average scores across models
        avg_real_score = total_real_score / num_models

        if return_bbox:
            return avg_real_score, bbox
        return avg_real_score, None
