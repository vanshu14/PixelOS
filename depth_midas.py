# depth_midas.py
# Lightweight MiDaS v2 depth helper using torch.hub
import torch
import cv2
import numpy as np

class MiDaSDepth:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load model from pytorch hub (MiDaS v2.1)
        self.model_type = "DPT_Hybrid"  # good tradeoff speed/quality
        model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        model.to(self.device).eval()
        self.model = model
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.transforms.dpt_transform if "DPT" in self.model_type else self.transforms.midas_transform

    def predict_depth(self, frame):
        """Return depth map (float32) same HxW as input, units relative (not metric)."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch.unsqueeze(0))
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy()
        # normalize to meters-like range for display (not true meters)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        # invert so closer = smaller number? keep closer = smaller -> we keep as-is
        return depth.astype(np.float32)
