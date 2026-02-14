# explainability/gradcam_action.py

import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np

from action_recognition.cnn3d_model import generate_model
from action_recognition.infer_action import load_clip

# ---------------- CONFIG ----------------
IMG_SIZE = 112
FRAMES = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------


class ActionGradCAM:
    def __init__(self, model_path):
        self.model = generate_model(num_classes=101)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()

        self.activations = None
        self.gradients = None

        # 🔥 Last Conv3D layer in YOUR CNN3D
        target_layer = self.model.features[12]

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out          # (B, C, T, H, W)

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]    # (B, C, T, H, W)

    def generate(self, video_path):
        """
        Returns:
        cam_maps: (T, H, W) numpy array in [0,1]
        class_idx: predicted class
        """
        clip = load_clip(video_path).to(DEVICE)  # (1, 3, T, H, W)

        logits = self.model(clip)
        class_idx = logits.argmax(dim=1).item()

        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward()

        # ---- Grad-CAM computation ----
        # Global average pooling on gradients
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)  # (B, T, H, W)
        cam = F.relu(cam)

        # Normalize safely
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = cam[0].detach().cpu().numpy()  # (T, H, W)
        return cam, class_idx


def save_gradcam_video(video_path, cam_maps, out_path):
    """
    Saves Grad-CAM overlay video.
    """
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    # 🔥 Sample same number of frames as CAMs
    idxs = np.linspace(0, len(frames) - 1, len(cam_maps)).astype(int)
    frames = [frames[i] for i in idxs]

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    for frame, cam in zip(frames, cam_maps):
        # 1️⃣ Resize CAM to frame size
        cam_resized = cv2.resize(cam, (width, height))

        # 2️⃣ Convert to heatmap
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

        # 3️⃣ Ensure correct dtype
        frame = frame.astype(np.uint8)
        heatmap = heatmap.astype(np.uint8)

        # 4️⃣ Overlay
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        writer.write(overlay)

    writer.release()
    print(f"✅ Grad-CAM video saved at: {out_path}")
