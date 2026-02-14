# explainability/activation_maps.py

import cv2
import torch
import numpy as np

from action_recognition.cnn3d_model import generate_model
from action_recognition.infer_action import load_clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ActivationMapExplainer:
    """
    Temporal activation-based explainability.
    Measures feature energy over time.
    """

    def __init__(self, model_path):
        self.model = generate_model(num_classes=101)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def generate(self, video_path):
        """
        Returns:
        importance: (T,) numpy array in [0,1]
        """

        clip = load_clip(video_path).to(DEVICE)  # (1, 3, T, H, W)

        # Get features
        _, feats = self.model(clip, return_features=True)  # (1, 256)

        # Feature energy → temporal importance
        T = clip.shape[2]
        importance = torch.norm(feats, p=2, dim=1)  # (1,)
        importance = importance.repeat(T)           # (T,)

        importance = importance.cpu().numpy()
        importance = importance / (importance.max() + 1e-8)

        return importance


def save_activation_video(video_path, importance, out_path):
    """
    Saves video with temporal activation bar overlay.
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

    # Match importance length to frames
    idxs = np.linspace(0, len(frames) - 1, len(importance)).astype(int)
    frames = [frames[i] for i in idxs]

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    bar_height = 20

    for frame, score in zip(frames, importance):
        frame = frame.copy()

        # Draw activation bar
        bar_width = int(score * width)
        cv2.rectangle(
            frame,
            (0, height - bar_height),
            (bar_width, height),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"Activation: {score:.2f}",
            (10, height - bar_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        writer.write(frame)

    writer.release()
    print(f"✅ Activation video saved at: {out_path}")
