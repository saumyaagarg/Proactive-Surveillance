# captioning/caption_frames.py

import torch
import cv2
import json
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BlipProcessor,
    BlipForConditionalGeneration
)


class SceneCaptioner:

    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pegasus_available = False

        # ===============================
        # 1️⃣ GIT
        # ===============================
        print("Loading GIT-large captioning model...")
        self.git_processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
        self.git_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/git-large-coco"
        ).to(self.device)
        self.git_model.eval()
        print("✓ GIT loaded")

        # ===============================
        # 2️⃣ BLIP
        # ===============================
        print("Loading BLIP captioning model...")
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.blip_model.eval()
        print("✓ BLIP loaded")

        # ===============================
        # 3️⃣ PEGASUS SUMMARIZER (with fallback)
        # ===============================
        try:
            print("Loading PEGASUS summarizer...")
            self.sum_tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
            self.sum_model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/pegasus-xsum"
            ).to(self.device)
            self.sum_model.eval()
            self.pegasus_available = True
            print("✓ PEGASUS loaded")
        except Exception as e:
            print(f"⚠️ PEGASUS failed to load: {e}. Using BLIP summarization fallback.")
            self.pegasus_available = False
            self.sum_tokenizer = None
            self.sum_model = None

    # ...existing code...

    @torch.no_grad()
    def build_final_caption(
        self,
        frame_captions,
        action_label=None,
        action_confidence=None
    ):

        if not frame_captions:
            return "No visual description available."

        # Remove duplicates
        unique = list(dict.fromkeys(frame_captions))

        text_block = " ".join(unique)

        # Use PEGASUS if available, otherwise use direct captions
        if self.pegasus_available and self.sum_model is not None:
            try:
                inputs = self.sum_tokenizer(
                    text_block,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                summary_ids = self.sum_model.generate(
                    **inputs,
                    max_length=60,
                    num_beams=4,
                    early_stopping=True
                )

                summary = self.sum_tokenizer.decode(
                    summary_ids[0],
                    skip_special_tokens=True
                ).strip()

                # Fallback if summarizer collapses
                if len(summary.split()) < 5:
                    summary = " ".join(unique)
            except Exception as e:
                print(f"⚠️ PEGASUS summarization failed: {e}. Using frame captions directly.")
                summary = " ".join(unique)
        else:
            # Fallback: use frame captions directly
            summary = " ".join(unique)

        if action_label:
            summary += (
                f" The recognized activity is {action_label} "
                f"(confidence {action_confidence:.2f})."
            )

        return summary


def save_captions(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✓ Captions saved at: {path}")
