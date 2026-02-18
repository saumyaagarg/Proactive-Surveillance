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
        # 3️⃣ PEGASUS SUMMARIZER
        # ===============================
        print("Loading PEGASUS summarizer...")
        self.sum_tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
        self.sum_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/pegasus-xsum"
        ).to(self.device)
        self.sum_model.eval()
        print("✓ PEGASUS loaded")


    # =========================================================
    # Caption Quality Filter
    # =========================================================
    def is_generic(self, caption: str):
        caption = caption.lower().strip()

        generic_patterns = [
            "all images are copyrighted",
            "copyright",
            "image",
            "photo",
            "picture",
            "stock photo",
            "getty",
            "shutterstock"
        ]

        if len(caption) < 6:
            return True

        if any(p in caption for p in generic_patterns):
            return True

        return False


    # =========================================================
    # Frame Captioning with Ensemble Selection
    # =========================================================
    @torch.no_grad()
    def caption_video_frames(self, frames):

        selected_captions = []

        for frame in frames:

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            # ---- GIT ----
            git_inputs = self.git_processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)

            git_ids = self.git_model.generate(
                **git_inputs,
                max_length=40
            )

            git_caption = self.git_processor.batch_decode(
                git_ids,
                skip_special_tokens=True
            )[0].strip()


            # ---- BLIP ----
            blip_inputs = self.blip_processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)

            blip_ids = self.blip_model.generate(
                **blip_inputs,
                max_length=40
            )

            blip_caption = self.blip_processor.decode(
                blip_ids[0],
                skip_special_tokens=True
            ).strip()


            # ---- Ensemble Decision ----
            if self.is_generic(git_caption) and not self.is_generic(blip_caption):
                final = blip_caption
            elif self.is_generic(blip_caption) and not self.is_generic(git_caption):
                final = git_caption
            else:
                # Choose longer (more descriptive) caption
                final = max(git_caption, blip_caption, key=len)

            selected_captions.append(final)

        return selected_captions


    # =========================================================
    # PEGASUS Summarization
    # =========================================================
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

        # PEGASUS input
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
