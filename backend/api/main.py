# api/main.py

import os
import shutil
import uuid
import json
import subprocess

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI(title="Visionary AI Backend")

# Enable CORS for Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Directories
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve output files
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


# ---------------------------------------------------
# API Endpoint
# ---------------------------------------------------
@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)):

    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        input_path = os.path.join(DATA_DIR, f"{file_id}.mp4")

        # Save uploaded video
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # ---------------------------------------------------
        # Run Pipeline
        # ---------------------------------------------------
        # Modify run_pipeline.py to accept video path as arg
        subprocess.run(
            ["python", "run_pipeline.py", input_path],
            check=True
        )

        # ---------------------------------------------------
        # Read Final Report
        # ---------------------------------------------------
        report_path = os.path.join(OUTPUT_DIR, "final_report.json")

        if not os.path.exists(report_path):
            return JSONResponse(
                status_code=500,
                content={"error": "Pipeline failed to generate report"}
            )

        with open(report_path, "r") as f:
            report = json.load(f)

        # Format response for frontend
        response = {
            "action": {
                "label": report["action_recognition"]["action"],
                "confidence": report["action_recognition"]["confidence"]
            },
            "anomaly": report["anomaly"],
            "caption": report["caption"],
            "outputs": {
                "gradcam_video": "http://localhost:8000/output/gradcam_video.mp4",
                "activation_video": "http://localhost:8000/output/activation_video.mp4"
            }
        }

        return response

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
