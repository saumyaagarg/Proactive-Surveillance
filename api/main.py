# api/main.py

import os
import shutil
import uuid
import subprocess
import json

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------
app = FastAPI(title="Visionary AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Directory Setup
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PIPELINE_PATH = os.path.join(BASE_DIR, "run_pipeline.py")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# Templates & Static Files
# ---------------------------------------------------
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# ---------------------------------------------------
# Homepage
# ---------------------------------------------------
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------
# Video Analysis API
# ---------------------------------------------------
@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)):

    file_id = str(uuid.uuid4())
    input_path = os.path.join(DATA_DIR, f"{file_id}.mp4")

    try:
        # 1️⃣ Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # 2️⃣ Run pipeline safely
        subprocess.run(
            ["python", PIPELINE_PATH, input_path],
            check=True,
            cwd=BASE_DIR
        )

        # 3️⃣ Load final report
        report_path = os.path.join(OUTPUT_DIR, "final_report.json")

        if not os.path.exists(report_path):
            return JSONResponse(
                status_code=500,
                content={"error": "Pipeline did not generate final_report.json"}
            )

        with open(report_path, "r") as f:
            report = json.load(f)

        # 4️⃣ Return FULL report directly
        return report

    except subprocess.CalledProcessError as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Pipeline execution failed: {str(e)}"}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected server error: {str(e)}"}
        )
