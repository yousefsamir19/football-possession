import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from football_analyzer import FootballAnalyzer

app = FastAPI(title="Football Possession Analyzer")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

analyzer = FootballAnalyzer(model_path=MODEL_PATH)


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    input_path = os.path.join(INPUTS_DIR, file.filename)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output_filename = f"analyzed_{file.filename}"
    output_path = os.path.join(OUTPUTS_DIR, output_filename)

    try:
        stats = analyzer.analyze_video(input_path, output_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})

    return {
        "status": "completed",
        "video": output_filename,
        "team_a_possession": stats["team_a_possession"],
        "team_b_possession": stats["team_b_possession"],
        "players_detected": stats["players_detected"],
        "current_possession": stats["current_possession"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
