import os
import shutil
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from detector import Detector, box_center
from team_classifier import TeamClassifier
from possession import PossessionTracker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

for d in (INPUTS_DIR, OUTPUTS_DIR):
    os.makedirs(d, exist_ok=True)

app = FastAPI(title="Football Possession Analyzer")

TEAM_COLOR = {0: (255, 100, 0), 1: (0, 100, 255)}


def process_video(input_path, output_path):
    detector = Detector(MODEL_PATH)
    classifier = TeamClassifier()
    possession = PossessionTracker()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        players, ball_box = detector.track(frame)
        classifier.classify(frame, players)
        current_label = possession.update(players, ball_box, classifier)

        for tid, (x1, y1, x2, y2) in players:
            color = TEAM_COLOR.get(classifier.team_of(tid), (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {tid}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if ball_box is not None:
            bx, by = box_center(ball_box).astype(int)
            cv2.circle(frame, (bx, by), 6, (0, 255, 0), -1)

        cv2.putText(frame, f"Possession: {current_label}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()

    team_a_pct, team_b_pct = possession.percentages()
    return {
        "team_a_possession": team_a_pct,
        "team_b_possession": team_b_pct,
        "players_detected": classifier.players_detected(),
        "current_possession": possession.current_label,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    input_path = os.path.join(INPUTS_DIR, file.filename)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output_filename = f"analyzed_{file.filename}"
    output_path = os.path.join(OUTPUTS_DIR, output_filename)

    try:
        stats = process_video(input_path, output_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})

    return {"status": "completed", "video": output_filename, **stats}


@app.get("/health")
def health():
    return {"status": "ok"}