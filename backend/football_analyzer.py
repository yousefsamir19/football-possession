import cv2
from ultralytics import YOLO
from pathlib import Path


# =========================
# Paths
# =========================

MODEL_PATH = Path(__file__).parent / "models" / "best.pt"

model = YOLO(str(MODEL_PATH))


# =========================
# Track video
# =========================

def track_video(input_path, output_path):

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    # Get video information
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    print(f"Video: {width}x{height}")
    print(f"FPS: {fps}")

    # Make sure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # OpenCV writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create output video")

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # YOLO + ByteTrack
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.4,
            verbose=False
        )

        # Draw boxes + tracking IDs
        annotated_frame = results[0].plot()

        writer.write(annotated_frame)

        frame_count += 1

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    writer.release()

    print(f"Finished. Frames: {frame_count}")
    print(f"Saved to: {output_path}")

    return output_path