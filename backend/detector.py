import numpy as np
from ultralytics import YOLO


def box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


class Detector:
    """YOLO + ByteTrack: detects players (with track IDs) and the ball."""

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        names = self.model.names
        self.player_cls_ids = [i for i, n in names.items() if "player" in n.lower()] or [0]
        self.ball_cls_ids = [i for i, n in names.items() if "ball" in n.lower()] or [1]

    def track(self, frame):
        """Returns (players, ball_box).
        players: list of (track_id, box) for each detected player.
        ball_box: (x1, y1, x2, y2) or None.
        """
        result = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", verbose=False
        )[0]

        players = []
        ball_box = None

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return players, ball_box

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        for i, (box, c) in enumerate(zip(xyxy, cls)):
            if c in self.player_cls_ids and ids is not None:
                players.append((int(ids[i]), tuple(box)))
            elif c in self.ball_cls_ids and ball_box is None:
                ball_box = tuple(box)

        return players, ball_box