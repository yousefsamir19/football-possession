import cv2
import numpy as np
from collections import deque, Counter
from ultralytics import YOLO
from sklearn.cluster import KMeans


class FootballAnalyzer:
    """Detects players/ball, tracks players, splits teams by jersey color,
    and estimates ball possession."""

    def __init__(self, model_path="models/best12.pt"):
        self.model = YOLO(model_path)
        self.names = self.model.names  # {class_id: class_name}
        self.player_cls_ids = [i for i, n in self.names.items() if "player" in n.lower()]
        self.ball_cls_ids = [i for i, n in self.names.items() if "ball" in n.lower()]
        # fallback if custom model uses different naming (e.g. class 0 = player, 1 = ball)
        if not self.player_cls_ids:
            self.player_cls_ids = [0]
        if not self.ball_cls_ids:
            self.ball_cls_ids = [1]

        self.warmup_samples_needed = 40   # samples collected before freezing team colors
        self.smoothing_window = 15        # frames used for temporal smoothing

    # ---------- helpers ----------

    def _jersey_color(self, frame, box):
        x1, y1, x2, y2 = box
        h = y2 - y1
        # upper-body crop (jersey area), avoids shorts/grass
        crop = frame[y1:y1 + max(1, h // 2), x1:x2]
        if crop.size == 0:
            return None
        mean_color = crop.reshape(-1, 3).mean(axis=0)  # BGR
        return mean_color

    def _box_center(self, box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def _fit_kmeans(self, samples):
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        km.fit(np.array(samples))
        return km

    # ---------- main entry point ----------

    def analyze_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        track_color_samples = {}   # track_id -> list of color samples
        track_team = {}            # track_id -> 0 or 1
        kmeans = None
        color_pool = []            # flat pool of samples for warmup fitting

        possession_history = deque(maxlen=self.smoothing_window)
        team_frame_counts = {0: 0, 1: 0}
        current_possession_label = "Unknown"
        team_label = {0: "Team A", 1: "Team B"}
        team_color = {0: (255, 100, 0), 1: (0, 100, 255)}

        results_stream = self.model.track(
            source=input_path,
            tracker="bytetrack.yaml",
            persist=True,
            stream=True,
            verbose=False,
        )

        for result in results_stream:
            frame = result.orig_img
            if frame is None:
                continue

            player_boxes = []   # (track_id, box)
            ball_box = None

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                cls = boxes.cls.cpu().numpy().astype(int)
                ids = boxes.id.cpu().numpy().astype(int)

                for box, c, tid in zip(xyxy, cls, ids):
                    if c in self.player_cls_ids:
                        player_boxes.append((int(tid), tuple(box)))
                    elif c in self.ball_cls_ids and ball_box is None:
                        ball_box = tuple(box)

            # handle case where ball has no tracker id (common for small/fast objects)
            if ball_box is None and boxes is not None and boxes.id is None:
                xyxy = boxes.xyxy.cpu().numpy().astype(int) if boxes is not None else []
                cls = boxes.cls.cpu().numpy().astype(int) if boxes is not None else []
                for box, c in zip(xyxy, cls):
                    if c in self.ball_cls_ids:
                        ball_box = tuple(box)
                        break

            # ---- team assignment ----
            for tid, box in player_boxes:
                if tid in track_team:
                    continue
                color = self._jersey_color(frame, box)
                if color is None:
                    continue
                track_color_samples.setdefault(tid, []).append(color)

                if kmeans is None:
                    color_pool.append(color)
                    if len(color_pool) >= self.warmup_samples_needed:
                        kmeans = self._fit_kmeans(color_pool)
                        # assign team to every track id seen so far, using its average color
                        for t_id, samples in track_color_samples.items():
                            avg = np.mean(samples, axis=0)
                            track_team[t_id] = int(kmeans.predict([avg])[0])
                else:
                    track_team[tid] = int(kmeans.predict([color])[0])

            # ---- possession ----
            possession_team = None
            if ball_box is not None and player_boxes:
                ball_center = self._box_center(ball_box)
                nearest_tid, nearest_dist = None, float("inf")
                for tid, box in player_boxes:
                    dist = np.linalg.norm(self._box_center(box) - ball_center)
                    if dist < nearest_dist:
                        nearest_dist, nearest_tid = dist, tid
                if nearest_tid is not None and nearest_tid in track_team:
                    possession_team = track_team[nearest_tid]

            if possession_team is not None:
                possession_history.append(possession_team)

            if possession_history:
                smoothed = Counter(possession_history).most_common(1)[0][0]
                current_possession_label = team_label[smoothed]
                team_frame_counts[smoothed] += 1

            # ---- drawing ----
            for tid, box in player_boxes:
                x1, y1, x2, y2 = box
                team = track_team.get(tid)
                color = team_color.get(team, (200, 200, 200))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {tid}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if ball_box is not None:
                bx, by = self._box_center(ball_box).astype(int)
                cv2.circle(frame, (bx, by), 6, (0, 255, 255), -1)

            cv2.putText(frame, f"Possession: {current_possession_label}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            writer.write(frame)

        cap.release()
        writer.release()

        total = team_frame_counts[0] + team_frame_counts[1]
        if total > 0:
            team_a_pct = round(team_frame_counts[0] / total * 100)
            team_b_pct = 100 - team_a_pct
        else:
            team_a_pct, team_b_pct = 0, 0

        return {
            "team_a_possession": team_a_pct,
            "team_b_possession": team_b_pct,
            "players_detected": len(track_team) if track_team else len(track_color_samples),
            "current_possession": current_possession_label,
        }
