import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """Jersey color -> KMeans(2) -> Team A / Team B."""

    def __init__(self, warmup_samples=40):
        self.warmup_samples = warmup_samples
        self.track_colors = {}   # track_id -> list of color samples
        self.track_team = {}     # track_id -> 0 or 1
        self.kmeans = None
        self.color_pool = []

    def _jersey_color(self, frame, box):
        x1, y1, x2, y2 = box
        h = y2 - y1
        crop = frame[y1:y1 + max(1, h // 2), x1:x2]
        if crop.size == 0:
            return None
        return crop.reshape(-1, 3).mean(axis=0)

    def classify(self, frame, players):
        """players: list of (track_id, box). Updates and returns track_team dict."""
        for tid, box in players:
            if tid in self.track_team:
                continue

            color = self._jersey_color(frame, box)
            if color is None:
                continue
            self.track_colors.setdefault(tid, []).append(color)

            if self.kmeans is None:
                self.color_pool.append(color)
                if len(self.color_pool) >= self.warmup_samples:
                    self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
                    self.kmeans.fit(np.array(self.color_pool))
                    for t_id, samples in self.track_colors.items():
                        avg = np.mean(samples, axis=0)
                        self.track_team[t_id] = int(self.kmeans.predict([avg])[0])
            else:
                self.track_team[tid] = int(self.kmeans.predict([color])[0])

        return self.track_team

    def team_of(self, track_id):
        return self.track_team.get(track_id)

    def players_detected(self):
        return len(self.track_team) if self.track_team else len(self.track_colors)