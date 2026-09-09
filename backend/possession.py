import numpy as np
from collections import deque, Counter
from detector import box_center

TEAM_LABEL = {0: "Team A", 1: "Team B"}


class PossessionTracker:
    """Nearest player to ball -> possession -> temporal smoothing -> percentages."""

    def __init__(self, smoothing_window=15):
        self.history = deque(maxlen=smoothing_window)
        self.team_frame_counts = {0: 0, 1: 0}
        self.current_label = "Unknown"

    def update(self, players, ball_box, team_classifier):
        possession_team = None

        if ball_box is not None and players:
            ball_center = box_center(ball_box)
            nearest_tid, nearest_dist = None, float("inf")
            for tid, box in players:
                dist = np.linalg.norm(box_center(box) - ball_center)
                if dist < nearest_dist:
                    nearest_dist, nearest_tid = dist, tid

            if nearest_tid is not None:
                team = team_classifier.team_of(nearest_tid)
                if team is not None:
                    possession_team = team

        if possession_team is not None:
            self.history.append(possession_team)

        if self.history:
            smoothed = Counter(self.history).most_common(1)[0][0]
            self.current_label = TEAM_LABEL[smoothed]
            self.team_frame_counts[smoothed] += 1

        return self.current_label

    def percentages(self):
        total = self.team_frame_counts[0] + self.team_frame_counts[1]
        if total == 0:
            return 0, 0
        team_a_pct = round(self.team_frame_counts[0] / total * 100)
        return team_a_pct, 100 - team_a_pct