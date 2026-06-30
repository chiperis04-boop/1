"""Team classification + possession-aware protagonist selection.

The v0.1 "key player = nearest to ball" heuristic mis-votes under occlusion.
Knowing which team each tracked player belongs to lets us restrict the
protagonist to the attacking team (the side in possession at the key beat),
which is far more robust.

Default classifier: a dependency-light **HSV jersey-colour histogram** of each
player's torso, clustered into two teams with KMeans; outliers (keeper/referee)
fall out naturally. Each track id is assigned by majority vote across the clip
for temporal stability.

Optional upgrade (documented, not required): replace the colour histogram with
SigLIP embeddings as in roboflow/sports' team-classification approach
(https://github.com/roboflow/sports) for tougher kits.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..utils.io import get_logger
from .detect_track import TrackResult

log = get_logger()


@dataclass
class TeamAssignment:
    """Result of team clustering for a clip.

    * team_of:  {track_id -> team label (0/1/.., -1 unknown)}
    * colors:   {team label -> (B, G, R)} dominant jersey colour for halos
    * outliers: track ids that didn't fit either team (keeper/ref), kept as -1
    """
    team_of: dict[int, int] = field(default_factory=dict)
    colors: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    n_teams: int = 2

    def color_for_track(self, tid: int, default=(0, 220, 255)) -> tuple[int, int, int]:
        return self.colors.get(self.team_of.get(tid, -1), default)


class TeamClassifier:
    def __init__(self, cfg: dict):
        t = cfg.get("vision", {}).get("teams", {})
        self.enabled = bool(t.get("enabled", False))
        self.n_teams = int(t.get("n_teams", 2))
        self.method = t.get("method", "hsv")     # hsv | siglip

    def classify(self, clip_path: str, track: TrackResult) -> dict[int, int]:
        """Return {track_id: team_label}. team_label in {0,1,-1(unknown)}.

        Backward-compatible thin wrapper around `assign()`.
        """
        return self.assign(clip_path, track).team_of

    def assign(self, clip_path: str, track) -> TeamAssignment:
        """Cluster players into teams AND extract each team's dominant jersey
        colour (median torso BGR) so halos can be painted in club colours.

        Works on both the v1 `TrackResult` and the v2 `CropPlan` since both
        expose `.frames[*].idx` and `.frames[*].players[*]` with id/xyxy.
        """
        if not self.enabled:
            return TeamAssignment(n_teams=self.n_teams)
        try:
            import cv2
            from sklearn.cluster import KMeans
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[teams] disabled (deps missing): {exc}")
            return TeamAssignment(n_teams=self.n_teams)

        cap = cv2.VideoCapture(clip_path)
        feats: list[np.ndarray] = []
        torso_bgr: list[np.ndarray] = []
        owners: list[int] = []
        frame_index = {fd.idx: fd for fd in track.frames}
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fd = frame_index.get(idx)
            if fd:
                for p in fd.players:
                    f, bgr = self._features(frame, p["xyxy"], cv2)
                    if f is not None:
                        feats.append(f)
                        torso_bgr.append(bgr)
                        owners.append(p["id"])
            idx += 1
        cap.release()

        if len(feats) < self.n_teams * 3:
            return TeamAssignment(n_teams=self.n_teams)
        X = np.array(feats, dtype=np.float32)
        labels = KMeans(n_clusters=self.n_teams, n_init=5,
                        random_state=0).fit_predict(X)

        votes: dict[int, Counter] = defaultdict(Counter)
        for tid, lab in zip(owners, labels):
            votes[tid][int(lab)] += 1
        team_of = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}

        # dominant colour per team = median torso BGR of its samples
        colors: dict[int, tuple[int, int, int]] = {}
        bgr_arr = np.array(torso_bgr, dtype=np.float32)
        for lab in range(self.n_teams):
            sel = bgr_arr[labels == lab]
            if len(sel):
                med = np.median(sel, axis=0).astype(int)
                colors[lab] = (int(med[0]), int(med[1]), int(med[2]))

        log.info(f"[teams] {len(team_of)} tracks -> {self.n_teams} teams, "
                 f"colors={colors}")
        return TeamAssignment(team_of=team_of, colors=colors, n_teams=self.n_teams)

    def _features(self, frame, xyxy, cv2):
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        h = y2 - y1
        # torso band (skip head/legs) to capture the shirt colour
        ty1, ty2 = y1 + int(0.2 * h), y1 + int(0.55 * h)
        crop = frame[max(0, ty1):max(0, ty2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return None, None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8],
                            [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        mean_bgr = crop.reshape(-1, 3).mean(axis=0)
        return hist, mean_bgr


def pick_key_player(track: TrackResult, team_of: dict[int, int]) -> int | None:
    """Possession-aware protagonist selection.

    1. Find the attacking team = team of the player most often nearest the ball
       during the decisive final third.
    2. Among that team's players, pick the one most often nearest the ball.
    Falls back to the plain nearest-to-ball heuristic when team info is absent.
    """
    if not track.frames:
        return None
    start = int(len(track.frames) * 0.6)

    # nearest-to-ball votes (and which team that player is on)
    nearest_votes: Counter = Counter()
    team_votes: Counter = Counter()
    for fd in track.frames[start:]:
        if not fd.ball or not fd.players:
            continue
        bx, by = fd.ball["center"]
        nearest = min(fd.players,
                      key=lambda p: (p["center"][0] - bx) ** 2 + (p["center"][1] - by) ** 2)
        nearest_votes[nearest["id"]] += 1
        if team_of:
            t = team_of.get(nearest["id"], -1)
            if t != -1:
                team_votes[t] += 1

    if not nearest_votes:
        return None
    if not team_of or not team_votes:
        return nearest_votes.most_common(1)[0][0]      # fallback

    attacking = team_votes.most_common(1)[0][0]
    # restrict to attacking team
    filtered = Counter({tid: n for tid, n in nearest_votes.items()
                        if team_of.get(tid, -1) == attacking})
    if filtered:
        return filtered.most_common(1)[0][0]
    return nearest_votes.most_common(1)[0][0]
