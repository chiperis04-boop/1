"""Football analytics bundle for the v2 studio.

One call (`analyze`) runs the three "deep football math" passes on a tracked
clip and returns them together, plus the resolved hero track id:

  * team assignment + club colours   (vision/teams.TeamClassifier.assign)
  * jersey number recognition        (vision/jerseys.JerseyReader.read)
  * ball possession runs / share      (vision/possession.analyze_possession)

Hero resolution order:
  1. jersey match — parse the number from the Director's hero description and
     find the track wearing it (most reliable);
  2. team-aware nearest-to-ball (vision/teams.pick_key_player) when teams known;
  3. plain geometric pick already on the CropPlan.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..utils.io import get_logger
from .jerseys import JerseyReader, JerseyResult, number_from_description
from .possession import PossessionResult, analyze_possession
from .teams import TeamAssignment, TeamClassifier, pick_key_player

log = get_logger()


@dataclass
class Analytics:
    teams: TeamAssignment
    jerseys: JerseyResult
    possession: PossessionResult
    hero_id: int | None = None
    hero_number: int | None = None
    hero_source: str = "geometric"     # jersey | team_nearest | geometric

    def color_for_track(self, tid, default=(0, 220, 255)):
        return self.teams.color_for_track(tid, default)

    def possession_share_pct(self) -> dict[int, int]:
        return {t: round(v * 100) for t, v in self.possession.share.items()}


def analyze(clip_path: str, track, calib=None, cfg: dict | None = None,
            manifest=None, geometric_hero=None) -> Analytics:
    cfg = cfg or {}
    teams = TeamClassifier(cfg).assign(clip_path, track)
    jerseys = JerseyReader(cfg).read(clip_path, track)
    possession = analyze_possession(track, calib=calib, cfg=cfg,
                                    team_of=teams.team_of)

    hero_id, hero_number, source = _resolve_hero(
        manifest, jerseys, teams, track, geometric_hero)

    log.info(f"[analytics] hero track={hero_id} (#{hero_number}, via {source})")
    return Analytics(teams=teams, jerseys=jerseys, possession=possession,
                     hero_id=hero_id, hero_number=hero_number, hero_source=source)


def _resolve_hero(manifest, jerseys: JerseyResult, teams: TeamAssignment,
                  track, geometric_hero):
    desc = getattr(manifest, "main_hero_description", "") if manifest else ""
    number = number_from_description(desc)

    # 1) jersey match
    if number is not None:
        tid = jerseys.track_for_number(number)
        if tid is not None:
            return tid, number, "jersey"

    # 2) team-aware possession pick (only meaningful with team labels)
    if teams.team_of:
        tid = pick_key_player(track, teams.team_of)
        if tid is not None:
            return tid, _num_for(jerseys, tid), "team_nearest"

    # 3) plain geometric fallback (already computed by the Cameraman)
    fallback = geometric_hero if geometric_hero is not None \
        else getattr(track, "hero_id", None) or getattr(track, "key_track_id", None)
    return fallback, _num_for(jerseys, fallback), "geometric"


def _num_for(jerseys: JerseyResult, tid):
    return jerseys.number_of.get(tid) if tid is not None else None
