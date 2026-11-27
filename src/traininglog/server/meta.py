"""Statistical meta-game analytics, Dual WPA, Ridge impact regression, synergy, and 4-lane engine."""

from __future__ import annotations

import json
import math
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .db import Database, sanitize_json_value

_WPA_SHRINK_K = 20.0
_SYNERGY_SHRINK_K = 30.0
_LANE_SHRINK_K = 15.0
_RIDGE_LAMBDA = 25.0
_PROB_EPS = 0.02


def _clamp_prob(p: float) -> float:
    return max(_PROB_EPS, min(1.0 - _PROB_EPS, float(p)))


def _logit(p: float) -> float:
    cp = _clamp_prob(p)
    return math.log(cp / (1.0 - cp))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-min(z, 60.0))
        return 1.0 / (1.0 + ez)
    ez = math.exp(max(z, -60.0))
    return ez / (1.0 + ez)


def _wilson_ci(wins: float, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = max(0.0, min(1.0, wins / n))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _safe_div(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    val = num / den
    return val if math.isfinite(val) else None


@dataclass(frozen=True)
class MetaFilter:
    project: str | None = None
    source: str | None = None
    manifest_hash: str | None = None
    mode: str = "1v1"
    exclude_bots: bool = True
    min_rating: float | None = None
    max_rating: float | None = None
    tier: str | None = None
    top_k_agents: int | None = None
    exclude_mirrors: bool = False
    min_games: int = 10
    min_pair_games: int | None = None
    since_ms: int | None = None
    until_ms: int | None = None
    faction: str | None = None
    card_type: str | None = None
    mana_cost_min: int | None = None
    mana_cost_max: int | None = None
    tag: str | None = None
    critic_perspective: str | None = None

    def canonical_key(self, prefix: str, project_id: str | None, extra: str = "") -> str:
        payload = asdict(self)
        payload["_prefix"] = prefix
        payload["_project_id"] = project_id or ""
        if extra:
            payload["_extra"] = extra
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class MetaAnalyticsEngine:
    """Computes card meta analytics over SQLite telemetry with watermarked caching."""

    def __init__(self, db: Database, max_mem_cache: int = 64):
        self.db = db
        self._max_mem_cache = max_mem_cache
        self._mem_cache: OrderedDict[str, tuple[int, dict[str, Any]]] = OrderedDict()

    def _get_cached(
        self, cache_key: str, project_id: str | None, watermark: int
    ) -> dict[str, Any] | None:
        item = self._mem_cache.get(cache_key)
        if item is not None:
            cached_wm, payload = item
            if cached_wm == watermark:
                self._mem_cache.move_to_end(cache_key)
                return payload
            del self._mem_cache[cache_key]

        db_hit = self.db.get_meta_cache(cache_key, watermark)
        if db_hit is not None:
            self._mem_cache[cache_key] = (watermark, db_hit)
            while len(self._mem_cache) > self._max_mem_cache:
                self._mem_cache.popitem(last=False)
            return db_hit
        return None

    def _put_cached(
        self, cache_key: str, project_id: str | None, watermark: int, payload: dict[str, Any]
    ) -> None:
        clean = sanitize_json_value(payload)
        self._mem_cache[cache_key] = (watermark, clean)
        self._mem_cache.move_to_end(cache_key)
        while len(self._mem_cache) > self._max_mem_cache:
            self._mem_cache.popitem(last=False)
        self.db.set_meta_cache(cache_key, project_id, watermark, clean)

    # ------------------------------------------------------------------
    # Filters & Catalog Loading
    # ------------------------------------------------------------------

    def get_filters(self, project: str | None = None) -> dict[str, Any]:
        project_id = self.db.resolve_project_id(project)
        with self.db._lock:
            proj_rows = self.db._conn.execute(
                """
                SELECT p.id, p.name, COUNT(g.id) AS meta_games
                FROM projects p
                LEFT JOIN meta_games g ON g.project_id = p.id
                GROUP BY p.id
                ORDER BY meta_games DESC, p.created_at DESC
                """
            ).fetchall()

            if not project_id:
                return {
                    "project": None,
                    "project_id": None,
                    "projects": [dict(r) for r in proj_rows],
                    "sources": [],
                    "manifests": [],
                    "modes": [],
                    "tiers": [],
                    "factions": [],
                    "card_types": [],
                    "critic_perspectives": [],
                    "rating_range": {"min": None, "max": None},
                    "total_games": 0,
                    "defaults": {"source": None, "manifest_hash": None},
                }

            proj_row = self.db._conn.execute(
                "SELECT id, name FROM projects WHERE id = ?", (project_id,)
            ).fetchone()

            src_rows = self.db._conn.execute(
                """
                SELECT source, COUNT(*) AS games, MAX(played_at) AS latest_at
                FROM meta_games
                WHERE project_id = ?
                GROUP BY source
                ORDER BY games DESC, latest_at DESC
                """,
                (project_id,),
            ).fetchall()

            man_rows = self.db._conn.execute(
                """
                SELECT m.manifest_hash,
                       m.patch_label,
                       m.notes,
                       m.num_lanes,
                       m.first_seen_at,
                       m.updated_at,
                       COUNT(g.id) AS games,
                       MAX(g.played_at) AS latest_at
                FROM meta_manifests m
                LEFT JOIN meta_games g
                    ON g.project_id = m.project_id AND g.manifest_hash = m.manifest_hash
                WHERE m.project_id = ?
                GROUP BY m.manifest_hash
                ORDER BY COALESCE(MAX(g.played_at), m.updated_at) DESC
                """,
                (project_id,),
            ).fetchall()

            mode_rows = self.db._conn.execute(
                """
                SELECT mode, COUNT(*) AS games
                FROM meta_games
                WHERE project_id = ?
                GROUP BY mode
                ORDER BY games DESC
                """,
                (project_id,),
            ).fetchall()

            tier_rows = self.db._conn.execute(
                """
                SELECT DISTINCT gp.tier
                FROM meta_game_players gp
                JOIN meta_games g ON g.id = gp.game_id
                WHERE g.project_id = ? AND gp.tier IS NOT NULL AND gp.tier != ''
                ORDER BY gp.tier ASC
                """,
                (project_id,),
            ).fetchall()

            persp_rows = self.db._conn.execute(
                """
                SELECT DISTINCT critic_perspective
                FROM meta_games
                WHERE project_id = ? AND critic_perspective IS NOT NULL AND critic_perspective != ''
                ORDER BY critic_perspective ASC
                """,
                (project_id,),
            ).fetchall()

            rating_row = self.db._conn.execute(
                """
                SELECT MIN(gp.rating) AS min_r, MAX(gp.rating) AS max_r, COUNT(DISTINCT g.id) AS total_games
                FROM meta_games g
                LEFT JOIN meta_game_players gp ON gp.game_id = g.id
                WHERE g.project_id = ?
                """,
                (project_id,),
            ).fetchone()

            cat_factions = self.db._conn.execute(
                """
                SELECT DISTINCT faction
                FROM meta_card_catalog
                WHERE project_id = ? AND faction IS NOT NULL AND faction != ''
                ORDER BY faction ASC
                """,
                (project_id,),
            ).fetchall()

            cat_types = self.db._conn.execute(
                """
                SELECT DISTINCT card_type
                FROM meta_card_catalog
                WHERE project_id = ? AND card_type IS NOT NULL AND card_type != ''
                ORDER BY card_type ASC
                """,
                (project_id,),
            ).fetchall()

        sources = [dict(r) for r in src_rows]
        manifests = [dict(r) for r in man_rows]
        default_source = sources[0]["source"] if sources else None
        default_manifest = manifests[0]["manifest_hash"] if manifests else None

        return sanitize_json_value(
            {
                "project": str(proj_row["name"]) if proj_row else project,
                "project_id": project_id,
                "projects": [dict(r) for r in proj_rows],
                "sources": sources,
                "manifests": manifests,
                "modes": [dict(r) for r in mode_rows],
                "tiers": [str(r["tier"]) for r in tier_rows],
                "factions": [str(r["faction"]) for r in cat_factions],
                "card_types": [str(r["card_type"]) for r in cat_types],
                "critic_perspectives": [str(r["critic_perspective"]) for r in persp_rows],
                "rating_range": {
                    "min": rating_row["min_r"] if rating_row else None,
                    "max": rating_row["max_r"] if rating_row else None,
                },
                "total_games": int(rating_row["total_games"] or 0) if rating_row else 0,
                "defaults": {
                    "source": default_source,
                    "manifest_hash": default_manifest,
                },
            }
        )

    def _load_catalog_map(
        self, project_id: str, manifest_hash: str | None
    ) -> dict[int, dict[str, Any]]:
        with self.db._lock:
            if manifest_hash and manifest_hash != "all":
                rows = self.db._conn.execute(
                    """
                    SELECT mc.card_id, c.card_key, mc.display_name, mc.mana_cost, mc.faction,
                           mc.card_type, mc.rarity, mc.attack, mc.health, mc.tags_json, mc.text, mc.extra_json
                    FROM meta_card_catalog mc
                    JOIN meta_cards c ON c.id = mc.card_id
                    WHERE mc.project_id = ? AND mc.manifest_hash = ?
                    """,
                    (project_id, manifest_hash),
                ).fetchall()
            else:
                rows = self.db._conn.execute(
                    """
                    SELECT mc.card_id, c.card_key, mc.display_name, mc.mana_cost, mc.faction,
                           mc.card_type, mc.rarity, mc.attack, mc.health, mc.tags_json, mc.text, mc.extra_json,
                           mm.updated_at
                    FROM meta_card_catalog mc
                    JOIN meta_cards c ON c.id = mc.card_id
                    LEFT JOIN meta_manifests mm
                        ON mm.project_id = mc.project_id AND mm.manifest_hash = mc.manifest_hash
                    WHERE mc.project_id = ?
                    ORDER BY COALESCE(mm.updated_at, 0) ASC
                    """,
                    (project_id,),
                ).fetchall()

        catalog: dict[int, dict[str, Any]] = {}
        for r in rows:
            cid = int(r["card_id"])
            catalog[cid] = {
                "card_id": cid,
                "card_key": str(r["card_key"]),
                "display_name": str(r["display_name"]),
                "mana_cost": r["mana_cost"],
                "faction": r["faction"],
                "card_type": r["card_type"],
                "rarity": r["rarity"],
                "attack": r["attack"],
                "health": r["health"],
                "tags": json.loads(r["tags_json"] or "[]"),
                "text": str(r["text"] or ""),
                "extra": json.loads(r["extra_json"] or "{}"),
            }
        return catalog

    def _card_matches_catalog_filter(
        self, cat: dict[str, Any] | None, flt: MetaFilter
    ) -> bool:
        has_cat_filter = any(
            v is not None and v != ""
            for v in (flt.faction, flt.card_type, flt.mana_cost_min, flt.mana_cost_max, flt.tag)
        )
        if not has_cat_filter:
            return True
        if cat is None:
            return False
        if flt.faction and str(cat.get("faction") or "").lower() != flt.faction.lower():
            return False
        if flt.card_type and str(cat.get("card_type") or "").lower() != flt.card_type.lower():
            return False
        mc = cat.get("mana_cost")
        if flt.mana_cost_min is not None and (mc is None or int(mc) < flt.mana_cost_min):
            return False
        if flt.mana_cost_max is not None and (mc is None or int(mc) > flt.mana_cost_max):
            return False
        if flt.tag:
            tags_lower = {str(t).lower() for t in (cat.get("tags") or [])}
            if flt.tag.lower() not in tags_lower:
                return False
        return True

    def _load_filtered_dataset(
        self, flt: MetaFilter, project_id: str, *, include_plays: bool = True
    ) -> dict[str, Any]:
        clauses = ["g.project_id = ?"]
        params: list[Any] = [project_id]

        if flt.source and flt.source != "all":
            clauses.append("g.source = ?")
            params.append(flt.source)
        if flt.manifest_hash and flt.manifest_hash != "all":
            clauses.append("g.manifest_hash = ?")
            params.append(flt.manifest_hash)
        if flt.mode and flt.mode != "all":
            clauses.append("g.mode = ?")
            params.append(flt.mode)
        if flt.since_ms is not None:
            clauses.append("g.played_at >= ?")
            params.append(int(flt.since_ms))
        if flt.until_ms is not None:
            clauses.append("g.played_at <= ?")
            params.append(int(flt.until_ms))

        where = " AND ".join(clauses)

        with self.db._lock:
            top_k_names: set[str] | None = None
            if flt.top_k_agents is not None and flt.top_k_agents > 0:
                tk_rows = self.db._conn.execute(
                    """
                    SELECT c.name
                    FROM agent_ratings ar
                    JOIN checkpoints c ON c.id = ar.agent_id
                    WHERE ar.project_id = ? AND c.is_bot = 0 AND ar.games_played > 0
                    ORDER BY ar.glicko_mu DESC
                    LIMIT ?
                    """,
                    (project_id, int(flt.top_k_agents)),
                ).fetchall()
                top_k_names = {str(r["name"]) for r in tk_rows}

            card_rows = self.db._conn.execute(
                "SELECT id, card_key FROM meta_cards WHERE project_id = ?",
                (project_id,),
            ).fetchall()
            id_to_key = {int(r["id"]): str(r["card_key"]) for r in card_rows}
            key_to_id = {str(r["card_key"]): int(r["id"]) for r in card_rows}

            game_rows = self.db._conn.execute(
                f"""
                SELECT g.id, g.source, g.mode, g.manifest_hash, g.num_players, g.num_lanes,
                       g.rounds, g.critic_model, g.critic_perspective, g.played_at, g.day_bucket
                FROM meta_games g
                WHERE {where}
                ORDER BY g.played_at ASC, g.id ASC
                """,
                tuple(params),
            ).fetchall()

            games_by_id: dict[int, dict[str, Any]] = {}
            for r in game_rows:
                games_by_id[int(r["id"])] = {
                    "id": int(r["id"]),
                    "source": str(r["source"]),
                    "mode": str(r["mode"]),
                    "manifest_hash": str(r["manifest_hash"]),
                    "num_players": int(r["num_players"]),
                    "num_lanes": int(r["num_lanes"]),
                    "rounds": int(r["rounds"] or 0),
                    "critic_model": r["critic_model"],
                    "critic_perspective": r["critic_perspective"],
                    "played_at": int(r["played_at"]),
                    "day_bucket": int(r["day_bucket"]),
                    "players": [],
                }

            if games_by_id:
                gids = list(games_by_id.keys())
                for i in range(0, len(gids), 500):
                    chunk = gids[i : i + 500]
                    gp_rows = self.db._conn.execute(
                        f"""
                        SELECT id, game_id, seat, player_key, actor_type, rating,
                               tier, deck_name, deck_hash, rank, outcome_score
                        FROM meta_game_players
                        WHERE game_id IN ({','.join('?' for _ in chunk)})
                        ORDER BY game_id ASC, seat ASC
                        """,
                        tuple(chunk),
                    ).fetchall()
                    for gp in gp_rows:
                        games_by_id[int(gp["game_id"])]["players"].append(dict(gp))

            # Filter whole games by participant constraints so opponent/regression pairings stay intact.
            filtered_games: list[dict[str, Any]] = []
            filtered_gp_ids: list[int] = []
            for g in games_by_id.values():
                players = g["players"]
                if len(players) < 2:
                    continue
                if flt.exclude_bots and any(p["actor_type"] == "bot" for p in players):
                    continue
                if flt.min_rating is not None:
                    if any(p["rating"] is None or float(p["rating"]) < flt.min_rating for p in players):
                        continue
                if flt.max_rating is not None:
                    if any(p["rating"] is None or float(p["rating"]) > flt.max_rating for p in players):
                        continue
                if flt.tier:
                    if any(str(p.get("tier") or "") != flt.tier for p in players):
                        continue
                if top_k_names is not None:
                    if any(str(p["player_key"]) not in top_k_names for p in players):
                        continue
                filtered_games.append(g)
                for p in players:
                    filtered_gp_ids.append(int(p["id"]))

            cards_by_gp: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
            plays_by_gp: dict[int, list[dict[str, Any]]] = defaultdict(list)

            if filtered_gp_ids:
                for i in range(0, len(filtered_gp_ids), 500):
                    chunk = filtered_gp_ids[i : i + 500]
                    ph = ",".join("?" for _ in chunk)
                    gc_rows = self.db._conn.execute(
                        f"""
                        SELECT game_player_id, card_id, copies_in_deck, in_opening_hand,
                               mulligan_offered, mulligan_kept, drawn_count, played_count,
                               first_turn_played, lane_mask,
                               critic_play_wpa_sum, critic_play_wpa_count,
                               critic_draw_wpa_sum, critic_draw_wpa_count
                        FROM meta_game_cards
                        WHERE game_player_id IN ({ph})
                        """,
                        tuple(chunk),
                    ).fetchall()
                    for r in gc_rows:
                        cards_by_gp[int(r["game_player_id"])][int(r["card_id"])] = dict(r)

                    if include_plays:
                        cp_rows = self.db._conn.execute(
                            f"""
                            SELECT game_player_id, card_id, turn, lane, lane_context,
                                   win_prob_before, win_prob_after
                            FROM meta_card_plays
                            WHERE game_player_id IN ({ph})
                            """,
                            tuple(chunk),
                        ).fetchall()
                        for r in cp_rows:
                            plays_by_gp[int(r["game_player_id"])].append(dict(r))

        catalog = self._load_catalog_map(project_id, flt.manifest_hash)
        perspectives = sorted(
            {
                str(g["critic_perspective"])
                for g in filtered_games
                if g.get("critic_perspective")
            }
        )
        selected_perspective: str | None = None
        if flt.critic_perspective:
            selected_perspective = flt.critic_perspective
        else:
            for pref in ("reference_public", "opponent", "actor"):
                if pref in perspectives:
                    selected_perspective = pref
                    break
            if selected_perspective is None and perspectives:
                selected_perspective = perspectives[0]

        num_lanes = max([4] + [g["num_lanes"] for g in filtered_games])
        return {
            "project_id": project_id,
            "id_to_key": id_to_key,
            "key_to_id": key_to_id,
            "catalog": catalog,
            "games": filtered_games,
            "cards_by_gp": cards_by_gp,
            "plays_by_gp": plays_by_gp,
            "perspectives": perspectives,
            "selected_perspective": selected_perspective,
            "num_lanes": num_lanes,
        }

    # ------------------------------------------------------------------
    # Ridge Regression & Synergy Core
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ridge_impact(
        games: list[dict[str, Any]],
        cards_by_gp: dict[int, dict[int, dict[str, Any]]],
        card_ids: list[int],
    ) -> dict[int, float]:
        if len(games) < 5 or not card_ids:
            return {}
        cid_to_col = {cid: idx for idx, cid in enumerate(card_ids)}
        n_cards = len(card_ids)

        rows_x: list[np.ndarray] = []
        rows_y: list[float] = []

        one_v_one = [g for g in games if len(g["players"]) == 2]
        multi = [g for g in games if len(g["players"]) > 2]

        if one_v_one:
            n_cols = 2 + n_cards
            for g in one_v_one:
                p0, p1 = g["players"][0], g["players"][1]
                vec = np.zeros(n_cols, dtype=np.float64)
                vec[0] = 1.0
                if p0["rating"] is not None and p1["rating"] is not None:
                    vec[1] = (float(p0["rating"]) - float(p1["rating"])) / 400.0
                c0 = cards_by_gp.get(int(p0["id"]), {})
                c1 = cards_by_gp.get(int(p1["id"]), {})
                for cid, gc in c0.items():
                    if gc["drawn_count"] > 0:
                        col = cid_to_col.get(cid)
                        if col is not None:
                            vec[2 + col] += 1.0
                for cid, gc in c1.items():
                    if gc["drawn_count"] > 0:
                        col = cid_to_col.get(cid)
                        if col is not None:
                            vec[2 + col] -= 1.0
                rows_x.append(vec)
                rows_y.append(float(p0["outcome_score"]) - 0.5)

        if multi:
            n_cols = 2 + n_cards
            for g in multi:
                players = g["players"]
                k = len(players)
                mean_score = sum(float(p["outcome_score"]) for p in players) / k
                rated = [float(p["rating"]) for p in players if p["rating"] is not None]
                mean_rating = sum(rated) / len(rated) if rated else None

                mean_drawn: dict[int, float] = defaultdict(float)
                for p in players:
                    for cid, gc in cards_by_gp.get(int(p["id"]), {}).items():
                        if gc["drawn_count"] > 0:
                            mean_drawn[cid] += 1.0 / k

                for p in players:
                    vec = np.zeros(n_cols, dtype=np.float64)
                    vec[0] = 1.0 if int(p["seat"]) == 0 else 0.0
                    if mean_rating is not None and p["rating"] is not None:
                        vec[1] = (float(p["rating"]) - mean_rating) / 400.0
                    pc = cards_by_gp.get(int(p["id"]), {})
                    for cid, avg_d in mean_drawn.items():
                        col = cid_to_col.get(cid)
                        if col is None:
                            continue
                        d = 1.0 if (cid in pc and pc[cid]["drawn_count"] > 0) else 0.0
                        vec[2 + col] = d - avg_d
                    rows_x.append(vec)
                    rows_y.append(float(p["outcome_score"]) - mean_score)

        if len(rows_x) < 5:
            return {}

        x_mat = np.vstack(rows_x)
        y_vec = np.asarray(rows_y, dtype=np.float64)

        xtx = x_mat.T @ x_mat
        xty = x_mat.T @ y_vec
        reg = np.zeros(xtx.shape[0], dtype=np.float64)
        reg[0] = 1e-4
        reg[1] = 1e-4
        reg[2:] = _RIDGE_LAMBDA
        xtx += np.diag(reg)

        try:
            beta = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            beta, *_ = np.linalg.lstsq(xtx, xty, rcond=None)

        out: dict[int, float] = {}
        for cid, col in cid_to_col.items():
            val = float(beta[2 + col])
            if math.isfinite(val):
                out[cid] = val
        return out

    @staticmethod
    def _compute_pair_metrics(
        games: list[dict[str, Any]],
        cards_by_gp: dict[int, dict[int, dict[str, Any]]],
        card_base_wr: dict[int, float],
        base_wr: float,
        min_pair_games: int,
    ) -> dict[str, dict[tuple[int, int], dict[str, Any]]]:
        """Compute multi-level synergy (played, same_lane, deck) and 1v1 opponent counters."""
        total_pg = sum(len(g["players"]) for g in games)
        if total_pg == 0:
            return {"played": {}, "same_lane": {}, "deck": {}, "counters": {}}

        played_counts: dict[tuple[int, int], list[float]] = {}
        lane_counts: dict[tuple[int, int], list[float]] = {}
        deck_counts: dict[tuple[int, int], list[float]] = {}
        counter_counts: dict[tuple[int, int], list[float]] = {}

        single_played_freq: dict[int, int] = defaultdict(int)
        single_deck_freq: dict[int, int] = defaultdict(int)
        single_drawn_freq: dict[int, int] = defaultdict(int)

        def _acc(store: dict[tuple[int, int], list[float]], k: tuple[int, int], sc: float) -> None:
            entry = store.get(k)
            if entry is None:
                store[k] = [1.0, sc]
            else:
                entry[0] += 1.0
                entry[1] += sc

        for g in games:
            players = g["players"]
            for p in players:
                gp_id = int(p["id"])
                sc = float(p["outcome_score"])
                cmap = cards_by_gp.get(gp_id, {})
                deck_ids = sorted(cmap.keys())
                for cid in deck_ids:
                    single_deck_freq[cid] += 1
                    gc = cmap[cid]
                    if gc["drawn_count"] > 0:
                        single_drawn_freq[cid] += 1
                    if gc["played_count"] > 0:
                        single_played_freq[cid] += 1

                for i in range(len(deck_ids)):
                    a = deck_ids[i]
                    gca = cmap[a]
                    pa = gca["played_count"] > 0
                    la = int(gca["lane_mask"] or 0)
                    for j in range(i + 1, len(deck_ids)):
                        b = deck_ids[j]
                        gcb = cmap[b]
                        _acc(deck_counts, (a, b), sc)
                        if pa and gcb["played_count"] > 0:
                            _acc(played_counts, (a, b), sc)
                            if (la & int(gcb["lane_mask"] or 0)) != 0:
                                _acc(lane_counts, (a, b), sc)

            if len(players) == 2:
                p0, p1 = players[0], players[1]
                s0 = float(p0["outcome_score"])
                s1 = float(p1["outcome_score"])
                d0 = [cid for cid, gc in cards_by_gp.get(int(p0["id"]), {}).items() if gc["drawn_count"] > 0]
                d1 = [cid for cid, gc in cards_by_gp.get(int(p1["id"]), {}).items() if gc["drawn_count"] > 0]
                for a in d0:
                    for b in d1:
                        if a == b:
                            continue
                        _acc(counter_counts, (a, b), s0)
                        _acc(counter_counts, (b, a), s1)

        logit_base = _logit(base_wr)

        def _finalize(
            store: dict[tuple[int, int], list[float]],
            freq_map: dict[int, int],
            *,
            is_counter: bool = False,
        ) -> dict[tuple[int, int], dict[str, Any]]:
            out: dict[tuple[int, int], dict[str, Any]] = {}
            for (a, b), (n_f, wins_f) in store.items():
                n = int(n_f)
                if n < min_pair_games:
                    continue
                wr = wins_f / n
                pa = card_base_wr.get(a, base_wr)
                pb = card_base_wr.get(b, base_wr)
                if is_counter:
                    p_exp = _sigmoid(_logit(pa) + _logit(1.0 - pb) - _logit(0.5))
                else:
                    p_exp = _sigmoid(_logit(pa) + _logit(pb) - logit_base)
                raw_lift = wr - p_exp
                shrunk_lift = (n / (n + _SYNERGY_SHRINK_K)) * raw_lift
                fa = max(1, freq_map.get(a, 1)) / max(1, total_pg)
                fb = max(1, freq_map.get(b, 1)) / max(1, total_pg)
                fab = max(1e-9, n / max(1, total_pg))
                pmi = math.log2(max(1e-9, fab / max(1e-9, fa * fb)))
                out[(a, b)] = {
                    "games": n,
                    "wins": wins_f,
                    "win_rate": wr,
                    "expected_wr": p_exp,
                    "lift_raw": raw_lift,
                    "lift": shrunk_lift,
                    "pmi": pmi,
                }
            return out

        return {
            "played": _finalize(played_counts, single_played_freq),
            "same_lane": _finalize(lane_counts, single_played_freq),
            "deck": _finalize(deck_counts, single_deck_freq),
            "counters": _finalize(counter_counts, single_drawn_freq, is_counter=True),
        }

    # ------------------------------------------------------------------
    # Core Overview & Card Aggregation
    # ------------------------------------------------------------------

    def _compute_full_state(self, flt: MetaFilter) -> dict[str, Any]:
        project_id = self.db.resolve_project_id(flt.project)
        if not project_id:
            return {
                "project": flt.project,
                "project_id": None,
                "kpis": {
                    "games": 0,
                    "player_games": 0,
                    "cards": 0,
                    "avg_rounds": 0.0,
                    "p1_win_rate": None,
                    "critic_coverage_pct": 0.0,
                    "critic_perspective": flt.critic_perspective,
                },
                "cards": [],
                "synergies": {"played": [], "same_lane": [], "deck": [], "counters": []},
                "_internal": None,
            }

        ds = self._load_filtered_dataset(flt, project_id, include_plays=True)
        games = ds["games"]
        cards_by_gp = ds["cards_by_gp"]
        plays_by_gp = ds["plays_by_gp"]
        id_to_key = ds["id_to_key"]
        catalog = ds["catalog"]
        selected_persp = ds["selected_perspective"]
        num_lanes = ds["num_lanes"]

        total_games = len(games)
        total_pg = sum(len(g["players"]) for g in games)
        if total_games == 0 or total_pg == 0:
            return {
                "project": flt.project,
                "project_id": project_id,
                "kpis": {
                    "games": 0,
                    "player_games": 0,
                    "cards": 0,
                    "avg_rounds": 0.0,
                    "p1_win_rate": None,
                    "critic_coverage_pct": 0.0,
                    "critic_perspective": selected_persp,
                },
                "cards": [],
                "synergies": {"played": [], "same_lane": [], "deck": [], "counters": []},
                "_internal": None,
            }

        p1_scores = [
            float(g["players"][0]["outcome_score"])
            for g in games
            if g["players"] and int(g["players"][0]["seat"]) == 0
        ]
        p1_wr = _safe_div(sum(p1_scores), len(p1_scores))
        avg_rounds = sum(int(g["rounds"] or 0) for g in games) / total_games
        base_wr = sum(float(p["outcome_score"]) for g in games for p in g["players"]) / total_pg

        # Per-card accumulators
        stats: dict[int, dict[str, Any]] = {}

        def _card_acc(cid: int) -> dict[str, Any]:
            st = stats.get(cid)
            if st is None:
                st = {
                    "games_in_deck": 0,
                    "copies_total": 0,
                    "deck_wins": 0.0,
                    "oh_games": 0,
                    "oh_wins": 0.0,
                    "mull_offered": 0,
                    "mull_kept": 0,
                    "mull_kept_wins": 0.0,
                    "mull_tossed_wins": 0.0,
                    "drawn_games": 0,
                    "drawn_wins": 0.0,
                    "gns_games": 0,
                    "gns_wins": 0.0,
                    "played_games": 0,
                    "played_wins": 0.0,
                    "not_played_in_deck_games": 0,
                    "not_played_in_deck_wins": 0.0,
                    "drawn_not_played_games": 0,
                    "drawn_not_played_wins": 0.0,
                    "total_drawn_copies": 0,
                    "total_played_copies": 0,
                    "first_turn_sum": 0,
                    "first_turn_count": 0,
                    "rounds_when_played_sum": 0,
                    # Mirror-filtered raw accumulators (identical to full when exclude_mirrors=False)
                    "nm_oh_games": 0,
                    "nm_oh_wins": 0.0,
                    "nm_drawn_games": 0,
                    "nm_drawn_wins": 0.0,
                    "nm_gns_games": 0,
                    "nm_gns_wins": 0.0,
                    "nm_played_games": 0,
                    "nm_played_wins": 0.0,
                    "nm_not_played_games": 0,
                    "nm_not_played_wins": 0.0,
                    "nm_dnp_games": 0,
                    "nm_dnp_wins": 0.0,
                    # Critic WPA
                    "critic_play_sum": 0.0,
                    "critic_play_cnt": 0,
                    "critic_draw_sum": 0.0,
                    "critic_draw_cnt": 0,
                    # Seat split (GIH)
                    "p1_games": 0,
                    "p1_wins": 0.0,
                    "p2_games": 0,
                    "p2_wins": 0.0,
                    # Decks
                    "decks": defaultdict(lambda: [0, 0.0]),
                    # Lane stats: lane_idx -> [plays, wins, wpa_sum, wpa_cnt]
                    "lanes": {i: [0, 0.0, 0.0, 0] for i in range(num_lanes)},
                    "global_plays": [0, 0.0, 0.0, 0],
                    "lane_contexts": defaultdict(lambda: [0, 0.0, 0.0, 0]),
                    # Turn curve: bucket (1..13) -> [plays, wins, wpa_sum, wpa_cnt]
                    "turns": defaultdict(lambda: [0, 0.0, 0.0, 0]),
                    "all_plays_turn_sum": 0,
                    "all_plays_count": 0,
                }
                stats[cid] = st
            return st

        critic_games_count = 0
        for g in games:
            persp = g.get("critic_perspective")
            use_critic = (
                selected_persp is None
                or persp is None
                or str(persp) == selected_persp
            )
            game_has_critic = False
            players = g["players"]
            is_1v1 = len(players) == 2

            # Precompute 1v1 mirror sets for this game
            both_in_deck: set[int] = set()
            both_drawn: set[int] = set()
            both_played: set[int] = set()
            both_oh: set[int] = set()
            if is_1v1 and flt.exclude_mirrors:
                c0 = cards_by_gp.get(int(players[0]["id"]), {})
                c1 = cards_by_gp.get(int(players[1]["id"]), {})
                common = set(c0.keys()) & set(c1.keys())
                both_in_deck = common
                for cid in common:
                    if c0[cid]["drawn_count"] > 0 and c1[cid]["drawn_count"] > 0:
                        both_drawn.add(cid)
                    if c0[cid]["played_count"] > 0 and c1[cid]["played_count"] > 0:
                        both_played.add(cid)
                    if c0[cid]["in_opening_hand"] > 0 and c1[cid]["in_opening_hand"] > 0:
                        both_oh.add(cid)

            for p in players:
                gp_id = int(p["id"])
                sc = float(p["outcome_score"])
                seat = int(p["seat"])
                deck_label = str(p.get("deck_name") or p.get("deck_hash") or "Unnamed")
                cmap = cards_by_gp.get(gp_id, {})

                for cid, gc in cmap.items():
                    st = _card_acc(cid)
                    st["games_in_deck"] += 1
                    st["copies_total"] += int(gc["copies_in_deck"] or 1)
                    st["deck_wins"] += sc
                    st["decks"][deck_label][0] += 1
                    st["decks"][deck_label][1] += sc

                    in_oh = int(gc["in_opening_hand"] or 0) > 0
                    drawn_c = int(gc["drawn_count"] or 0)
                    played_c = int(gc["played_count"] or 0)
                    drawn = drawn_c > 0
                    played = played_c > 0

                    st["total_drawn_copies"] += drawn_c
                    st["total_played_copies"] += played_c

                    if in_oh:
                        st["oh_games"] += 1
                        st["oh_wins"] += sc
                        if cid not in both_oh:
                            st["nm_oh_games"] += 1
                            st["nm_oh_wins"] += sc

                    if int(gc["mulligan_offered"] or 0) > 0:
                        st["mull_offered"] += 1
                        if int(gc["mulligan_kept"] or 0) > 0:
                            st["mull_kept"] += 1
                            st["mull_kept_wins"] += sc
                        else:
                            st["mull_tossed_wins"] += sc

                    if drawn:
                        st["drawn_games"] += 1
                        st["drawn_wins"] += sc
                        if seat == 0:
                            st["p1_games"] += 1
                            st["p1_wins"] += sc
                        elif seat == 1:
                            st["p2_games"] += 1
                            st["p2_wins"] += sc
                        if cid not in both_drawn:
                            st["nm_drawn_games"] += 1
                            st["nm_drawn_wins"] += sc
                    else:
                        st["gns_games"] += 1
                        st["gns_wins"] += sc
                        if cid not in both_in_deck:
                            st["nm_gns_games"] += 1
                            st["nm_gns_wins"] += sc

                    if played:
                        st["played_games"] += 1
                        st["played_wins"] += sc
                        if gc["first_turn_played"] is not None:
                            st["first_turn_sum"] += int(gc["first_turn_played"])
                            st["first_turn_count"] += 1
                        st["rounds_when_played_sum"] += int(g["rounds"] or 0)
                        if cid not in both_played:
                            st["nm_played_games"] += 1
                            st["nm_played_wins"] += sc
                    else:
                        st["not_played_in_deck_games"] += 1
                        st["not_played_in_deck_wins"] += sc
                        if cid not in both_in_deck:
                            st["nm_not_played_games"] += 1
                            st["nm_not_played_wins"] += sc
                        if drawn:
                            st["drawn_not_played_games"] += 1
                            st["drawn_not_played_wins"] += sc
                            if cid not in both_drawn:
                                st["nm_dnp_games"] += 1
                                st["nm_dnp_wins"] += sc

                    if use_critic:
                        cp_cnt = int(gc["critic_play_wpa_count"] or 0)
                        if cp_cnt > 0 and gc["critic_play_wpa_sum"] is not None:
                            st["critic_play_sum"] += float(gc["critic_play_wpa_sum"])
                            st["critic_play_cnt"] += cp_cnt
                            game_has_critic = True
                        cd_cnt = int(gc["critic_draw_wpa_count"] or 0)
                        if cd_cnt > 0 and gc["critic_draw_wpa_sum"] is not None:
                            st["critic_draw_sum"] += float(gc["critic_draw_wpa_sum"])
                            st["critic_draw_cnt"] += cd_cnt
                            game_has_critic = True

                for pl in plays_by_gp.get(gp_id, []):
                    cid = int(pl["card_id"])
                    st = _card_acc(cid)
                    turn = int(pl["turn"] or 0)
                    t_bucket = min(13, max(1, turn)) if turn > 0 else 1
                    st["all_plays_turn_sum"] += turn
                    st["all_plays_count"] += 1

                    wpa_val: float | None = None
                    if (
                        use_critic
                        and pl["win_prob_before"] is not None
                        and pl["win_prob_after"] is not None
                    ):
                        wpa_val = float(pl["win_prob_after"]) - float(pl["win_prob_before"])

                    tb = st["turns"][t_bucket]
                    tb[0] += 1
                    tb[1] += sc
                    if wpa_val is not None:
                        tb[2] += wpa_val
                        tb[3] += 1

                    lane = pl["lane"]
                    if lane is None:
                        gp_arr = st["global_plays"]
                        gp_arr[0] += 1
                        gp_arr[1] += sc
                        if wpa_val is not None:
                            gp_arr[2] += wpa_val
                            gp_arr[3] += 1
                    elif 0 <= int(lane) < num_lanes:
                        ln_arr = st["lanes"][int(lane)]
                        ln_arr[0] += 1
                        ln_arr[1] += sc
                        if wpa_val is not None:
                            ln_arr[2] += wpa_val
                            ln_arr[3] += 1

                    lctx = pl.get("lane_context")
                    if lctx:
                        ctx_arr = st["lane_contexts"][str(lctx)]
                        ctx_arr[0] += 1
                        ctx_arr[1] += sc
                        if wpa_val is not None:
                            ctx_arr[2] += wpa_val
                            ctx_arr[3] += 1

            if game_has_critic:
                critic_games_count += 1

        all_card_ids = sorted(stats.keys())
        impact_map = self._compute_ridge_impact(games, cards_by_gp, all_card_ids)

        card_base_wr = {
            cid: (_safe_div(st["drawn_wins"], st["drawn_games"]) or base_wr)
            for cid, st in stats.items()
        }
        eff_min_pair = (
            flt.min_pair_games
            if flt.min_pair_games is not None
            else max(3, min(15, total_pg // 50))
        )
        pair_metrics = self._compute_pair_metrics(
            games, cards_by_gp, card_base_wr, base_wr, eff_min_pair
        )

        # Build per-card synergy lookup
        card_synergies: dict[int, dict[str, list[dict[str, Any]]]] = {
            cid: {"played": [], "same_lane": [], "deck": [], "counters": []}
            for cid in all_card_ids
        }
        for level in ("played", "same_lane", "deck"):
            for (a, b), m in pair_metrics[level].items():
                key_a = id_to_key.get(a, str(a))
                key_b = id_to_key.get(b, str(b))
                card_synergies[a][level].append({"partner": key_b, "partner_id": b, **m})
                card_synergies[b][level].append({"partner": key_a, "partner_id": a, **m})

        for (a, b), m in pair_metrics["counters"].items():
            if a in card_synergies:
                card_synergies[a]["counters"].append(
                    {"opponent": id_to_key.get(b, str(b)), "opponent_id": b, **m}
                )

        cards_out: list[dict[str, Any]] = []
        for cid in all_card_ids:
            st = stats[cid]
            ckey = id_to_key.get(cid, str(cid))
            cat = catalog.get(cid)
            if not self._card_matches_catalog_filter(cat, flt):
                continue

            g_deck = st["games_in_deck"]
            if g_deck < flt.min_games and st["drawn_games"] < flt.min_games:
                # Still include if min_games == 0, else skip
                if flt.min_games > 0:
                    continue

            use_nm = bool(flt.exclude_mirrors)
            oh_n = st["nm_oh_games"] if use_nm else st["oh_games"]
            oh_w = st["nm_oh_wins"] if use_nm else st["oh_wins"]
            dr_n = st["nm_drawn_games"] if use_nm else st["drawn_games"]
            dr_w = st["nm_drawn_wins"] if use_nm else st["drawn_wins"]
            gns_n = st["nm_gns_games"] if use_nm else st["gns_games"]
            gns_w = st["nm_gns_wins"] if use_nm else st["gns_wins"]
            pl_n = st["nm_played_games"] if use_nm else st["played_games"]
            pl_w = st["nm_played_wins"] if use_nm else st["played_wins"]
            np_n = st["nm_not_played_games"] if use_nm else st["not_played_in_deck_games"]
            np_w = st["nm_not_played_wins"] if use_nm else st["not_played_in_deck_wins"]
            dnp_n = st["nm_dnp_games"] if use_nm else st["drawn_not_played_games"]
            dnp_w = st["nm_dnp_wins"] if use_nm else st["drawn_not_played_wins"]

            gih_wr = _safe_div(dr_w, dr_n)
            gns_wr = _safe_div(gns_w, gns_n)
            oh_wr = _safe_div(oh_w, oh_n)
            played_wr = _safe_div(pl_w, pl_n)
            not_played_wr = _safe_div(np_w, np_n)
            dnp_wr = _safe_div(dnp_w, dnp_n)

            ci_low, ci_high = _wilson_ci(dr_w, dr_n)
            pl_ci_low, pl_ci_high = _wilson_ci(pl_w, pl_n)

            drawn_wpa_raw = (
                (gih_wr - gns_wr) if (gih_wr is not None and gns_wr is not None) else None
            )
            n_eff_drawn = min(dr_n, gns_n)
            drawn_wpa = (
                (n_eff_drawn / (n_eff_drawn + _WPA_SHRINK_K)) * drawn_wpa_raw
                if drawn_wpa_raw is not None
                else None
            )

            played_lift_raw = (
                (played_wr - not_played_wr)
                if (played_wr is not None and not_played_wr is not None)
                else None
            )
            n_eff_played = min(pl_n, np_n)
            played_lift = (
                (n_eff_played / (n_eff_played + _WPA_SHRINK_K)) * played_lift_raw
                if played_lift_raw is not None
                else None
            )

            play_vs_hold = (
                (played_wr - dnp_wr) if (played_wr is not None and dnp_wr is not None) else None
            )

            critic_play_wpa = _safe_div(st["critic_play_sum"], st["critic_play_cnt"])
            critic_draw_wpa = _safe_div(st["critic_draw_sum"], st["critic_draw_cnt"])

            iso_impact = impact_map.get(cid)
            drawn_presence = st["drawn_games"] / total_pg
            meta_war = (
                iso_impact * drawn_presence * 100.0 if iso_impact is not None else None
            )

            # Synergy summaries
            syn_played = sorted(
                card_synergies[cid]["played"], key=lambda x: x["lift"], reverse=True
            )
            syn_lane = sorted(
                card_synergies[cid]["same_lane"], key=lambda x: x["lift"], reverse=True
            )
            syn_deck = sorted(
                card_synergies[cid]["deck"], key=lambda x: x["lift"], reverse=True
            )
            syn_primary = syn_played if syn_played else syn_deck
            pos_lifts = [s["lift"] for s in syn_primary if s["lift"] > 0]
            top_syn = (
                {
                    "partner": syn_primary[0]["partner"],
                    "lift": syn_primary[0]["lift"],
                    "games": syn_primary[0]["games"],
                }
                if syn_primary and syn_primary[0]["lift"] > 0
                else None
            )
            top_lane_syn = (
                {
                    "partner": syn_lane[0]["partner"],
                    "lift": syn_lane[0]["lift"],
                    "games": syn_lane[0]["games"],
                }
                if syn_lane and syn_lane[0]["lift"] > 0
                else None
            )
            neg_syn_sorted = sorted(syn_primary, key=lambda x: x["lift"])
            worst_anti = (
                {
                    "partner": neg_syn_sorted[0]["partner"],
                    "lift": neg_syn_sorted[0]["lift"],
                    "games": neg_syn_sorted[0]["games"],
                }
                if neg_syn_sorted and neg_syn_sorted[0]["lift"] < 0
                else None
            )
            synergy_reliance = (
                sum(pos_lifts[:3]) / len(pos_lifts[:3]) if pos_lifts else 0.0
            )

            # 4-Lane preference & misplay gap
            lane_total_plays = sum(v[0] for v in st["lanes"].values())
            global_plays_n = st["global_plays"][0]
            lanes_list: list[dict[str, Any]] = []
            entropy = 0.0
            card_pl_wr = _safe_div(st["played_wins"], st["played_games"])
            min_lane_plays = max(3, lane_total_plays // 20)

            for l_idx in range(num_lanes):
                lp, lw, l_wpa_s, l_wpa_c = st["lanes"][l_idx]
                share = (lp / lane_total_plays) if lane_total_plays > 0 else 0.0
                if share > 0:
                    entropy -= share * math.log(share)
                l_wr = _safe_div(lw, lp)
                wr_delta = (
                    (lp / (lp + _LANE_SHRINK_K)) * (l_wr - card_pl_wr)
                    if (l_wr is not None and card_pl_wr is not None)
                    else None
                )
                lanes_list.append(
                    {
                        "lane": l_idx,
                        "label": f"L{l_idx}",
                        "plays": lp,
                        "share": share,
                        "win_rate": l_wr,
                        "critic_wpa": _safe_div(l_wpa_s, l_wpa_c),
                        "wr_delta": wr_delta,
                    }
                )

            concentration = (
                max(0.0, min(1.0, 1.0 - (entropy / math.log(num_lanes))))
                if (lane_total_plays > 0 and num_lanes > 1)
                else 0.0
            )
            favored_lane = (
                max(lanes_list, key=lambda x: x["plays"])["lane"]
                if lane_total_plays > 0
                else None
            )
            eligible_best = [
                x for x in lanes_list if x["plays"] >= min_lane_plays and x["wr_delta"] is not None
            ]
            best_lane_obj = (
                max(eligible_best, key=lambda x: float(x["wr_delta"]))
                if eligible_best
                else None
            )
            fav_lane_obj = (
                next((x for x in lanes_list if x["lane"] == favored_lane), None)
                if favored_lane is not None
                else None
            )
            misplay_gap = (
                float(best_lane_obj["wr_delta"]) - float(fav_lane_obj["wr_delta"])
                if (
                    best_lane_obj is not None
                    and fav_lane_obj is not None
                    and fav_lane_obj["wr_delta"] is not None
                )
                else 0.0
            )
            has_misplay_gap = (
                best_lane_obj is not None
                and favored_lane is not None
                and best_lane_obj["lane"] != favored_lane
                and misplay_gap >= 0.02
            )

            p1_wr_c = _safe_div(st["p1_wins"], st["p1_games"])
            p2_wr_c = _safe_div(st["p2_wins"], st["p2_games"])
            seat_delta = (
                (p1_wr_c - p2_wr_c) if (p1_wr_c is not None and p2_wr_c is not None) else None
            )

            inclusion_rate = g_deck / total_pg
            avg_copies = st["copies_total"] / g_deck if g_deck > 0 else 0.0
            draw_to_play = (
                st["played_games"] / st["drawn_games"] if st["drawn_games"] > 0 else None
            )
            avg_first_turn = _safe_div(st["first_turn_sum"], st["first_turn_count"])
            avg_turn_played = _safe_div(st["all_plays_turn_sum"], st["all_plays_count"])

            cards_out.append(
                {
                    "card_id": cid,
                    "card_key": ckey,
                    "display_name": cat["display_name"] if cat else ckey,
                    "catalog": cat,
                    "games_in_deck": g_deck,
                    "games_drawn": st["drawn_games"],
                    "games_played": st["played_games"],
                    "total_played_copies": st["total_played_copies"],
                    "inclusion_rate": inclusion_rate,
                    "avg_copies": avg_copies,
                    "draw_to_play_rate": draw_to_play,
                    "oh_wr": oh_wr,
                    "gih_wr": gih_wr,
                    "gih_ci_low": ci_low,
                    "gih_ci_high": ci_high,
                    "gns_wr": gns_wr,
                    "played_wr": played_wr,
                    "played_ci_low": pl_ci_low,
                    "played_ci_high": pl_ci_high,
                    "play_vs_hold_delta": play_vs_hold,
                    "drawn_wpa": drawn_wpa,
                    "drawn_wpa_raw": drawn_wpa_raw,
                    "played_lift": played_lift,
                    "played_lift_raw": played_lift_raw,
                    "critic_play_wpa": critic_play_wpa,
                    "critic_play_samples": st["critic_play_cnt"],
                    "critic_draw_wpa": critic_draw_wpa,
                    "critic_draw_samples": st["critic_draw_cnt"],
                    "isolated_impact": iso_impact,
                    "meta_war": meta_war,
                    "synergy_reliance": synergy_reliance,
                    "top_synergy": top_syn,
                    "top_same_lane_synergy": top_lane_syn,
                    "worst_anti_synergy": worst_anti,
                    "lane_preference": {
                        "num_lanes": num_lanes,
                        "lane_plays": lane_total_plays,
                        "global_plays": global_plays_n,
                        "favored_lane": favored_lane,
                        "best_lane": best_lane_obj["lane"] if best_lane_obj else None,
                        "concentration": concentration,
                        "misplay_gap": misplay_gap,
                        "has_misplay_gap": has_misplay_gap,
                        "lanes": lanes_list,
                    },
                    "avg_first_turn": avg_first_turn,
                    "avg_turn_played": avg_turn_played,
                    "p1_wr": p1_wr_c,
                    "p2_wr": p2_wr_c,
                    "seat_delta": seat_delta,
                }
            )

        cards_out.sort(
            key=lambda c: (
                -(c["meta_war"] if c["meta_war"] is not None else -1e9),
                -c["inclusion_rate"],
                c["card_key"],
            )
        )

        # Build global synergy lists
        def _top_pairs(level: str, *, reverse: bool = True, limit: int = 40) -> list[dict[str, Any]]:
            items = []
            for (a, b), m in pair_metrics[level].items():
                items.append(
                    {
                        "card_a": id_to_key.get(a, str(a)),
                        "card_b": id_to_key.get(b, str(b)),
                        **m,
                    }
                )
            items.sort(key=lambda x: x["lift"], reverse=reverse)
            return items[:limit]

        global_synergies = {
            "played": _top_pairs("played", reverse=True),
            "played_anti": _top_pairs("played", reverse=False),
            "same_lane": _top_pairs("same_lane", reverse=True),
            "deck": _top_pairs("deck", reverse=True),
            "counters": sorted(
                [
                    {
                        "card": id_to_key.get(a, str(a)),
                        "opponent": id_to_key.get(b, str(b)),
                        **m,
                    }
                    for (a, b), m in pair_metrics["counters"].items()
                ],
                key=lambda x: x["lift"],
            )[:40],
        }

        return {
            "project": flt.project,
            "project_id": project_id,
            "kpis": {
                "games": total_games,
                "player_games": total_pg,
                "cards": len(cards_out),
                "avg_rounds": avg_rounds,
                "p1_win_rate": p1_wr,
                "base_win_rate": base_wr,
                "critic_coverage_pct": (critic_games_count / total_games) * 100.0,
                "critic_perspective": selected_persp,
                "available_perspectives": ds["perspectives"],
            },
            "cards": cards_out,
            "synergies": global_synergies,
            "_internal": {
                "stats": stats,
                "ds": ds,
                "card_synergies": card_synergies,
                "cards_by_id": {c["card_id"]: c for c in cards_out},
            },
        }

    def get_overview(self, flt: MetaFilter) -> dict[str, Any]:
        project_id = self.db.resolve_project_id(flt.project)
        wm = self.db.get_meta_watermark(project_id)
        ck = flt.canonical_key("overview", project_id)
        cached = self._get_cached(ck, project_id, wm)
        if cached is not None:
            return cached

        full = self._compute_full_state(flt)
        payload = {
            "project": full["project"],
            "project_id": full["project_id"],
            "kpis": full["kpis"],
            "cards": full["cards"],
        }
        self._put_cached(ck, project_id, wm, payload)
        return sanitize_json_value(payload)

    def get_synergies(
        self, flt: MetaFilter, level: str = "played", limit: int = 40
    ) -> dict[str, Any]:
        project_id = self.db.resolve_project_id(flt.project)
        wm = self.db.get_meta_watermark(project_id)
        ck = flt.canonical_key("synergies", project_id, f"{level}:{limit}")
        cached = self._get_cached(ck, project_id, wm)
        if cached is not None:
            return cached

        full = self._compute_full_state(flt)
        syn = full["synergies"]
        if level == "counters":
            positive = sorted(syn.get("counters", []), key=lambda x: x["lift"], reverse=True)[:limit]
            negative = sorted(syn.get("counters", []), key=lambda x: x["lift"])[:limit]
        elif level == "same_lane":
            positive = syn.get("same_lane", [])[:limit]
            negative = sorted(syn.get("same_lane", []), key=lambda x: x["lift"])[:limit]
        elif level == "deck":
            positive = syn.get("deck", [])[:limit]
            negative = sorted(syn.get("deck", []), key=lambda x: x["lift"])[:limit]
        else:
            positive = syn.get("played", [])[:limit]
            negative = syn.get("played_anti", [])[:limit]

        payload = {
            "project_id": full["project_id"],
            "level": level,
            "games": full["kpis"]["games"],
            "positive": positive,
            "negative": negative,
        }
        self._put_cached(ck, project_id, wm, payload)
        return sanitize_json_value(payload)

    def get_card_detail(self, card_key: str, flt: MetaFilter) -> dict[str, Any] | None:
        project_id = self.db.resolve_project_id(flt.project)
        if not project_id:
            return None
        wm = self.db.get_meta_watermark(project_id)
        ck = flt.canonical_key("card_detail", project_id, card_key)
        cached = self._get_cached(ck, project_id, wm)
        if cached is not None:
            return cached

        # Use min_games=0 internally so clicking any known card still shows its dossier
        eff_flt = MetaFilter(**{**asdict(flt), "min_games": 0})
        full = self._compute_full_state(eff_flt)
        internal = full.get("_internal")
        if not internal:
            return None
        ds = internal["ds"]
        cid = ds["key_to_id"].get(card_key)
        if cid is None or cid not in internal["stats"]:
            return None

        card_summary = internal["cards_by_id"].get(cid)
        if card_summary is None:
            return None
        st = internal["stats"][cid]
        syns = internal["card_synergies"][cid]

        # Mulligan stats
        mulligan = None
        if st["mull_offered"] > 0:
            tossed = st["mull_offered"] - st["mull_kept"]
            mulligan = {
                "offered": st["mull_offered"],
                "kept": st["mull_kept"],
                "keep_rate": st["mull_kept"] / st["mull_offered"],
                "kept_wr": _safe_div(st["mull_kept_wins"], st["mull_kept"]),
                "tossed_wr": _safe_div(st["mull_tossed_wins"], tossed),
            }

        # Lane context breakdown (if present)
        total_ctx_plays = sum(v[0] for v in st["lane_contexts"].values())
        lane_contexts = []
        for ctx_name, (cp, cw, cws, cwc) in sorted(
            st["lane_contexts"].items(), key=lambda kv: -kv[1][0]
        ):
            lane_contexts.append(
                {
                    "context": ctx_name,
                    "plays": cp,
                    "share": (cp / total_ctx_plays) if total_ctx_plays > 0 else 0.0,
                    "win_rate": _safe_div(cw, cp),
                    "critic_wpa": _safe_div(cws, cwc),
                }
            )

        # Turn curve
        total_turn_plays = sum(v[0] for v in st["turns"].values())
        turn_curve = []
        for t_num in sorted(st["turns"].keys()):
            tp, tw, tws, twc = st["turns"][t_num]
            turn_curve.append(
                {
                    "turn": t_num,
                    "label": "T13+" if t_num >= 13 else f"T{t_num}",
                    "plays": tp,
                    "share": (tp / total_turn_plays) if total_turn_plays > 0 else 0.0,
                    "win_rate": _safe_div(tw, tp),
                    "critic_wpa": _safe_div(tws, twc),
                }
            )

        # Top decks
        top_decks = []
        for dname, (dg, dw) in sorted(st["decks"].items(), key=lambda kv: -kv[1][0])[:10]:
            top_decks.append(
                {
                    "deck_name": dname,
                    "games": dg,
                    "share": (dg / st["games_in_deck"]) if st["games_in_deck"] > 0 else 0.0,
                    "win_rate": _safe_div(dw, dg),
                }
            )

        # Synergies & counters
        played_sorted = sorted(syns["played"], key=lambda x: x["lift"], reverse=True)
        lane_sorted = sorted(syns["same_lane"], key=lambda x: x["lift"], reverse=True)
        deck_sorted = sorted(syns["deck"], key=lambda x: x["lift"], reverse=True)
        counters_sorted = sorted(syns["counters"], key=lambda x: x["lift"])

        # Trend for this single card
        trend = self._compute_trends_from_ds(ds, [cid], bucket="day").get(card_key, [])

        payload = {
            "project_id": project_id,
            "kpis": full["kpis"],
            "card": card_summary,
            "funnel": {
                "in_deck": st["games_in_deck"],
                "opening_hand": st["oh_games"],
                "drawn": st["drawn_games"],
                "played": st["played_games"],
                "not_seen": st["gns_games"],
                "drawn_not_played": st["drawn_not_played_games"],
                "in_deck_wr": _safe_div(st["deck_wins"], st["games_in_deck"]),
                "oh_wr": card_summary["oh_wr"],
                "gih_wr": card_summary["gih_wr"],
                "played_wr": card_summary["played_wr"],
                "gns_wr": card_summary["gns_wr"],
                "mulligan": mulligan,
            },
            "seat_split": {
                "p1_games": st["p1_games"],
                "p1_wr": card_summary["p1_wr"],
                "p2_games": st["p2_games"],
                "p2_wr": card_summary["p2_wr"],
                "seat_delta": card_summary["seat_delta"],
            },
            "lane_contexts": lane_contexts,
            "turn_curve": turn_curve,
            "avg_rounds_when_played": _safe_div(
                st["rounds_when_played_sum"], st["played_games"]
            ),
            "synergies": {
                "played": [s for s in played_sorted if s["lift"] > 0][:10],
                "same_lane": [s for s in lane_sorted if s["lift"] > 0][:10],
                "deck": [s for s in deck_sorted if s["lift"] > 0][:10],
                "anti": [s for s in sorted(syns["played"] or syns["deck"], key=lambda x: x["lift"]) if s["lift"] < 0][:8],
            },
            "counters": {
                "worst_against": [c for c in counters_sorted if c["lift"] < 0][:8],
                "best_against": [c for c in reversed(counters_sorted) if c["lift"] > 0][:8],
            },
            "top_decks": top_decks,
            "trend": trend,
        }
        self._put_cached(ck, project_id, wm, payload)
        return sanitize_json_value(payload)

    def _compute_trends_from_ds(
        self, ds: dict[str, Any], card_ids: list[int], bucket: str = "day"
    ) -> dict[str, list[dict[str, Any]]]:
        games = ds["games"]
        cards_by_gp = ds["cards_by_gp"]
        id_to_key = ds["id_to_key"]
        target_set = set(card_ids)
        if not target_set or not games:
            return {}

        bucket_pg: dict[int, int] = defaultdict(int)
        # (bucket_id, cid) -> [deck_n, drawn_n, drawn_w, gns_n, gns_w, played_n, played_w]
        agg: dict[tuple[int, int], list[float]] = {}

        for g in games:
            db = int(g["day_bucket"])
            b_key = (db // 7) * 7 if bucket == "week" else db
            bucket_pg[b_key] += len(g["players"])
            for p in g["players"]:
                sc = float(p["outcome_score"])
                cmap = cards_by_gp.get(int(p["id"]), {})
                for cid in target_set:
                    gc = cmap.get(cid)
                    if gc is None:
                        continue
                    rec = agg.get((b_key, cid))
                    if rec is None:
                        rec = [0.0] * 7
                        agg[(b_key, cid)] = rec
                    rec[0] += 1.0
                    if gc["drawn_count"] > 0:
                        rec[1] += 1.0
                        rec[2] += sc
                    else:
                        rec[3] += 1.0
                        rec[4] += sc
                    if gc["played_count"] > 0:
                        rec[5] += 1.0
                        rec[6] += sc

        out: dict[str, list[dict[str, Any]]] = {
            id_to_key[cid]: [] for cid in card_ids if cid in id_to_key
        }
        sorted_buckets = sorted(bucket_pg.keys())
        for b_key in sorted_buckets:
            tot_pg = bucket_pg[b_key]
            ts_ms = b_key * 86_400_000
            for cid in card_ids:
                ckey = id_to_key.get(cid)
                if not ckey:
                    continue
                rec = agg.get((b_key, cid))
                if not rec:
                    continue
                dk_n, dr_n, dr_w, gns_n, gns_w, pl_n, pl_w = rec
                gih_wr = _safe_div(dr_w, int(dr_n))
                gns_wr = _safe_div(gns_w, int(gns_n))
                pl_wr = _safe_div(pl_w, int(pl_n))
                raw_wpa = (gih_wr - gns_wr) if (gih_wr is not None and gns_wr is not None) else None
                n_eff = min(int(dr_n), int(gns_n))
                wpa = (
                    (n_eff / (n_eff + _WPA_SHRINK_K)) * raw_wpa
                    if raw_wpa is not None
                    else None
                )
                out[ckey].append(
                    {
                        "day_bucket": b_key,
                        "ts_ms": ts_ms,
                        "games": int(dk_n),
                        "inclusion_rate": dk_n / tot_pg if tot_pg > 0 else 0.0,
                        "gih_wr": gih_wr,
                        "drawn_wpa": wpa,
                        "played_wr": pl_wr,
                    }
                )
        return out

    def get_trends(
        self,
        flt: MetaFilter,
        *,
        cards: list[str] | None = None,
        limit_cards: int = 8,
        bucket: str = "day",
    ) -> dict[str, Any]:
        project_id = self.db.resolve_project_id(flt.project)
        if not project_id:
            return {"project_id": None, "bucket": bucket, "trends": {}}
        wm = self.db.get_meta_watermark(project_id)
        card_sig = ",".join(cards or [])
        ck = flt.canonical_key("trends", project_id, f"{card_sig}:{limit_cards}:{bucket}")
        cached = self._get_cached(ck, project_id, wm)
        if cached is not None:
            return cached

        ds = self._load_filtered_dataset(flt, project_id, include_plays=False)
        selected_keys = [c for c in (cards or []) if c in ds["key_to_id"]]
        if not selected_keys:
            overview = self.get_overview(flt)
            selected_keys = [c["card_key"] for c in overview["cards"][:limit_cards]]

        cids = [ds["key_to_id"][k] for k in selected_keys if k in ds["key_to_id"]]
        trends = self._compute_trends_from_ds(ds, cids, bucket=bucket)
        payload = {"project_id": project_id, "bucket": bucket, "trends": trends}
        self._put_cached(ck, project_id, wm, payload)
        return sanitize_json_value(payload)

    def get_patch_diff(
        self,
        base_flt: MetaFilter,
        manifest_a: str,
        manifest_b: str,
    ) -> dict[str, Any]:
        project_id = self.db.resolve_project_id(base_flt.project)
        if not project_id:
            return {"project_id": None, "manifest_a": manifest_a, "manifest_b": manifest_b, "diffs": []}
        wm = self.db.get_meta_watermark(project_id)
        ck = base_flt.canonical_key("patch_diff", project_id, f"{manifest_a}->{manifest_b}")
        cached = self._get_cached(ck, project_id, wm)
        if cached is not None:
            return cached

        flt_a = MetaFilter(**{**asdict(base_flt), "manifest_hash": manifest_a, "min_games": 0})
        flt_b = MetaFilter(**{**asdict(base_flt), "manifest_hash": manifest_b, "min_games": 0})
        ov_a = self.get_overview(flt_a)
        ov_b = self.get_overview(flt_b)

        map_a = {c["card_key"]: c for c in ov_a["cards"]}
        map_b = {c["card_key"]: c for c in ov_b["cards"]}
        cat_a = {v["card_key"]: v for v in self._load_catalog_map(project_id, manifest_a).values()}
        cat_b = {v["card_key"]: v for v in self._load_catalog_map(project_id, manifest_b).values()}

        all_keys = sorted(set(map_a.keys()) | set(map_b.keys()) | set(cat_a.keys()) | set(cat_b.keys()))
        diffs: list[dict[str, Any]] = []

        def _delta(vb: float | None, va: float | None) -> float | None:
            if vb is None or va is None:
                return None
            return vb - va

        for k in all_keys:
            ca = map_a.get(k)
            cb = map_b.get(k)
            meta_a = cat_a.get(k)
            meta_b = cat_b.get(k)

            catalog_changes: list[dict[str, Any]] = []
            if meta_a and meta_b:
                for field in ("mana_cost", "attack", "health", "faction", "card_type", "rarity", "text"):
                    if meta_a.get(field) != meta_b.get(field):
                        catalog_changes.append(
                            {"field": field, "old": meta_a.get(field), "new": meta_b.get(field)}
                        )

            inc_a = ca["inclusion_rate"] if ca else 0.0
            inc_b = cb["inclusion_rate"] if cb else 0.0
            diffs.append(
                {
                    "card_key": k,
                    "display_name": (cb or ca or {}).get("display_name")
                    or (meta_b or meta_a or {}).get("display_name")
                    or k,
                    "catalog_changes": catalog_changes,
                    "games_a": ca["games_in_deck"] if ca else 0,
                    "games_b": cb["games_in_deck"] if cb else 0,
                    "inclusion_a": inc_a,
                    "inclusion_b": inc_b,
                    "delta_inclusion": inc_b - inc_a,
                    "gih_wr_a": ca["gih_wr"] if ca else None,
                    "gih_wr_b": cb["gih_wr"] if cb else None,
                    "delta_gih_wr": _delta(cb["gih_wr"] if cb else None, ca["gih_wr"] if ca else None),
                    "drawn_wpa_a": ca["drawn_wpa"] if ca else None,
                    "drawn_wpa_b": cb["drawn_wpa"] if cb else None,
                    "delta_drawn_wpa": _delta(cb["drawn_wpa"] if cb else None, ca["drawn_wpa"] if ca else None),
                    "impact_a": ca["isolated_impact"] if ca else None,
                    "impact_b": cb["isolated_impact"] if cb else None,
                    "delta_impact": _delta(cb["isolated_impact"] if cb else None, ca["isolated_impact"] if ca else None),
                    "meta_war_a": ca["meta_war"] if ca else None,
                    "meta_war_b": cb["meta_war"] if cb else None,
                    "delta_meta_war": _delta(cb["meta_war"] if cb else None, ca["meta_war"] if ca else None),
                }
            )

        diffs.sort(
            key=lambda d: (
                -(abs(d["delta_drawn_wpa"]) if d["delta_drawn_wpa"] is not None else 0.0),
                -abs(d["delta_inclusion"]),
            )
        )
        payload = {
            "project_id": project_id,
            "manifest_a": manifest_a,
            "manifest_b": manifest_b,
            "games_a": ov_a["kpis"]["games"],
            "games_b": ov_b["kpis"]["games"],
            "diffs": diffs,
        }
        self._put_cached(ck, project_id, wm, payload)
        return sanitize_json_value(payload)
