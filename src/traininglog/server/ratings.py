"""Rating math engines: Glicko-2 (1:1 cardgame parity), Elo, and OpenSkill (Plackett-Luce)."""

from __future__ import annotations

import math
from dataclasses import dataclass

GLICKO2_SCALE = 173.7178


@dataclass
class Glicko2Rating:
    """Player rating state in Glicko-2."""

    mu: float = 1500.0
    rd: float = 350.0
    vol: float = 0.06

    @property
    def r(self) -> float:
        return (self.mu - 1500.0) / GLICKO2_SCALE

    @property
    def phi(self) -> float:
        return self.rd / GLICKO2_SCALE

    @classmethod
    def from_glicko2_scale(cls, r: float, phi: float, vol: float) -> Glicko2Rating:
        return cls(
            mu=r * GLICKO2_SCALE + 1500.0,
            rd=phi * GLICKO2_SCALE,
            vol=vol,
        )

    def confidence_interval_95(self) -> tuple[float, float]:
        return (self.mu - 2.0 * self.rd, self.mu + 2.0 * self.rd)


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + (3.0 * phi**2) / (math.pi**2))


def _E(r: float, r_j: float, phi_j: float) -> float:
    exponent = max(-60.0, min(60.0, -_g(phi_j) * (r - r_j)))
    return 1.0 / (1.0 + math.exp(exponent))


def _update_glicko2_player(
    player: Glicko2Rating,
    opponents: list[tuple[Glicko2Rating, float]],
    tau: float = 0.5,
    epsilon: float = 0.000001,
) -> Glicko2Rating:
    if not opponents:
        phi_prime = math.sqrt(player.phi**2 + player.vol**2)
        return Glicko2Rating.from_glicko2_scale(player.r, phi_prime, player.vol)

    r = player.r
    phi = player.phi
    sigma = player.vol

    v_inv = 0.0
    delta_sum = 0.0
    for opp, s in opponents:
        g_j = _g(opp.phi)
        e_j = _E(r, opp.r, opp.phi)
        v_inv += (g_j**2) * e_j * (1.0 - e_j)
        delta_sum += g_j * (s - e_j)

    v = 1.0 / max(v_inv, 1e-12)
    delta = v * delta_sum
    a = math.log(max(sigma**2, 1e-12))

    def f(x: float) -> float:
        e_x = math.exp(max(-60.0, min(60.0, x)))
        d_sq = delta**2
        phi_sq = phi**2
        num1 = e_x * (d_sq - phi_sq - v - e_x)
        den1 = 2.0 * ((phi_sq + v + e_x) ** 2)
        return (num1 / max(den1, 1e-18)) - ((x - a) / (tau**2))

    A = a
    if delta**2 > (phi**2 + v):
        B = math.log(delta**2 - phi**2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0 and k < 100:
            k += 1
        B = a - k * tau

    f_A = f(A)
    f_B = f(B)
    iters = 0
    while abs(B - A) > epsilon and iters < 200:
        iters += 1
        denom = f_B - f_A
        if abs(denom) < 1e-18:
            break
        C = A + (A - B) * f_A / denom
        f_C = f(C)
        if f_C * f_B <= 0:
            A = B
            f_A = f_B
        else:
            f_A = f_A / 2.0
        B = C
        f_B = f_C

    sigma_prime = math.exp(A / 2.0)
    phi_star = math.sqrt(phi**2 + sigma_prime**2)
    phi_prime = 1.0 / math.sqrt((1.0 / (phi_star**2)) + (1.0 / v))
    r_prime = r + (phi_prime**2) * delta_sum
    return Glicko2Rating.from_glicko2_scale(r_prime, phi_prime, sigma_prime)


def rate_glicko2_match(
    rating_a: Glicko2Rating,
    rating_b: Glicko2Rating,
    score_a: float,
    tau: float = 0.5,
    epsilon: float = 0.000001,
) -> tuple[Glicko2Rating, Glicko2Rating]:
    """Update Glicko-2 ratings for two players after a match with score_a in [0, 1]."""
    s_a = max(0.0, min(1.0, float(score_a)))
    new_a = _update_glicko2_player(rating_a, [(rating_b, s_a)], tau, epsilon)
    new_b = _update_glicko2_player(rating_b, [(rating_a, 1.0 - s_a)], tau, epsilon)
    return new_a, new_b


def rate_elo_match(
    elo_a: float,
    elo_b: float,
    score_a: float,
    k: float = 32.0,
) -> tuple[float, float]:
    """Update standard Elo ratings for two players."""
    s_a = max(0.0, min(1.0, float(score_a)))
    exp_arg = max(-60.0, min(60.0, (elo_b - elo_a) / 400.0))
    expected_a = 1.0 / (1.0 + 10.0**exp_arg)
    delta = k * (s_a - expected_a)
    return elo_a + delta, elo_b - delta


@dataclass
class OpenSkillRating:
    """Player rating state in OpenSkill (Plackett-Luce model)."""

    mu: float = 25.0
    sigma: float = 25.0 / 3.0

    @property
    def ordinal(self) -> float:
        return self.mu - 3.0 * self.sigma


def rate_openskill_match(
    teams: list[list[OpenSkillRating]],
    ranks: list[int],
    *,
    beta: float = 25.0 / 6.0,
    tau: float = 25.0 / 300.0,
) -> list[list[OpenSkillRating]]:
    """Update OpenSkill Plackett-Luce ratings for N teams/players (lower rank = better)."""
    if len(teams) != len(ranks) or len(teams) < 2:
        return [
            [OpenSkillRating(mu=p.mu, sigma=p.sigma) for p in team]
            for team in teams
        ]

    team_mu: list[float] = []
    team_sigma_sq: list[float] = []
    beta_sq = beta**2
    tau_sq = tau**2

    for team in teams:
        mu_sum = sum(p.mu for p in team)
        sig_sq_sum = sum((p.sigma**2 + tau_sq) for p in team)
        team_mu.append(mu_sum)
        team_sigma_sq.append(max(1e-6, sig_sq_sum))

    c = math.sqrt(sum(s_sq + beta_sq for s_sq in team_sigma_sq))
    scaled_mu = [m / c for m in team_mu]
    max_scaled = max(scaled_mu)
    exp_mu = [math.exp(max(-60.0, min(0.0, sm - max_scaled))) for sm in scaled_mu]

    unique_ranks = sorted(set(ranks))
    rank_totals: dict[int, float] = {}
    rank_counts: dict[int, int] = {}
    for r in unique_ranks:
        rank_totals[r] = sum(exp_mu[k] for k, rk in enumerate(ranks) if rk >= r)
        rank_counts[r] = sum(1 for rk in ranks if rk == r)

    updated_teams: list[list[OpenSkillRating]] = []
    for i, team in enumerate(teams):
        r_i = ranks[i]
        s_sq_i = team_sigma_sq[i]
        gamma_i = math.sqrt(s_sq_i) / c

        omega_sum = 0.0
        delta_sum = 0.0
        for r in unique_ranks:
            if r > r_i:
                continue
            denom = max(rank_totals[r], 1e-18)
            pi_ir = exp_mu[i] / denom
            if r == r_i:
                omega_sum += (1.0 / rank_counts[r]) - pi_ir
            else:
                omega_sum -= pi_ir
            delta_sum += gamma_i * (s_sq_i / (c**2)) * pi_ir * (1.0 - pi_ir)

        omega = omega_sum * (s_sq_i / c)

        new_team: list[OpenSkillRating] = []
        for p in team:
            p_var = p.sigma**2 + tau_sq
            weight = p_var / s_sq_i
            new_mu = p.mu + weight * omega
            sigma_factor = max(1e-4, 1.0 - weight * delta_sum)
            new_sigma = max(0.01, math.sqrt(p_var * sigma_factor))
            new_team.append(OpenSkillRating(mu=new_mu, sigma=new_sigma))
        updated_teams.append(new_team)

    return updated_teams
