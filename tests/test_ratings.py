"""Unit tests for Glicko-2 (1:1 cardgame parity), Elo, and OpenSkill rating math."""

from __future__ import annotations

import math

import pytest

from traininglog.server.ratings import (
    GLICKO2_SCALE,
    Glicko2Rating,
    OpenSkillRating,
    rate_elo_match,
    rate_glicko2_match,
    rate_openskill_match,
)


def _reference_glickman_example() -> tuple[Glicko2Rating, Glicko2Rating]:
    """Reference single-opponent Glicko-2 check matching cardgame/agents/glicko2.py."""
    # Player A: 1500 ± 200, Player B: 1400 ± 30, score_a = 0.75 (750/1000 games)
    a = Glicko2Rating(mu=1500.0, rd=200.0, vol=0.06)
    b = Glicko2Rating(mu=1400.0, rd=30.0, vol=0.06)
    return rate_glicko2_match(a, b, 0.75)


def test_glicko2_exact_reference_parity() -> None:
    # Verify scaling constants and confidence interval
    r0 = Glicko2Rating()
    assert math.isclose(r0.r, 0.0)
    assert math.isclose(r0.phi, 350.0 / GLICKO2_SCALE, rel_tol=1e-9)
    lo, hi = r0.confidence_interval_95()
    assert math.isclose(lo, 800.0) and math.isclose(hi, 2200.0)

    # Exact manual Illinois check against Glickman (2012) single match
    a_after, b_after = _reference_glickman_example()
    assert a_after.mu > 1500.0
    assert a_after.rd < 200.0
    assert b_after.mu < 1400.0
    # Re-verifying symmetry when equal players draw 50-50
    ea, eb = rate_glicko2_match(Glicko2Rating(), Glicko2Rating(), 0.5)
    assert math.isclose(ea.mu, 1500.0, abs_tol=1e-9)
    assert math.isclose(eb.mu, 1500.0, abs_tol=1e-9)
    assert math.isclose(ea.rd, eb.rd, abs_tol=1e-9)
    assert ea.rd < 350.0

    # Verify exact numerical parity against inline cardgame/agents/glicko2.py algorithm
    tau = 0.5
    eps = 0.000001
    p_a = Glicko2Rating(mu=1580.0, rd=110.0, vol=0.059)
    p_b = Glicko2Rating(mu=1490.0, rd=140.0, vol=0.061)
    score_a = 0.634

    def ref_single(player: Glicko2Rating, opp: Glicko2Rating, s: float) -> Glicko2Rating:
        g_j = 1.0 / math.sqrt(1.0 + (3.0 * opp.phi**2) / (math.pi**2))
        e_j = 1.0 / (1.0 + math.exp(-g_j * (player.r - opp.r)))
        v = 1.0 / ((g_j**2) * e_j * (1.0 - e_j))
        delta = v * g_j * (s - e_j)
        a = math.log(player.vol**2)

        def f(x: float) -> float:
            e_x = math.exp(x)
            return (e_x * (delta**2 - player.phi**2 - v - e_x)) / (
                2.0 * ((player.phi**2 + v + e_x) ** 2)
            ) - (x - a) / (tau**2)

        A = a
        if delta**2 > (player.phi**2 + v):
            B = math.log(delta**2 - player.phi**2 - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
            B = a - k * tau
        f_A, f_B = f(A), f(B)
        while abs(B - A) > eps:
            C = A + (A - B) * f_A / (f_B - f_A)
            f_C = f(C)
            if f_C * f_B <= 0:
                A, f_A = B, f_B
            else:
                f_A /= 2.0
            B, f_B = C, f_C
        sigma_p = math.exp(A / 2.0)
        phi_star = math.sqrt(player.phi**2 + sigma_p**2)
        phi_p = 1.0 / math.sqrt((1.0 / (phi_star**2)) + (1.0 / v))
        r_p = player.r + (phi_p**2) * g_j * (s - e_j)
        return Glicko2Rating.from_glicko2_scale(r_p, phi_p, sigma_p)

    got_a, got_b = rate_glicko2_match(p_a, p_b, score_a)
    ref_a = ref_single(p_a, p_b, score_a)
    ref_b = ref_single(p_b, p_a, 1.0 - score_a)
    assert math.isclose(got_a.mu, ref_a.mu, abs_tol=1e-6)
    assert math.isclose(got_a.rd, ref_a.rd, abs_tol=1e-6)
    assert math.isclose(got_a.vol, ref_a.vol, abs_tol=1e-6)
    assert math.isclose(got_b.mu, ref_b.mu, abs_tol=1e-6)
    assert math.isclose(got_b.rd, ref_b.rd, abs_tol=1e-6)


def test_elo_symmetry_and_upset() -> None:
    a, b = rate_elo_match(1500.0, 1500.0, 1.0, k=32.0)
    assert math.isclose(a, 1516.0)
    assert math.isclose(b, 1484.0)
    assert math.isclose(a + b, 3000.0)

    # Underdog upset yields larger gain than favorite winning
    u_a, _ = rate_elo_match(1200.0, 1600.0, 1.0, k=32.0)
    f_a, _ = rate_elo_match(1600.0, 1200.0, 1.0, k=32.0)
    assert (u_a - 1200.0) > (f_a - 1600.0)


def test_openskill_ffa_ties_teams_and_overflow_safety() -> None:
    # 4-player FFA strictly ranks mu updates
    p = [OpenSkillRating() for _ in range(4)]
    out = rate_openskill_match([[x] for x in p], [1, 2, 3, 4])
    mus = [team[0].mu for team in out]
    assert mus[0] > mus[1] > mus[2] > mus[3]
    assert mus[0] > 25.0 and mus[3] < 25.0
    assert all(team[0].sigma < 25.0 / 3.0 for team in out)

    # Exact tie between equal players produces zero mu shift
    tie_out = rate_openskill_match([[OpenSkillRating()], [OpenSkillRating()]], [1, 1])
    assert math.isclose(tie_out[0][0].mu, 25.0, abs_tol=1e-9)
    assert math.isclose(tie_out[1][0].mu, 25.0, abs_tol=1e-9)

    # 2v2 team match: winning team members gain, losing team members drop
    t_out = rate_openskill_match(
        [
            [OpenSkillRating(mu=25.0), OpenSkillRating(mu=20.0)],
            [OpenSkillRating(mu=25.0), OpenSkillRating(mu=25.0)],
        ],
        [1, 2],
    )
    assert t_out[0][0].mu > 25.0 and t_out[0][1].mu > 20.0
    assert t_out[1][0].mu < 25.0 and t_out[1][1].mu < 25.0

    # Extreme rating gap must never overflow exp()
    ext = rate_openskill_match(
        [[OpenSkillRating(mu=5000.0, sigma=0.5)], [OpenSkillRating(mu=-5000.0, sigma=0.5)]],
        [1, 2],
    )
    assert math.isfinite(ext[0][0].mu) and math.isfinite(ext[1][0].mu)
