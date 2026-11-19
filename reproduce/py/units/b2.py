# -*- coding: utf-8 -*-
"""B2 · 세 집단 이상 비교 — 파이썬 엔진 (자료 = journey_fac · journey_exp 정제본)."""
from math import comb

import numpy as np
from scipy import stats

from common import calc, load

FAC = lambda: load("fac")
EXP = lambda: load("exp")


def grp():
    f = FAC()
    return [f[(f.elem == e) & (f.frame == fr)].mil_t2.values for e in (0, 1) for fr in (0, 1)]


def pair_ps():
    g = grp()
    return [stats.ttest_ind(g[i], g[j]).pvalue for i in range(4) for j in range(i + 1, 4)]


def two():
    e = EXP()
    return e[e.cond == 1].mil.values, e[e.cond == 0].mil.values


# §B2.0 · 실습 3 곱셈과 본페로니 문턱 (정해진 산술)
calc("b2.p_none6")(lambda: .95 ** 6)
calc("b2.p_any6")(lambda: 1 - .95 ** 6)
calc("b2.p_any3")(lambda: 1 - .95 ** 3)
calc("b2.p_any15")(lambda: 1 - .95 ** 15)
calc("b2.bonf5")(lambda: .05 / 5)
calc("b2.pairs3")(lambda: comb(3, 2))
calc("b2.pairs6")(lambda: comb(6, 2))
calc("b2.bonf3")(lambda: .05 / comb(3, 2))
calc("b2.bonf15")(lambda: .05 / comb(6, 2))
# §B2.1 두 집단 (실험판) F = t²
calc("b2.t_exp")(lambda: stats.ttest_ind(*two()).statistic)
calc("b2.t2_exp")(lambda: stats.ttest_ind(*two()).statistic ** 2)
calc("b2.F_exp")(lambda: stats.f_oneway(*two()).statistic)
calc("b2.df2_exp")(lambda: len(two()[0]) + len(two()[1]) - 2)
calc("b2.exp_group_n")(lambda: len(two()[0]))
calc("b2.exp_diff")(lambda: two()[0].mean() - two()[1].mean())
# §B2.1 요인판 네 칸
calc("b2.F4")(lambda: stats.f_oneway(*grp()).statistic)
calc("b2.p4")(lambda: stats.f_oneway(*grp()).pvalue)
calc("b2.df1_4")(lambda: len(grp()) - 1)
calc("b2.df2_4")(lambda: sum(len(x) for x in grp()) - len(grp()))
calc("b2.fac_n")(lambda: sum(len(x) for x in grp()))
calc("b2.k4")(lambda: len(grp()))
calc("b2.cell_min_mean")(lambda: min(x.mean() for x in grp()))
calc("b2.cell_max_mean")(lambda: max(x.mean() for x in grp()))
calc("b2.cell_range")(lambda: max(x.mean() for x in grp()) - min(x.mean() for x in grp()))
calc("b2.cell_n_min")(lambda: min(len(x) for x in grp()))
calc("b2.cell_n_max")(lambda: max(len(x) for x in grp()))
# §B2.2 쌍별 여섯
calc("b2.n_pairs")(lambda: len(pair_ps()))
calc("b2.n_sig")(lambda: sum(q < .05 for q in pair_ps()))
calc("b2.min_p")(lambda: min(pair_ps()))
calc("b2.bonf6")(lambda: .05 / len(pair_ps()))
calc("b2.n_sig_bonf")(lambda: sum(q < .05 / len(pair_ps()) for q in pair_ps()))


# §B2.1 「논문의 분산분석 표 읽기」 = 네 칸 일원배치의 제곱합 분해
def _aov():
    g = [FAC()[(FAC().elem == e) & (FAC().frame == f)].mil_t2.values for e in (0, 1) for f in (0, 1)]
    a = np.concatenate(g)
    ssb = sum(len(x) * (x.mean() - a.mean()) ** 2 for x in g)
    ssw = sum(((x - x.mean()) ** 2).sum() for x in g)
    dfb, dfw = len(g) - 1, len(a) - len(g)
    return {"ssb": ssb, "ssw": ssw, "sst": ((a - a.mean()) ** 2).sum(), "dfb": dfb, "dfw": dfw, "dft": len(a) - 1,
            "msb": ssb / dfb, "msw": ssw / dfw, "F": (ssb / dfb) / (ssw / dfw), "eta2": ssb / (ssb + ssw), "n": len(a)}


for _k in ("ssb", "ssw", "sst", "dfb", "dfw", "dft", "msb", "msw", "F", "eta2", "n"):
    calc(f"b2.aov_{_k}")(lambda k=_k: _aov()[k])
calc("b2.aov_eta2_pct")(lambda: _aov()["eta2"] * 100)
calc("b2.aov_p")(lambda: stats.f.sf(_aov()["F"], _aov()["dfb"], _aov()["dfw"]))
calc("b2.aov_msb_over_msw_rounded")(lambda: round(_aov()["msb"], 2) / round(_aov()["msw"], 2))


# §B2.0 · 실습 3 (2026-11-19 3부 점검 반영) = 효과가 없는 세계에서 쌍별 t검정을 보정 없이 했을 때
# 하나라도 위양성이 나오는 비율. 집단이 쌍을 나눠 쓰므로 독립 공식(1 − .95^m)보다 낮다.
def _fwer(k, n=108, reps=5000, seed=73):
    g = np.random.default_rng(seed)
    x = g.normal(0, 1, (reps, k, n))
    m, v = x.mean(axis=2), x.var(axis=2, ddof=1)
    hit = np.zeros(reps, bool)
    for i in range(k):
        for j in range(i + 1, k):
            t = (m[:, i] - m[:, j]) / np.sqrt((v[:, i] + v[:, j]) / n)
            hit |= 2 * stats.t.sf(np.abs(t), 2 * n - 2) < .05
    return hit.mean()


calc("b2.fwer_sim4")(lambda: _fwer(4))
calc("b2.fwer_sim6")(lambda: _fwer(6))
