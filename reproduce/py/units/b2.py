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
