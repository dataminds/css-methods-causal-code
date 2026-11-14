# -*- coding: utf-8 -*-
"""1장 · 논문 한 편의 해부 — 파이썬 엔진."""
import numpy as np

from common import calc, cohen_d, load

E = lambda: load("exp")          # 실험판 정제본 366명

calc("ch01.paper_mil_diff")(lambda: 5.14 - 4.83)


def _groups():
    d = E()
    return d[d.cond == 1].mil.to_numpy(float), d[d.cond == 0].mil.to_numpy(float)


def _pooled_sd():
    a, b = _groups()
    return np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))


calc("ch01.exp_mil_diff")(lambda: _groups()[0].mean() - _groups()[1].mean())
calc("ch01.exp_mil_pooled_sd")(_pooled_sd)
calc("ch01.exp_mil_d")(lambda: cohen_d(*_groups()))
