# -*- coding: utf-8 -*-
"""시나리오 S7 · 학위 논문 한 편 — 파이썬 엔진 (T3 표본 크기 역산 ; 나머지 수는 S1 · S2 · S3 계산 키로 회수)."""
import numpy as np
from scipy import stats

from common import calc, load


def _power(n, d, alpha=.05):
    """집단당 n명 · 두 집단 t검정 · 양측 · 비중심 t."""
    df = 2 * n - 2
    nc = d * np.sqrt(n / 2)
    c = stats.t.ppf(1 - alpha / 2, df)
    return float(1 - stats.nct.cdf(c, df, nc) + stats.nct.cdf(-c, df, nc))


def _need(d, target=.80):
    n = 2
    while _power(n, d) < target:
        n += 1
    return n


calc("s7.need_d40")(lambda: _need(0.40))
calc("s7.need_d22")(lambda: _need(0.22))
calc("s7.pow_100_d22")(lambda: _power(100, 0.22))
calc("s7.pow_190_d22")(lambda: _power(190, 0.22))
calc("s7.pow_183_d22")(lambda: _power(183, 0.22))

# T5 표본 흐름 (제외 인원)
calc("s7.svy_excl")(lambda: len(load("svy", clean=False)) - len(load("svy")))
calc("s7.fac_raw")(lambda: len(load("fac", clean=False)))
calc("s7.fac_excl")(lambda: len(load("fac", clean=False)) - len(load("fac")))
calc("s7.fac_cell_min")(lambda: int(load("fac").groupby(["elem", "frame"]).size().min()))
calc("s7.fac_cell_max")(lambda: int(load("fac").groupby(["elem", "frame"]).size().max()))
