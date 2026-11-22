# -*- coding: utf-8 -*-
"""B4 · 검정이 기대는 가정과 순위 검정 — 파이썬 엔진.

2026-11-15 14장에서 갈라 나온 단위. 14장에서 옮긴 대조 줄은 계산 이름 ch14.* 를 그대로 쓰고
(py/units/ch14.py), 이 파일은 B4 에서 새로 쓴 문장의 수를 계산한다.
2026-11-22 §B4.1 보강 = 정규성이 검정력에 주는 대가 (네 세계 · t검정 대 맨-휘트니) ·
샤피로-윌크 검정의 기각률 (세 분포 × 표본 20 · 2,000). 씨앗 73 · 세계마다 새 난수열."""
import numpy as np
from scipy import stats

from common import calc

_c: dict = {}
R_SIM = 2000          # 세계를 짓는 횟수
N_POW = 30            # 검정력 세계의 집단당 인원
SHIFT = 0.6           # 한 집단에만 더하는 참 효과


def _contam(g, size):
    """오염 정규 = 열에 아홉은 표준정규, 열에 하나는 표준편차 5 인 정규 (가끔 멀리 튀는 값)."""
    z = g.normal(0, 1, size)
    far = g.random(size) < .1
    return np.where(far, 5 * z, z)


_WORLDS = {
    "norm": lambda g, s: g.normal(0, 1, s),
    "t10": lambda g, s: g.standard_t(10, s),
    "t3": lambda g, s: g.standard_t(3, s),
    "cn": _contam,
}


# ── §B4.1 검정력 = 집단당 30명 · 한 집단에 0.6 · 2,000번 ; 세계마다 default_rng(73) ──
def _power(world):
    key = ("pow", world)
    if key not in _c:
        g = np.random.default_rng(73)
        f = _WORLDS[world]
        U = f(g, (R_SIM, N_POW)) + SHIFT      # 효과가 있는 집단 2,000 개를 먼저, 이어서 비교 집단 2,000 개
        V = f(g, (R_SIM, N_POW))
        pt = stats.ttest_ind(U, V, axis=1).pvalue
        pm = stats.mannwhitneyu(U, V, axis=1).pvalue      # scipy 기본 = 정규 근사 · 연속성 수정
        _c[key] = (np.mean(pt < .05), np.mean(pm < .05))
    return _c[key]


for _w in _WORLDS:
    calc(f"b4.pow_{_w}_t")(lambda w=_w: _power(w)[0])
    calc(f"b4.pow_{_w}_mw")(lambda w=_w: _power(w)[1])


# ── §B4.1 샤피로-윌크 기각률 = 표본 20 · 2,000 · 2,000번 ; 칸마다 default_rng(73) ──
def _sw(world, n):
    key = ("sw", world, n)
    if key not in _c:
        g = np.random.default_rng(73)
        X = _WORLDS[world](g, (R_SIM, n))
        _c[key] = np.mean([stats.shapiro(x).pvalue < .05 for x in X])
    return _c[key]


for _w in ("norm", "t10", "t3"):
    for _n in (20, 2000):
        calc(f"b4.sw_{_w}_{_n}")(lambda w=_w, n=_n: _sw(w, n))
