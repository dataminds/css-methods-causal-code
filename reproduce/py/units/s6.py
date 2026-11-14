# -*- coding: utf-8 -*-
"""S6 · 내용분석 — 파이썬 엔진. 책 코드와 같은 정의(가중치 W = 1 − (거리/4)² 의 가중 카파)."""
import numpy as np
import pandas as pd
from scipy import stats

from common import calc, load, ols

CD = lambda: load("coding")
TAGS = ["p", "s", "q", "a", "c", "t", "l"]
CATS = list(range(-2, 3))
_K = len(CATS)
_IDX = {c: i for i, c in enumerate(CATS)}


def _kappa(x, y, weighted=True):
    O = np.zeros((_K, _K))
    for a, b in zip(x, y):
        O[_IDX[a], _IDX[b]] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    if weighted:
        W = np.array([[1 - ((CATS[i] - CATS[j]) / (_K - 1)) ** 2 for j in range(_K)] for i in range(_K)])
    else:
        W = np.eye(_K)
    return (np.sum(W * O) - np.sum(W * E)) / (1 - np.sum(W * E))


def _k(t):
    return _kappa(CD()[f"coder1_{t}"].values, CD()[f"coder2_{t}"].values)


def _present(t):
    m = (CD()[f"coder1_{t}"] + CD()[f"coder2_{t}"]) / 2
    return float((m > 0).mean())


def _exact(t):
    return float((CD()[f"coder1_{t}"] == CD()[f"coder2_{t}"]).mean())


def _within1(t):
    return float((np.abs(CD()[f"coder1_{t}"] - CD()[f"coder2_{t}"]) <= 1).mean())


def _totals(cols):
    c1 = CD()[[f"coder1_{t}" for t in cols]].mean(axis=1)
    c2 = CD()[[f"coder2_{t}" for t in cols]].mean(axis=1)
    b, se, p, r2 = ols(CD().mil, [(c1 + c2) / 2])
    return float(np.corrcoef(c1, c2)[0, 1]), float(b[1]), float(p[1]), float(r2)


SIX = [t for t in TAGS if t != "a"]


def _ct():
    cd = CD()
    tv = ((cd.coder1_t + cd.coder2_t) / 2 > 0).astype(int)
    lv = ((cd.coder1_l + cd.coder2_l) / 2 > 0).astype(int)
    return tv, lv, pd.crosstab(tv, lv)


calc("s6.n_stories")(lambda: len(CD()))
for _t in TAGS:
    calc(f"s6.wkappa_{_t}")(lambda t=_t: _k(t))
calc("s6.kappa_unweighted_p")(lambda: _kappa(CD().coder1_p.values, CD().coder2_p.values, weighted=False))
calc("s6.exact_pct_p")(lambda: _exact("p") * 100)
calc("s6.exact_pct_a")(lambda: _exact("a") * 100)
calc("s6.within1_pct_a")(lambda: _within1("a") * 100)
calc("s6.exact_pct_min")(lambda: min(_exact(t) for t in TAGS) * 100)
calc("s6.exact_pct_max")(lambda: max(_exact(t) for t in TAGS) * 100)
calc("s6.within1_pct_min")(lambda: min(_within1(t) for t in TAGS) * 100)
calc("s6.within1_pct_max")(lambda: max(_within1(t) for t in TAGS) * 100)
calc("s6.wkappa_min")(lambda: min(_k(t) for t in TAGS))
calc("s6.wkappa_max")(lambda: max(_k(t) for t in TAGS))
calc("s6.present_pct_min")(lambda: min(_present(t) for t in TAGS) * 100)
calc("s6.present_pct_max")(lambda: max(_present(t) for t in TAGS) * 100)
calc("s6.wkappa_self_min")(lambda: min(_kappa(CD()[f"coder1_{t}"].values, CD()[f"coder1_{t}"].values) for t in TAGS))
calc("s6.total_r7")(lambda: _totals(TAGS)[0])
calc("s6.total_r6")(lambda: _totals(SIX)[0])
calc("s6.b7")(lambda: _totals(TAGS)[1])
calc("s6.b6")(lambda: _totals(SIX)[1])
calc("s6.r2_7")(lambda: _totals(TAGS)[3])
calc("s6.r2_6")(lambda: _totals(SIX)[3])
# 「p < .001」 문턱 주장 = 문턱 아래면 .001, 아니면 −1 (실패)
calc("s6.p7_below_001")(lambda: 0.001 if _totals(TAGS)[2] < .001 else -1.0)
calc("s6.s3_r_hjs_mil")(lambda: load("svy").hjs.corr(load("svy").mil))
for (_i, _j), _nm in (((0, 0), "t0l0"), ((0, 1), "t0l1"), ((1, 0), "t1l0"), ((1, 1), "t1l1")):
    calc(f"s6.ct_{_nm}")(lambda i=_i, j=_j: int(_ct()[2].loc[i, j]))
calc("s6.chi2_yates")(lambda: stats.chi2_contingency(_ct()[2])[0])
calc("s6.chi2_noyates")(lambda: stats.chi2_contingency(_ct()[2], correction=False)[0])
calc("s6.chi2_p_below_001")(lambda: 0.001 if stats.chi2_contingency(_ct()[2])[1] < .001 else -1.0)
calc("s6.legacy_pct_if_transform")(lambda: _ct()[1][_ct()[0] == 1].mean() * 100)
calc("s6.legacy_pct_if_no_transform")(lambda: _ct()[1][_ct()[0] == 0].mean() * 100)
