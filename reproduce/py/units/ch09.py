# -*- coding: utf-8 -*-
"""9장 · 인과 주장은 설계에 기댄다 — 파이썬 엔진."""
import math

import numpy as np
import pandas as pd

from common import calc, cohen_d, load, ols

# 먼저 절 · §9.1 상자 = Rogers 외(2023) 연구 4 인원의 산술
calc("ch09.attr_n")(lambda: 275 - 247)
calc("ch09.attr_pct")(lambda: (275 - 247) / 275 * 100)
calc("ch09.excl_pct")(lambda: 7 / 247 * 100)

# §9.2 이중차분 가상 수의 산술
calc("ch09.did_policy")(lambda: 58 - 50)
calc("ch09.did_comp")(lambda: 53 - 48)
calc("ch09.did_effect")(lambda: (58 - 50) - (53 - 48))

# §9.1 · §9.7 실험판 원자료 인원
calc("ch09.exp_n_raw")(lambda: len(load("exp", clean=False)))

# §9.3 패널판
P = lambda: load("panel")


def _w1_r():
    w1 = P()[P().wave == 1]
    return w1.hjs.corr(w1.mil)


def _within_r():
    p = P()
    wi = p[["hjs", "mil"]] - p.groupby("id")[["hjs", "mil"]].transform("mean")
    return wi.hjs.corr(wi.mil)


def _cross_lag(yv, ylag, xlag):
    wide = P().pivot(index="id", columns="wave", values=["hjs", "mil"]).dropna()
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    b, *_ = ols(z(wide[yv]).values, [z(wide[ylag]).values, z(wide[xlag]).values])
    return b[2]


calc("ch09.panel_r_w1")(_w1_r)
calc("ch09.panel_r_within")(_within_r)
calc("ch09.lag_hm12")(lambda: _cross_lag(("mil", 2), ("mil", 1), ("hjs", 1)))
calc("ch09.lag_hm23")(lambda: _cross_lag(("mil", 3), ("mil", 2), ("hjs", 2)))
calc("ch09.lag_mh12")(lambda: _cross_lag(("hjs", 2), ("hjs", 1), ("mil", 1)))
calc("ch09.lag_mh23")(lambda: _cross_lag(("hjs", 3), ("hjs", 2), ("mil", 2)))


# §9.1 두 코드 블록 = 한 rng 를 이어 쓴다 (씨앗 73)
def _balance():
    rng = np.random.default_rng(73)
    n = 400
    age = rng.normal(35, 10, n); sex = rng.integers(0, 2, n)
    opt = rng.normal(0, 1, n); grit = rng.normal(0, 1, n)
    z = rng.integers(0, 2, n)
    p = 1 / (1 + np.exp(-1.2 * opt))
    zs = (rng.random(n) < p).astype(int)
    out = {"d_opt_self": cohen_d(opt[zs == 1], opt[zs == 0])}
    base = rng.normal(0, 1, n)
    pre = base + rng.normal(0, 0.5, n)
    order = np.argsort(pre)
    simple, block = [], []
    for _ in range(2000):
        z = rng.integers(0, 2, n)
        simple.append(cohen_d(base[z == 1], base[z == 0]))
        g = np.zeros(n, int)
        for i in range(0, n, 2):
            g[order[i:i + 2][rng.integers(0, 2)]] = 1
        block.append(cohen_d(base[g == 1], base[g == 0]))
    for lab, v in (("simple", simple), ("block", block)):
        v = np.array(v)
        out[lab] = (v.mean(), v.std(ddof=1), np.mean(np.abs(v) > 0.2))
    return out


_c = {}


def bal():
    if not _c:
        _c.update(_balance())
    return _c


calc("ch09.sim_d_opt_self")(lambda: bal()["d_opt_self"])
for lab in ("simple", "block"):
    calc(f"ch09.sim_{lab}_mean")(lambda lab=lab: bal()[lab][0])
    calc(f"ch09.sim_{lab}_sd")(lambda lab=lab: bal()[lab][1])
    calc(f"ch09.sim_{lab}_over")(lambda lab=lab: bal()[lab][2])
calc("ch09.sim_simple_over_per100")(lambda: bal()["simple"][2] * 100)


# §9.3 세 세계 (상자 코드 ; asym = 상자 끝 ⚠ 의 바꾼 C 줄)
def worlds(seed=73, n=5000, asym=False):
    g = np.random.default_rng(seed)
    e = lambda s=1: g.normal(0, s, n)
    x1 = e(); y1 = .3 * x1 + e(); x2 = .6 * x1 + e(.8); y2 = .5 * y1 + .3 * x1 + e(.8)
    A = (x1, y1, x2, y2)
    y1 = e(); x1 = .3 * y1 + e(); y2 = .6 * y1 + e(.8); x2 = .5 * x1 + .3 * y1 + e(.8)
    B = (x1, y1, x2, y2)
    T = e()
    if asym:
        C = (.64 * T + e(), .64 * T + e(.5), .64 * T + e(), .64 * T + e(.5))
    else:
        C = tuple(.64 * T + e() for _ in range(4))
    return {"A": A, "B": B, "C": C}


def lag(before_self, cause, after):
    q = pd.qcut(before_self, 10, labels=False)
    return float(np.mean([np.polyfit(cause[q == k], after[q == k], 1)[0] for k in range(10)]))


_w = {}


def three(seed, asym=False):
    key = (seed, asym)
    if key not in _w:
        res = {}
        for nm, (x1, y1, x2, y2) in worlds(seed, asym=asym).items():
            res[nm] = (np.corrcoef(x1, y1)[0, 1], lag(y1, x1, y2), lag(x1, y1, x2))
        _w[key] = res
    return _w[key]


for seed in (73, 37):
    for nm in "ABC":
        for j, q in enumerate(("cross", "xy", "yx")):
            calc(f"ch09.w{seed}_{nm}_{q}")(lambda seed=seed, nm=nm, j=j: three(seed)[nm][j])
calc("ch09.wasym_C_xy")(lambda: three(73, asym=True)["C"][1])
calc("ch09.wasym_C_yx")(lambda: three(73, asym=True)["C"][2])


def _tenths_band(vals):
    """「0.2대」 = 값들의 소수 첫째 자리(버림)가 모두 같으면 그 값, 아니면 NaN."""
    fl = {math.floor(v * 10 + 1e-12) / 10 for v in vals}
    return fl.pop() if len(fl) == 1 else float("nan")


calc("ch09.w73_cross_band")(lambda: _tenths_band([three(73)[nm][0] for nm in "ABC"]))
calc("ch09.w73_C_lag_band")(lambda: _tenths_band([three(73)["C"][1], three(73)["C"][2]]))
