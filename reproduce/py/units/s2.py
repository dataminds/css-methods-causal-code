# -*- coding: utf-8 -*-
"""시나리오 S2 · 요인과 사전-사후 — 파이썬 엔진."""
import os
import sys

import numpy as np
from scipy import stats

import common
from common import calc, load, ols

F = lambda: load("fac")   # 정제본 430명


def _cell(e, f, v="mil_t2"):
    d = F()
    return d[(d.elem == e) & (d.frame == f)][v]


calc("s2.n")(lambda: len(F()))
calc("s2.cell_n_min")(lambda: int(F().groupby(["elem", "frame"]).size().min()))
calc("s2.cell_n_max")(lambda: int(F().groupby(["elem", "frame"]).size().max()))
for e in (0, 1):
    for f in (0, 1):
        calc(f"s2.m{e}{f}")(lambda e=e, f=f: _cell(e, f).mean())
calc("s2.diff_elem_only")(lambda: _cell(1, 0).mean() - _cell(0, 0).mean())
calc("s2.diff_frame_only")(lambda: _cell(0, 1).mean() - _cell(0, 0).mean())
calc("s2.diff_both")(lambda: _cell(1, 1).mean() - _cell(0, 0).mean())
calc("s2.did")(lambda: (_cell(1, 1).mean() - _cell(1, 0).mean()) - (_cell(0, 1).mean() - _cell(0, 0).mean()))


def _fit():
    d = F()
    e, f = d.elem.values, d.frame.values
    return ols(d.mil_t2.values, [e, f, e * f])


def _ci(side):
    b, se, _, _ = _fit()
    t = stats.t.ppf(.975, len(F()) - 4)
    return b[3] + side * t * se[3]


calc("s2.b3")(lambda: _fit()[0][3])
calc("s2.p3")(lambda: _fit()[2][3])
calc("s2.ci_lo")(lambda: _ci(-1))
calc("s2.ci_hi")(lambda: _ci(1))
calc("s2.t3_sq")(lambda: (_fit()[0][3] / _fit()[1][3]) ** 2)
calc("s2.resid_df")(lambda: len(F()) - 4)


# 사전-사후 함정 (짝지은 t)
def _paired(e, f):
    d = F()
    c = d[(d.elem == e) & (d.frame == f)]
    dd = c.mil_t2.values - c.mil_t1.values
    t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
    return c.mil_t1.mean(), c.mil_t2.mean(), t, float(2 * stats.t.sf(abs(t), len(dd) - 1))


for lab, (e, f) in (("int", (1, 1)), ("ctl", (0, 0))):
    for k, nm in enumerate(("pre", "post", "t", "p")):
        calc(f"s2.{lab}_{nm}")(lambda e=e, f=f, k=k: _paired(e, f)[k])
calc("s2.ctl_gain")(lambda: _paired(0, 0)[1] - _paired(0, 0)[0])


def _gain_diff():
    # 함정 절 둘째 코드 : 두 셀의 변화 폭을 웰치 t 로 견준다
    d = F()
    g = [(c.mil_t2 - c.mil_t1).values for c in
         (d[(d.elem == 1) & (d.frame == 1)], d[(d.elem == 0) & (d.frame == 0)])]
    r = stats.ttest_ind(g[0], g[1], equal_var=False)
    ci = r.confidence_interval(0.95)
    return g[0].mean(), g[1].mean(), g[0].mean() - g[1].mean(), float(ci.low), float(ci.high), float(r.pvalue)


for k, nm in enumerate(("int", "ctl", "diff", "lo", "hi", "p")):
    calc(f"s2.gain_{nm}")(lambda k=k: _gain_diff()[k])
calc("s2.r_prepost")(lambda: float(np.corrcoef(F().mil_t1, F().mil_t2)[0, 1]))

# 「먼저」 상자의 되짚기 산술 (논문 F 에서 부분 η²)
calc("s2.eta_elem")(lambda: 4.93 / (4.93 + 445))
calc("s2.eta_frame")(lambda: 6.06 / (6.06 + 445))


# §교훈 √2 = 설계 행렬만으로 정해지는 비 (책 코드와 같은 난수 ; 비는 난수와 무관)
def _sqrt2():
    rng = np.random.default_rng(73)
    n = 112
    e = np.repeat([0, 0, 1, 1], n); f = np.repeat([0, 1, 0, 1], n)
    y = 4.8 + 0.38 * e + 0.38 * e * f + rng.normal(0, 1.2, 4 * n)
    _, se, *_ = ols(y, [e, f, e * f])
    return se[3] / se[1]


calc("s2.se_ratio")(_sqrt2)

# §교훈 이상화된 검정력 곡선 = materials/make_book_figures.py fig_s2_power 와 같은 난수 순서
_cache = {}


def _ideal(N, reps=2000, eff=0.38, sd=1.20):
    key = ("ideal", N)
    if key not in _cache:
        rng = np.random.default_rng(73)
        n = N // 4; hm = hi = 0
        e = np.repeat([0, 0, 1, 1], n); f = np.repeat([0, 1, 0, 1], n)
        X1 = np.column_stack([np.ones(4 * n), e, f, e * f])
        XtXi = np.linalg.inv(X1.T @ X1)
        for _ in range(reps):
            y = 4.8 + eff * e + eff * e * f + rng.normal(0, sd, 4 * n)
            b, *_ = np.linalg.lstsq(X1, y, rcond=None)
            r_ = y - X1 @ b
            se = np.sqrt(np.diag(r_ @ r_ / (4 * n - 4) * XtXi))
            hm += abs(b[1] / se[1]) > 1.96
            hi += abs(b[3] / se[3]) > 1.96
        _cache[key] = (hm / reps, hi / reps)
    return _cache[key]


calc("s2.ideal_main_448")(lambda: _ideal(448)[0])
calc("s2.ideal_inter_896")(lambda: _ideal(896)[1])


# 요인판 실제 구조로 돌린 검정력 표 + 점근 참값 = materials/make_data_family.py gen_fac
def _gen_fac():
    mat = os.path.dirname(os.path.dirname(common.HERE))
    if mat not in sys.path:
        sys.path.insert(0, mat)
    from make_data_family import gen_fac
    return gen_fac


def _fac_power(n, reps=400):
    key = ("fac", n)
    if key not in _cache:
        gen_fac = _gen_fac()
        hits = 0
        for i in range(reps):
            f = gen_fac(np.random.default_rng(1000 + i), n=n)
            f = f[f.attn_1 == 1]
            _, _, pp, _ = ols(f.mil_t2.values, [f.elem.values, f.frame.values, (f.elem * f.frame).values])
            hits += pp[3] < .05
        _cache[key] = hits / reps
    return _cache[key]


def _truth():
    if "truth" not in _cache:
        big = _gen_fac()(np.random.default_rng(73), n=120000)
        b, *_ = ols(big.mil_t2.values, [big.elem.values, big.frame.values, (big.elem * big.frame).values])
        _cache["truth"] = b[3]
    return _cache["truth"]


calc("s2.fac_power_448")(lambda: _fac_power(448))
calc("s2.fac_power_896")(lambda: _fac_power(896))
calc("s2.fac_power_1792")(lambda: _fac_power(1792))
calc("s2.truth_b3")(_truth)
