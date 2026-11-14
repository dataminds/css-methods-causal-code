# -*- coding: utf-8 -*-
"""14장 · 상관과 회귀 — 파이썬 엔진."""
import os

import numpy as np
import pandas as pd
from scipy import stats

from common import DATA, calc, load, ols

S = lambda: load("svy")          # 정제본 566명
E = lambda: load("exp")          # 정제본 366명

_c: dict = {}


def _fit():
    """설문판 mil ~ hjs (책 §14.2 · §14.3 의 x · y · b · resid)."""
    if "fit" not in _c:
        x, y = S().hjs.values, S().mil.values
        b, se, p, r2 = ols(y, [x])
        resid = y - (b[0] + b[1] * x)
        _c["fit"] = dict(x=x, y=y, b=b, se=se, r2=r2, resid=resid)
    return _c["fit"]


F = lambda k: _fit()[k]

calc("ch14.n_svy")(lambda: len(S()))
calc("ch14.r")(lambda: np.corrcoef(F("x"), F("y"))[0, 1])
calc("ch14.r2")(lambda: F("r2"))
calc("ch14.r2_pct")(lambda: F("r2") * 100)
calc("ch14.unexpl_pct")(lambda: (1 - F("r2")) * 100)
calc("ch14.intercept")(lambda: F("b")[0])
calc("ch14.slope")(lambda: F("b")[1])
calc("ch14.slope_se")(lambda: F("se")[1])
calc("ch14.slope_t")(lambda: F("b")[1] / F("se")[1])
calc("ch14.z975")(lambda: stats.norm.ppf(.975))
calc("ch14.ci_lo")(lambda: F("b")[1] - 1.96 * F("se")[1])
calc("ch14.ci_hi")(lambda: F("b")[1] + 1.96 * F("se")[1])
calc("ch14.resid_sd")(lambda: F("resid").std(ddof=1))
calc("ch14.mil_sd")(lambda: F("y").std(ddof=1))
calc("ch14.hjs_sd")(lambda: F("x").std(ddof=1))
calc("ch14.err_reduction_pct")(lambda: (1 - F("resid").std(ddof=1) / F("y").std(ddof=1)) * 100)
calc("ch14.sd_ratio")(lambda: np.sqrt(1 - F("r2")))


def _far():
    x, y = F("x"), F("y")
    i = np.argmax(np.abs(x - x.mean()))
    keep = np.ones(len(x), bool)
    keep[i] = False
    return x[i], np.polyfit(x[keep], y[keep], 1)[0]


calc("ch14.far_hjs")(lambda: _far()[0])
calc("ch14.slope_drop1")(lambda: _far()[1])
calc("ch14.pred6")(lambda: F("b")[0] + F("b")[1] * 6)
calc("ch14.pred6_lo")(lambda: F("b")[0] + F("b")[1] * 6 - F("resid").std(ddof=1))
calc("ch14.pred6_hi")(lambda: F("b")[0] + F("b")[1] * 6 + F("resid").std(ddof=1))
calc("ch14.rev_slope")(lambda: np.polyfit(S().mil, S().hjs, 1)[0])   # 실습 5 의 뒤바뀐 인자

# §14.4 첫머리 = 실험판 cond 회귀
calc("ch14.exp_n")(lambda: len(E()))
calc("ch14.exp_b0")(lambda: ols(E().mil, [E().cond])[0][0])
calc("ch14.exp_b1")(lambda: ols(E().mil, [E().cond])[0][1])
calc("ch14.exp_p")(lambda: ols(E().mil, [E().cond])[2][1])

# 「먼저」 · 논문 상자 ③ 의 산술 (보고값 .14 · .03 · .74 와 반올림 범위로 계산)
_t_from_r = lambda r, df=590: r * np.sqrt(df) / np.sqrt(1 - r * r)
calc("ch14.paper_t_rounded")(lambda: .14 / .03)
calc("ch14.paper_t_min")(lambda: .135 / .035)
calc("ch14.paper_t_max")(lambda: .145 / .025)
calc("ch14.paper_beta_t")(lambda: _t_from_r(.74))
calc("ch14.paper_beta_t_lo")(lambda: _t_from_r(.735))
calc("ch14.paper_beta_t_hi")(lambda: _t_from_r(.745))


# §14.1 네 세계 = make_book_figures.fig_ch14_shapes 와 같은 생성기 · 같은 모수 (씨앗 73, 세계마다 새로)
_N_SHAPE, _K_OUT = 200, 10


def _shapes():
    if "shapes" not in _c:
        def lin(s):
            g = np.random.default_rng(73); x = g.normal(5, .8, _N_SHAPE)
            return x, 4.9 + 0.98 * (x - 5) + g.normal(0, s, _N_SHAPE)

        def cur(s):
            g = np.random.default_rng(73); x = g.uniform(2, 8, _N_SHAPE)
            return x, 2.2 + 1.35 * x - 0.105 * x ** 2 + g.normal(0, s, _N_SHAPE)

        def out(s):
            g = np.random.default_rng(73); k = _K_OUT
            return (np.concatenate([g.normal(4.6, .4, _N_SHAPE - k), np.full(k, s)]),
                    np.concatenate([g.normal(4.8, .95, _N_SHAPE - k), np.full(k, s + 0.6)]))

        def fan(s):
            g = np.random.default_rng(73); x = g.uniform(2, 8, _N_SHAPE)
            return x, 2.5 + 0.48 * x + g.normal(0, 1, _N_SHAPE) * s * x

        _c["shapes"] = [np.corrcoef(*f(s))[0, 1]
                        for f, s in ((lin, 1.0501), (cur, .5087), (out, 8.2669), (fan, .1871))]
    return _c["shapes"]


calc("ch14.shapes_r_min")(lambda: min(_shapes()))
calc("ch14.shapes_r_max")(lambda: max(_shapes()))
calc("ch14.shapes_outlier_rest")(lambda: _N_SHAPE - _K_OUT)   # 외딴 점 세계에서 열 명을 뺀 나머지


# §14.4 위양성률 = 책 [더 나아가기] 코드와 같은 난수 순서.
# 위양성() 은 부를 때마다 씨앗 73 으로 새로 시작하고, 뒤섞기() 는 부를 때마다 씨앗 37 로 새로 시작한다.
# 그래서 뒤섞기의 500 개 순열은 합친 표본의 길이만 같으면 매번 같다 → 순열 색인을 한 번 만들어 재사용한다
# (g.permutation(pool) 은 pool 의 값과 무관하게 같은 난수로 같은 자리바꿈을 한다).
def _perm_idx(N, B=500):
    key = ("perm", N, B)
    if key not in _c:
        g = np.random.default_rng(37)
        _c[key] = np.array([g.permutation(N) for _ in range(B)])
    return _c[key]


def _perm_p(x, y, B=500):
    obs = abs(x.mean() - y.mean())
    pool = np.concatenate([x, y]); n = len(x)
    P = pool[_perm_idx(len(pool), B)]
    return np.mean(np.abs(P[:, :n].mean(axis=1) - P[:, n:].mean(axis=1)) >= obs)


_t = lambda x, y: stats.ttest_ind(x, y).pvalue
_welch = lambda x, y: stats.ttest_ind(x, y, equal_var=False).pvalue


def _fp(draw, test, R=2000):
    """세계를 R 번 짓는 순서는 책과 같다. 검정만 (t · 웰치는) 한 번에 모아 계산해 시간을 줄인다."""
    g = np.random.default_rng(73)
    xs, ys = zip(*[draw(g) for _ in range(R)])
    xs, ys = np.array(xs), np.array(ys)
    if test in (_t, _welch):
        p = stats.ttest_ind(xs, ys, axis=1, equal_var=(test is _t)).pvalue
    else:
        p = np.array([test(x, y) for x, y in zip(xs, ys)])
    return np.sum(p < .05) / R


def _fp_cached(key, draw, test, R=2000):
    if key not in _c:
        _c[key] = _fp(draw, test, R)
    return _c[key]


for _n in (10, 30, 100):
    _skew = lambda g, n=_n: (g.exponential(1, n), g.exponential(1, n))
    calc(f"ch14.fp_skew{_n}_t")(lambda n=_n, d=_skew: _fp_cached(("skt", n), d, _t))
    calc(f"ch14.fp_skew{_n}_perm")(lambda n=_n, d=_skew: _fp_cached(("skp", n), d, _perm_p))

_small_big = lambda g: (g.normal(0, 3, 20), g.normal(0, 1, 60))     # 큰 분산이 작은 집단
_big_big = lambda g: (g.normal(0, 1, 20), g.normal(0, 3, 60))       # 큰 분산이 큰 집단
for _nm, _d in (("smallvar", _small_big), ("bigvar", _big_big)):
    calc(f"ch14.fp_{_nm}_t")(lambda nm=_nm, d=_d: _fp_cached((nm, "t"), d, _t))
    calc(f"ch14.fp_{_nm}_welch")(lambda nm=_nm, d=_d: _fp_cached((nm, "w"), d, _welch))
    calc(f"ch14.fp_{_nm}_perm")(lambda nm=_nm, d=_d: _fp_cached((nm, "p"), d, _perm_p, R=500))
calc("ch14.fp_smallvar_t_pct")(lambda: _fp_cached(("smallvar", "t"), _small_big, _t) * 100)


# §14.5 두꺼운 꼬리의 검정력 = 책 코드와 같은 난수 순서
def _power():
    if "power" not in _c:
        g = np.random.default_rng(73)
        U, V = [], []
        for _ in range(2000):                  # 뽑는 순서는 책 코드와 같다
            U.append(g.standard_t(2, 40) + 0.8)
            V.append(g.standard_t(2, 40))
        U, V = np.array(U), np.array(V)
        w1 = np.sum(stats.ttest_ind(U, V, axis=1).pvalue < .05)
        w2 = np.sum(stats.mannwhitneyu(U, V, axis=1).pvalue < .05)
        _c["power"] = (w1 / 2000, w2 / 2000)
    return _c["power"]


calc("ch14.power_t")(lambda: _power()[0])
calc("ch14.power_mw")(lambda: _power()[1])


# §14.4 독립성 「20% 좁게」 = 웹 부록 S5 의 블록 다시 뽑기와 같은 계산 (시계열판, 씨앗 73)
def _ts_narrow():
    if "ts" not in _c:
        ts = pd.read_csv(os.path.join(DATA, "journey_ts.csv"))
        b, se, p, _ = ols(ts.wellbeing, [ts.week, ts.campaign])
        X1 = np.column_stack([np.ones(len(ts)), ts.week, ts.campaign])
        fit, resid = X1 @ b, ts.wellbeing.values - X1 @ b
        rng = np.random.default_rng(73)
        boot = []
        for _ in range(2000):
            starts = rng.integers(0, len(ts) - 8, 13)
            e = np.concatenate([resid[s:s + 8] for s in starts])[:len(ts)]
            bb, *_ = np.linalg.lstsq(X1, fit + e, rcond=None)
            boot.append(bb[2])
        _c["ts"] = (se[2], float(np.std(boot, ddof=1)))
    return _c["ts"]


# 블록 표준오차가 소박한 표준오차보다 몇 % 큰가 (0.084 ÷ 0.070 − 1)
calc("ch14.ts_naive_narrow_pct")(lambda: (_ts_narrow()[1] / _ts_narrow()[0] - 1) * 100)
