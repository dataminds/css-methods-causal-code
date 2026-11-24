# -*- coding: utf-8 -*-
"""시나리오 S7 · 학위 논문 한 편 — 파이썬 엔진 (T3 표본 크기 역산 ; 나머지 수는 S1 · S2 · S3 계산 키로 회수)."""
import numpy as np
from scipy import stats

from common import calc, cronbach, load, ols


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


# ── T3 연구 1(설문) 표본 계획 (2026-11-24) ──────────────────────────────
def _fisher_n(r=.20, alpha=.05, power=.80):
    z = stats.norm.ppf
    return ((z(1 - alpha / 2) + z(power)) / np.arctanh(r)) ** 2 + 3


def _corr_sim_power(n=194, r=.20, reps=2000, seed=73):
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(reps):
        x = rng.normal(size=n)
        y = r * x + np.sqrt(1 - r ** 2) * rng.normal(size=n)
        hit += stats.pearsonr(x, y)[1] < .05
    return hit / reps


def _int_power(n, b3=.15, rel=.8, reps=800, seed=73):
    """16장 노트북 셀 4 의 설계(주효과 .2 · .2)에서 곱항 크기 · 신뢰도만 바꾼다."""
    rng = np.random.default_rng(seed)
    sd_e = np.sqrt((1 - rel) / rel)
    hit = 0
    for _ in range(reps):
        X, W = rng.normal(0, 1, n), rng.normal(0, 1, n)
        Y = .2 * X + .2 * W + b3 * X * W + rng.normal(0, 1, n)
        Xo, Wo = X + rng.normal(0, sd_e, n), W + rng.normal(0, sd_e, n)
        b, se, *_ = ols(Y, [Xo, Wo, Xo * Wo])
        hit += abs(b[3] / se[3]) > 1.96
    return hit / reps


_MEMO: dict = {}


def _memo(key, f):
    if key not in _MEMO:
        _MEMO[key] = f()
    return _MEMO[key]


calc("s7.plan_r20_n")(lambda: _fisher_n())
calc("s7.plan_r20_simpow")(lambda: _memo("cs", _corr_sim_power))
for _n in (400, 500, 600):
    calc(f"s7.plan_int_pow_{_n}")(lambda _n=_n: _memo(f"ip{_n}", lambda: _int_power(_n)))


# ── T6 연구 1 결과 절 (설문판 554명 ; 2026-11-24) ─────────────────────────
def _S():
    V = load("svy")
    return V[V.gender.isin([1, 2])]


def _X(stage):
    s = _S()
    fem = (s.gender == 2).astype(float)
    h = s.hjs - s.hjs.mean()
    r = s.refl - s.refl.mean()
    return [[s.age, fem], [s.age, fem, h], [s.age, fem, h, r], [s.age, fem, h, r, h * r]][stage - 1]


def _r2(stage):
    return ols(_S().mil, _X(stage))[3]


def _fchange(stage):
    """stage 단계에서 더한 항 하나의 F (분자 자유도 1)."""
    n = len(_S())
    df2 = n - len(_X(stage)) - 1
    return (_r2(stage) - _r2(stage - 1)) / ((1 - _r2(stage)) / df2), df2


def _lt001(p):
    return 0.001 if p < 0.001 else float("nan")


calc("s7.r1_f1")(lambda: (_r2(1) / 2) / ((1 - _r2(1)) / (len(_S()) - 3)))
calc("s7.r1_f1_p")(lambda: stats.f.sf((_r2(1) / 2) / ((1 - _r2(1)) / (len(_S()) - 3)), 2, len(_S()) - 3))
calc("s7.r1_f1_df2")(lambda: len(_S()) - 3)
calc("s7.r1_f2")(lambda: _fchange(2)[0])
calc("s7.r1_f2_df2")(lambda: _fchange(2)[1])
calc("s7.r1_f2_p_lt")(lambda: _lt001(stats.f.sf(_fchange(2)[0], 1, _fchange(2)[1])))
calc("s7.r1_dr2_3")(lambda: _r2(3) - _r2(2))
calc("s7.r1_f4")(lambda: _fchange(4)[0])
calc("s7.r1_f4_df2")(lambda: _fchange(4)[1])
calc("s7.r1_f4_p_lt")(lambda: _lt001(stats.f.sf(_fchange(4)[0], 1, _fchange(4)[1])))
calc("s7.r1_se_hjs2")(lambda: ols(_S().mil, _X(2))[1][3])
calc("s7.r1_se_int")(lambda: ols(_S().mil, _X(4))[1][5])


def _slope_ci(sign, side):
    """회고 습관을 ±1SD 로 옮겨 다시 중심화한 모형의 여정 지각 계수와 t 구간."""
    s = _S()
    fem = (s.gender == 2).astype(float)
    h = s.hjs - s.hjs.mean()
    r = s.refl - s.refl.mean()
    rr = r - sign * r.std(ddof=1)
    b, se, _, _ = ols(s.mil, [s.age, fem, h, rr, h * rr])
    t = stats.t.ppf(.975, len(s) - 6)
    return b[3] + side * t * se[3]


calc("s7.r1_slope_lo_lo")(lambda: _slope_ci(-1, -1))
calc("s7.r1_slope_lo_hi")(lambda: _slope_ci(-1, 1))
calc("s7.r1_slope_hi_lo")(lambda: _slope_ci(1, -1))
calc("s7.r1_slope_hi_hi")(lambda: _slope_ci(1, 1))


def _hc3(stage, j):
    y = np.asarray(_S().mil, float)
    X1 = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in _X(stage)])
    XtXi = np.linalg.inv(X1.T @ X1)
    e = y - X1 @ (XtXi @ X1.T @ y)
    h = np.einsum("ij,jk,ik->i", X1, XtXi, X1)
    meat = X1.T @ (X1 * (e ** 2 / (1 - h) ** 2)[:, None])
    return float(np.sqrt(np.diag(XtXi @ meat @ XtXi))[j])


calc("s7.r1_hc3_hjs")(lambda: _hc3(2, 3))
calc("s7.r1_hc3_int")(lambda: _hc3(4, 5))

# 상관 (566명) · 삶의 만족 알파
calc("s7.r1_r_df")(lambda: len(load("svy")) - 2)
calc("s7.r1_r_lo")(lambda: float(np.tanh(np.arctanh(np.corrcoef(load("svy").hjs, load("svy").mil)[0, 1])
                                         - stats.norm.ppf(.975) / np.sqrt(len(load("svy")) - 3))))
calc("s7.r1_r_hi")(lambda: float(np.tanh(np.arctanh(np.corrcoef(load("svy").hjs, load("svy").mil)[0, 1])
                                         + stats.norm.ppf(.975) / np.sqrt(len(load("svy")) - 3))))
calc("s7.r1_r_p_lt")(lambda: _lt001(stats.pearsonr(load("svy").hjs, load("svy").mil)[1]))
calc("s7.r1_swl_alpha")(lambda: cronbach(load("svy")[[f"swl_{i}" for i in range(1, 6)]]))


# 매개 (여정 지각 → 삶의 의미 → 삶의 만족 ; 나이 · 성별을 두 식 모두에)
def _med_parts():
    s = _S()
    fem = (s.gender == 2).astype(float)
    A = ols(s.mil, [s.hjs, s.age, fem])
    B = ols(s.swl, [s.mil, s.hjs, s.age, fem])
    C = ols(s.swl, [s.hjs, s.age, fem])
    return A, B, C


calc("s7.r1_med_a")(lambda: _med_parts()[0][0][1])
calc("s7.r1_med_a_se")(lambda: _med_parts()[0][1][1])
calc("s7.r1_med_a_p_lt")(lambda: _lt001(_med_parts()[0][2][1]))
calc("s7.r1_med_b")(lambda: _med_parts()[1][0][1])
calc("s7.r1_med_b_se")(lambda: _med_parts()[1][1][1])
calc("s7.r1_med_b_p_lt")(lambda: _lt001(_med_parts()[1][2][1]))
calc("s7.r1_med_c")(lambda: _med_parts()[2][0][1])
calc("s7.r1_med_c_se")(lambda: _med_parts()[2][1][1])
calc("s7.r1_med_c_p_lt")(lambda: _lt001(_med_parts()[2][2][1]))
calc("s7.r1_med_cp")(lambda: _med_parts()[1][0][2])
calc("s7.r1_med_cp_se")(lambda: _med_parts()[1][1][2])
calc("s7.r1_med_cp_p")(lambda: _med_parts()[1][2][2])
calc("s7.r1_med_ab")(lambda: _med_parts()[0][0][1] * _med_parts()[1][0][1])


def _med_boot(B=5000, seed=73):
    s = _S()
    X, M, Y, A = s.hjs.values, s.mil.values, s.swl.values, s.age.values
    F = (s.gender == 2).astype(float).values
    n = len(s)
    one = np.ones(n)
    rng = np.random.default_rng(seed)
    ab = np.empty(B)
    for i in range(B):
        j = rng.integers(0, n, n)
        a = np.linalg.lstsq(np.column_stack([one, X[j], A[j], F[j]]), M[j], rcond=None)[0][1]
        b = np.linalg.lstsq(np.column_stack([one, M[j], X[j], A[j], F[j]]), Y[j], rcond=None)[0][1]
        ab[i] = a * b
    return np.percentile(ab, [2.5, 97.5])


calc("s7.r1_med_boot_lo")(lambda: float(_memo("mb", _med_boot)[0]))
calc("s7.r1_med_boot_hi")(lambda: float(_memo("mb", _med_boot)[1]))
