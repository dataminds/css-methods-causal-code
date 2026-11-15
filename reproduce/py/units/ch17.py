# -*- coding: utf-8 -*-
"""17장 · 매개와 경로분석 — 파이썬 엔진.

2026-11-15 16장 분리 = 16장 units(ch16.py)의 매개 · 경로 계산을 복사해 이름을 ch17.* 로 바꿨다.
16장 쪽 원본은 fork-2 가 16장 개편 때 지운다.
"""
import numpy as np

from common import calc, load, ols

S = lambda: load("svy")          # 설문판 정제본 566명
E = lambda: load("exp")          # 실험판 정제본 366명

_c = {}


def memo(key, f):
    if key not in _c:
        _c[key] = f()
    return _c[key]


# §17.1 매개 3단계 (실험판 정제본)
def _med():
    e = E()
    c = ols(e.mil, [e.cond])[0][1]
    a = ols(e.hjs, [e.cond])[0][1]
    bc = ols(e.mil, [e.hjs, e.cond])[0]
    return dict(c=c, a=a, b=bc[1], cp=bc[2], ab=a * bc[1], sum=bc[2] + a * bc[1])


for k in ("c", "a", "b", "cp", "ab", "sum"):
    calc(f"ch17.med_{k}")(lambda k=k: memo("med", _med)[k])
# 표에 실린 반올림 수로 검산 (§17.1 상자 ; 정해진 산술)
calc("ch17.rounded_cp_plus_ab")(lambda: 0.15 + 0.13)
calc("ch17.rounded_a_times_b")(lambda: 0.21 * 0.65)
# §17.1 상자 = 다섯째 자리로 반올림한 c′ 과 a×b 의 합
calc("ch17.med_rounded5_sum")(lambda: round(float(memo("med", _med)["cp"]), 5) + round(float(memo("med", _med)["ab"]), 5))


# §17.4 경로 두 모형 (설문판 표준화)
def _paths():
    s = S()
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    X, M, Y = z(s.hjs.values), z(s.refl.values), z(s.mil.values)
    a1 = ols(M, [X])[0]; b1 = ols(Y, [M, X])[0]
    a2 = ols(M, [Y])[0]; b2 = ols(X, [M, Y])[0]
    return dict(a1=a1[1], b1=b1[1], c1=b1[2], a2=a2[1], b2=b2[1], c2=b2[2],
                rep1=a1[1] * b1[1] + b1[2], rep2=a2[1] * b2[1] + b2[2], obs=np.corrcoef(X, Y)[0, 1])


for k in ("a1", "b1", "c1", "a2", "b2", "c2", "rep1", "rep2", "obs"):
    calc(f"ch17.path_{k}")(lambda k=k: memo("paths", _paths)[k])

# 논문 상자 다시 계산 (Rogers 외 2023 보고값으로 하는 산술)
calc("ch17.paper_sym_lo")(lambda: .31 - 1.96 * .12)
calc("ch17.paper_sym_hi")(lambda: .31 + 1.96 * .12)
calc("ch17.paper_mean_diff")(lambda: 5.14 - 4.83)
# Van den Bulck (2004) CAIC 차이 (보고값 산술)
calc("ch17.vdb_caic_gap")(lambda: 456.8 - 426.8)
# 실습 4 = 네 변수의 정보 수
calc("ch17.info4_var")(lambda: 4)
calc("ch17.info4_cov")(lambda: 4 * 3 / 2)
calc("ch17.info4_total")(lambda: 4 * 5 / 2)


# §17.1 부트스트랩 간접효과 (2,000 번)
def _boot():
    e = E(); rng = np.random.default_rng(73); ind = []
    for _ in range(2000):
        s = e.iloc[rng.integers(0, len(e), len(e))]
        aa = ols(s.hjs, [s.cond])[0]
        bb = ols(s.mil, [s.hjs, s.cond])[0]
        ind.append(aa[1] * bb[1])
    return np.percentile(ind, [2.5, 97.5])


calc("ch17.boot_lo")(lambda: memo("boot", _boot)[0])
calc("ch17.boot_hi")(lambda: memo("boot", _boot)[1])


# §17.2 혼란된 매개 세계 (400명). path = 실습 2 의 M → Y 경로.
def _conf(path=0.0):
    rng = np.random.default_rng(73); n = 400
    cond = rng.integers(0, 2, n)
    U = rng.normal(0, 1, n)
    M = 0.60 * cond + 0.80 * U + rng.normal(0, .5, n)
    Y = 0.30 * cond + path * M + 0.80 * U + rng.normal(0, .5, n)
    a = ols(M, [cond])[0]; bc = ols(Y, [M, cond])[0]
    bu = ols(Y, [M, cond, U])[0]; c = ols(Y, [cond])[0]
    return dict(a=a[1], b=bc[1], cp=bc[2], ab=a[1] * bc[1], bu=bu[1], cpu=bu[2], c=c[1], abu=a[1] * bu[1])


for k in ("a", "b", "cp", "ab", "bu", "cpu", "c", "abu"):
    calc(f"ch17.conf_{k}")(lambda k=k: memo("conf", _conf)[k])
calc("ch17.conf04_abu")(lambda: memo("conf04", lambda: _conf(0.4))["abu"])


# 2026-11-15q §17.4 「원인과 결과를 바꾼 모형의 적합도」 = 세 변수 사슬 A → B → C (직접 경로 없음 ; 자유도 1)
# 최대우도 적합 함수 F = ln|Σ̂| − ln|S| + tr(S Σ̂⁻¹) − p, 검정 통계량 = N × F (lavaan 기본과 같은 N 배).
from scipy import stats as _st


def _chain_fit(a, b, c):
    def f():
        d = S()[[a, b, c]].values
        Sm = np.cov(d, rowvar=False, bias=True)          # 최대우도 공분산 (N 으로 나눔)
        Sh = Sm.copy()
        Sh[0, 2] = Sh[2, 0] = Sm[0, 1] * Sm[1, 2] / Sm[1, 1]   # 사슬이 재현하는 A–C 공분산
        F = np.log(np.linalg.det(Sh)) - np.log(np.linalg.det(Sm)) + np.trace(Sm @ np.linalg.inv(Sh)) - 3
        chi = len(d) * F
        return chi, _st.chi2.sf(chi, 1)
    return memo(("chain", a, b, c), f)


for _nm, (_a, _b, _c3) in {"hrm": ("hjs", "refl", "mil"), "mrh": ("mil", "refl", "hjs"),
                          "hmr": ("hjs", "mil", "refl"), "rmh": ("refl", "mil", "hjs"),
                          "rhm": ("refl", "hjs", "mil"), "mhr": ("mil", "hjs", "refl")}.items():
    calc(f"ch17.chain_{_nm}_chi")(lambda a=_a, b=_b, c=_c3: _chain_fit(a, b, c)[0])
    calc(f"ch17.chain_{_nm}_p")(lambda a=_a, b=_b, c=_c3: _chain_fit(a, b, c)[1])


def _partial_hm_r():                      # 회고 습관을 고정한 여정 지각–삶의 의미 편상관 (§17.4)
    s = S(); R = np.corrcoef([s.hjs, s.refl, s.mil])
    return (R[0, 2] - R[0, 1] * R[1, 2]) / np.sqrt((1 - R[0, 1] ** 2) * (1 - R[1, 2] ** 2))


calc("ch17.partial_hm_r")(_partial_hm_r)


# 2026-11-16b §17.2 횡단 설문의 매개 = 회고 습관 → 여정 지각 → 삶의 의미 와 원인 · 결과를 바꾼 순서 (코드 7 과 같은 난수 순서)
def _xmed(x, m, y):
    def f():
        s = S(); X, M, Y = s[x].values, s[m].values, s[y].values
        c = ols(Y, [X])[0][1]; a = ols(M, [X])[0][1]; b = ols(Y, [M, X])[0][1]
        g = np.random.default_rng(73); n = len(X); ab = []
        for _ in range(2000):
            i = g.integers(0, n, n)
            ab.append(ols(M[i], [X[i]])[0][1] * ols(Y[i], [M[i], X[i]])[0][1])
        lo, hi = np.percentile(ab, [2.5, 97.5])
        return dict(c=c, ab=a * b, lo=lo, hi=hi)
    return memo(("xmed", x, m, y), f)


for _nm, _t in {"fwd": ("refl", "hjs", "mil"), "rev": ("mil", "hjs", "refl")}.items():
    for _k in ("c", "ab", "lo", "hi"):
        calc(f"ch17.xmed_{_nm}_{_k}")(lambda t=_t, k=_k: _xmed(*t)[k])
