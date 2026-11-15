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


# 2026-11-16c §17.4 「실제 도구로 맞추기」 · §17.1 「매개변수가 둘 이상일 때」 = lavaan 결과를 numpy 로 따로 계산
# 재귀 경로 모형의 최대우도 추정 = 식마다 최소제곱 · 잔차 분산 = 잔차 제곱합 ÷ N · 출발 변수의 분산 · 공분산은 자료값 (fixed.x = FALSE)
from scipy import stats as _st2


def _path_fit(eqs, order):
    """eqs = {결과: [설명 변수]} ; order = 변수 순서. 반환 = χ² · df · CFI · RMSEA · 90% 상한 · SRMR · AIC · BIC · 계수."""
    s = S(); D = s[order].values; N, p = D.shape
    Sm = np.cov(D, rowvar=False, bias=True)
    ix = {v: i for i, v in enumerate(order)}
    B = np.zeros((p, p)); Psi = np.zeros((p, p)); coef = {}
    exo = [v for v in order if v not in eqs]
    for a in exo:
        for b in exo:
            Psi[ix[a], ix[b]] = Sm[ix[a], ix[b]]
    for y, xs in eqs.items():
        X = np.column_stack([np.ones(N)] + [s[x].values for x in xs]); yy = s[y].values
        bb = np.linalg.lstsq(X, yy, rcond=None)[0]
        for x, v in zip(xs, bb[1:]):
            B[ix[y], ix[x]] = v; coef[(y, x)] = v
        Psi[ix[y], ix[y]] = np.mean((yy - X @ bb) ** 2)
    IB = np.linalg.inv(np.eye(p) - B)
    Sig = IB @ Psi @ IB.T
    F = np.log(np.linalg.det(Sig)) - np.log(np.linalg.det(Sm)) + np.trace(Sm @ np.linalg.inv(Sig)) - p
    chi = N * F
    q = sum(len(v) for v in eqs.values()) + len(eqs) + len(exo) * (len(exo) + 1) // 2
    df = p * (p + 1) // 2 - q
    chib = N * (np.sum(np.log(np.diag(Sm))) - np.log(np.linalg.det(Sm))); dfb = p * (p - 1) // 2
    cfi = 1 - max(chi - df, 0) / max(chib - dfb, chi - df, 1e-12)
    rmsea = np.sqrt(max(chi - df, 0) / (df * N))
    # 90% 상한 = 비중심 카이제곱에서 P(χ² ≤ 관측) = .05 가 되는 비중심 모수
    from scipy.optimize import brentq
    g = lambda lam: _st2.ncx2.cdf(chi, df, lam) - 0.05
    lam_u = brentq(g, 1e-9, 1000) if g(1e-9) > 0 else 0.0
    rm_hi = np.sqrt(lam_u / (df * N))
    d = np.sqrt(np.diag(Sm))
    R = (Sm - Sig) / np.outer(d, d)
    srmr = np.sqrt(np.sum(R[np.tril_indices(p)] ** 2) / (p * (p + 1) / 2))
    ll = -N / 2 * (p * np.log(2 * np.pi) + np.log(np.linalg.det(Sig)) + np.trace(Sm @ np.linalg.inv(Sig)))
    return dict(chi=chi, df=df, p=_st2.chi2.sf(chi, df), cfi=cfi, rmsea=rmsea, rm_hi=rm_hi, srmr=srmr,
                aic=-2 * ll + 2 * q, bic=-2 * ll + q * np.log(N), coef=coef)


_PM = {"m1": ({"hjs": ["refl"], "mil": ["hjs"], "swl": ["mil", "hjs"]}, ["refl", "hjs", "mil", "swl"]),
       "m2": ({"mil": ["swl"], "hjs": ["mil", "swl"], "refl": ["hjs"]}, ["refl", "hjs", "mil", "swl"]),
       "m3": ({"hjs": ["refl"], "mil": ["hjs", "refl"], "swl": ["mil"]}, ["refl", "hjs", "mil", "swl"])}
for _m, (_e, _o) in _PM.items():
    for _k in ("chi", "df", "p", "cfi", "rmsea", "rm_hi", "srmr", "aic", "bic"):
        calc(f"ch17.pm_{_m}_{_k}")(lambda e=_e, o=_o, m=_m, k=_k: memo(("pm", m), lambda: _path_fit(e, o))[k])
_c1 = lambda y, x: memo(("pm", "m1"), lambda: _path_fit(*_PM["m1"]))["coef"][(y, x)]
calc("ch17.pm_m1_a")(lambda: _c1("hjs", "refl"))
calc("ch17.pm_m1_d")(lambda: _c1("mil", "hjs"))
calc("ch17.pm_m1_b")(lambda: _c1("swl", "mil"))
calc("ch17.pm_m1_e")(lambda: _c1("swl", "hjs"))
calc("ch17.pm_m1_serial")(lambda: _c1("hjs", "refl") * _c1("mil", "hjs") * _c1("swl", "mil"))
calc("ch17.pm_m1_viamil")(lambda: _c1("mil", "hjs") * _c1("swl", "mil"))


def _boot_paths():
    """사슬 모형의 간접효과 둘 · 병렬 매개(공변인 = 나이)의 특정 간접효과 · 차이 = 부트스트랩 5,000 (씨앗 73)."""
    def f():
        s = S(); n = len(s)
        R_, H, M, W, A = (s[k].values for k in ("refl", "hjs", "mil", "swl", "age"))
        g = np.random.default_rng(73); out = {k: [] for k in ("serial", "viamil", "i1", "i2", "tot", "diff")}
        one = np.ones(n)
        def co(y, *xs):
            return np.linalg.lstsq(np.column_stack((one[:len(y)],) + xs), y, rcond=None)[0]
        for _ in range(5000):
            i = g.integers(0, n, n); r, h, m, w, a = R_[i], H[i], M[i], W[i], A[i]
            aa = co(h, r)[1]; dd = co(m, h)[1]; bb = co(w, m, h)[1]
            out["serial"].append(aa * dd * bb); out["viamil"].append(dd * bb)
            a1 = co(h, r, a)[1]; a2 = co(m, r, a)[1]; bw = co(w, h, m, r, a)
            i1, i2 = a1 * bw[1], a2 * bw[2]
            out["i1"].append(i1); out["i2"].append(i2); out["tot"].append(i1 + i2); out["diff"].append(i1 - i2)
        return {k: np.percentile(v, [2.5, 97.5]) for k, v in out.items()}
    return memo("bootpaths", f)


def _par():
    s = S(); n = len(s); one = np.ones(n)
    co = lambda y, *xs: np.linalg.lstsq(np.column_stack((one,) + xs), y, rcond=None)[0]
    a1 = co(s.hjs.values, s.refl.values, s.age.values)[1]; a2 = co(s.mil.values, s.refl.values, s.age.values)[1]
    bw = co(s.swl.values, s.hjs.values, s.mil.values, s.refl.values, s.age.values)
    return dict(a1=a1, a2=a2, b1=bw[1], b2=bw[2], c=bw[3], i1=a1 * bw[1], i2=a2 * bw[2],
                tot=a1 * bw[1] + a2 * bw[2], diff=a1 * bw[1] - a2 * bw[2])


for _k in ("a1", "a2", "b1", "b2", "c", "i1", "i2", "tot", "diff"):
    calc(f"ch17.par_{_k}")(lambda k=_k: memo("par", _par)[k])
for _k in ("serial", "viamil", "i1", "i2", "tot", "diff"):
    calc(f"ch17.bp_{_k}_lo")(lambda k=_k: _boot_paths()[k][0])
    calc(f"ch17.bp_{_k}_hi")(lambda k=_k: _boot_paths()[k][1])
calc("ch17.pm_daic")(lambda: memo(("pm", "m3"), lambda: _path_fit(*_PM["m3"]))["aic"] - memo(("pm", "m1"), lambda: _path_fit(*_PM["m1"]))["aic"])
calc("ch17.n_raw")(lambda: len(load("svy", clean=False)))
calc("ch17.n_drop")(lambda: len(load("svy", clean=False)) - len(S()))
