# -*- coding: utf-8 -*-
"""16장 · 조건부 효과: 조절과 매개 — 파이썬 엔진."""
import numpy as np
from scipy import stats

from common import calc, load, ols

S = lambda: load("svy")          # 설문판 정제본 566명
E = lambda: load("exp")          # 실험판 정제본 366명

_c = {}


def memo(key, f):
    if key not in _c:
        _c[key] = f()
    return _c[key]


# §16.0 · §16.2 각주 = 14장 §14.2 단순회귀 (설문판)
calc("ch16.svy_slope")(lambda: ols(S().mil, [S().hjs])[0][1])
calc("ch16.svy_intercept")(lambda: ols(S().mil, [S().hjs])[0][0])


# §16.1 중앙값 분할
def _split():
    s = S(); med = s.refl.median()
    lo, hi = s[s.refl <= med], s[s.refl > med]
    sd, m = s.refl.std(ddof=1), s.refl.mean()
    return dict(n_lo=len(lo), n_hi=len(hi),
                b_lo=ols(lo.mil, [lo.hjs])[0][1], b_hi=ols(hi.mil, [hi.hjs])[0][1],
                z_lo=(lo.refl.mean() - m) / sd, z_hi=(hi.refl.mean() - m) / sd)


for k in ("n_lo", "n_hi", "b_lo", "b_hi", "z_lo", "z_hi"):
    calc(f"ch16.split_{k}")(lambda k=k: memo("split", _split)[k])


# §16.2 중심화 · 비중심화 모형
def _mod():
    s = S()
    h, r = s.hjs - s.hjs.mean(), s.refl - s.refl.mean()
    bc, sec, *_ = ols(s.mil, [h, r, h * r])
    br, ser, *_ = ols(s.mil, [s.hjs, s.refl, s.hjs * s.refl])
    sd = s.refl.std(ddof=1)
    return dict(b1=bc[1], b2=bc[2], b3=bc[3], se3=sec[3], raw_b1=br[1], raw_b2=br[2], raw_b3=br[3],
                raw_se3=ser[3], cor_raw=np.corrcoef(s.hjs, s.hjs * s.refl)[0, 1],
                cor_cen=np.corrcoef(h, h * r)[0, 1], refl_sd=sd,
                ss_lo=bc[1] - bc[3] * sd, ss_hi=bc[1] + bc[3] * sd)


for k in ("b1", "b2", "b3", "se3", "raw_b1", "raw_b2", "raw_b3", "raw_se3", "cor_raw", "cor_cen",
          "refl_sd", "ss_lo", "ss_hi"):
    calc(f"ch16.mod_{k}")(lambda k=k: memo("mod", _mod)[k])
calc("ch16.mod_refl_sd_neg")(lambda: -memo("mod", _mod)["refl_sd"])   # 표의 (−1.34)


# §16.5 매개 3단계 (실험판)
def _med():
    e = E()
    c = ols(e.mil, [e.cond])[0][1]
    a = ols(e.hjs, [e.cond])[0][1]
    bc = ols(e.mil, [e.hjs, e.cond])[0]
    return dict(c=c, a=a, b=bc[1], cp=bc[2], ab=a * bc[1], sum=bc[2] + a * bc[1])


for k in ("c", "a", "b", "cp", "ab", "sum"):
    calc(f"ch16.med_{k}")(lambda k=k: memo("med", _med)[k])
# 표에 실린 반올림 수로 검산 (§16.5 상자 ; 정해진 산술)
calc("ch16.rounded_cp_plus_ab")(lambda: 0.15 + 0.13)
calc("ch16.rounded_a_times_b")(lambda: 0.21 * 0.65)


# §16.6 경로 두 모형 (설문판 표준화)
def _paths():
    s = S()
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    X, M, Y = z(s.hjs.values), z(s.refl.values), z(s.mil.values)
    a1 = ols(M, [X])[0]; b1 = ols(Y, [M, X])[0]
    a2 = ols(M, [Y])[0]; b2 = ols(X, [M, Y])[0]
    return dict(a1=a1[1], b1=b1[1], c1=b1[2], a2=a2[1], b2=b2[1], c2=b2[2],
                rep1=a1[1] * b1[1] + b1[2], rep2=a2[1] * b2[1] + b2[2], obs=np.corrcoef(X, Y)[0, 1])


for k in ("a1", "b1", "c1", "a2", "b2", "c2", "rep1", "rep2", "obs"):
    calc(f"ch16.path_{k}")(lambda k=k: memo("paths", _paths)[k])

# 논문 상자 다시 계산 (Rogers 외 2023 보고값으로 하는 산술)
calc("ch16.paper_sym_lo")(lambda: .31 - 1.96 * .12)
calc("ch16.paper_sym_hi")(lambda: .31 + 1.96 * .12)
calc("ch16.paper_mean_diff")(lambda: 5.14 - 4.83)
# Van den Bulck (2004) CAIC 차이 (보고값 산술)
calc("ch16.vdb_caic_gap")(lambda: 456.8 - 426.8)
# 실습 7 = 네 변수의 정보 수
calc("ch16.info4_var")(lambda: 4)
calc("ch16.info4_cov")(lambda: 4 * 3 / 2)
calc("ch16.info4_total")(lambda: 4 * 5 / 2)


# ── 모의시행 (책 코드와 같은 난수 순서 ; 씨앗 73) ──
def _gs():
    """§16.1 [더 나아가기] 상자 = 참 기울기 0.3, 인원 300 · 40."""
    g = np.random.default_rng(73); res = []
    for n_ in (300, 40):
        x = g.normal(0, 1, n_); y = .3 * x + g.normal(0, 1, n_)
        b, se, p, _ = ols(y, [x])
        res.append((b[1], se[1], p[1]))
    z = (res[0][0] - res[1][0]) / np.sqrt(res[0][1] ** 2 + res[1][1] ** 2)
    return dict(b300=res[0][0], p300=res[0][2], b40=res[1][0], p40=res[1][2],
                diff=res[0][0] - res[1][0], pdiff=2 * stats.norm.sf(abs(z)))


for k in ("b300", "b40", "p40", "diff", "pdiff"):
    calc(f"ch16.gs_{k}")(lambda k=k: memo("gs", _gs)[k])
# 「< .001」 = 부등식 주장. p 가 .001 보다 작으면 .001 을 돌려 표기와 맞추고, 아니면 p 그대로(→ 실패)
calc("ch16.gs_p300_lt001")(lambda: .001 if memo("gs", _gs)["p300"] < .001 else memo("gs", _gs)["p300"])


def _power(n=400, rels=(1.0, 0.9, 0.8, 0.7, 0.6)):
    """§16.4 = 신뢰도마다 씨앗 73 으로 새로 시작, 800 번."""
    out = {}
    for rel in rels:
        rng = np.random.default_rng(73)
        sd_e = np.sqrt((1 - rel) / rel)
        hm = hi = 0
        for _ in range(800):
            X, W = rng.normal(0, 1, n), rng.normal(0, 1, n)
            Y = .2 * X + .2 * W + .2 * X * W + rng.normal(0, 1, n)
            Xo, Wo = X + rng.normal(0, sd_e, n), W + rng.normal(0, sd_e, n)
            b, se, *_ = ols(Y, [Xo, Wo, Xo * Wo])
            hm += abs(b[1] / se[1]) > 1.96
            hi += abs(b[3] / se[3]) > 1.96
        out[rel] = (hm / 800, hi / 800)
    return out


for rel, lab in ((1.0, "10"), (0.6, "06")):
    calc(f"ch16.pow{lab}_main")(lambda rel=rel: memo("pow", _power)[rel][0])
    calc(f"ch16.pow{lab}_int")(lambda rel=rel: memo("pow", _power)[rel][1])
calc("ch16.pow07_int")(lambda: memo("pow", _power)[0.7][1])
# 실습 5 = 신뢰도 0.7 에서 n = 800 (같은 코드에 n 만 바꿈)
calc("ch16.pow07_int_n800")(lambda: memo("pow800", lambda: _power(800, (0.7,)))[0.7][1])


def _prodrel():
    """§16.4 곱의 신뢰도 (10만 명 ; 0.8 다음 0.6)."""
    rng = np.random.default_rng(73); n = 100_000; out = {}
    for rel in (0.8, 0.6):
        sd_e = np.sqrt((1 - rel) / rel)
        X, W = rng.normal(0, 1, n), rng.normal(0, 1, n)
        Xo, Wo = X + rng.normal(0, sd_e, n), W + rng.normal(0, sd_e, n)
        out[rel] = (np.corrcoef(X, Xo)[0, 1] ** 2, np.corrcoef(X * W, Xo * Wo)[0, 1] ** 2)
    return out


calc("ch16.prodrel08_x")(lambda: memo("pr", _prodrel)[0.8][0])
calc("ch16.prodrel08_xw")(lambda: memo("pr", _prodrel)[0.8][1])
calc("ch16.prodrel06_x")(lambda: memo("pr", _prodrel)[0.6][0])
calc("ch16.prodrel06_xw")(lambda: memo("pr", _prodrel)[0.6][1])


def _boot():
    """§16.5 부트스트랩 간접효과 (2,000 번)."""
    e = E(); rng = np.random.default_rng(73); ind = []
    for _ in range(2000):
        s = e.iloc[rng.integers(0, len(e), len(e))]
        aa = ols(s.hjs, [s.cond])[0]
        bb = ols(s.mil, [s.hjs, s.cond])[0]
        ind.append(aa[1] * bb[1])
    return np.percentile(ind, [2.5, 97.5])


calc("ch16.boot_lo")(lambda: memo("boot", _boot)[0])
calc("ch16.boot_hi")(lambda: memo("boot", _boot)[1])


def _conf(path=0.0):
    """§16.5 혼란된 매개 세계 (400명). path = 실습 4 의 M → Y 경로."""
    rng = np.random.default_rng(73); n = 400
    cond = rng.integers(0, 2, n)
    U = rng.normal(0, 1, n)
    M = 0.60 * cond + 0.80 * U + rng.normal(0, .5, n)
    Y = 0.30 * cond + path * M + 0.80 * U + rng.normal(0, .5, n)
    a = ols(M, [cond])[0]; bc = ols(Y, [M, cond])[0]
    bu = ols(Y, [M, cond, U])[0]; c = ols(Y, [cond])[0]
    return dict(a=a[1], b=bc[1], cp=bc[2], ab=a[1] * bc[1], bu=bu[1], cpu=bu[2], c=c[1], abu=a[1] * bu[1])


for k in ("a", "b", "cp", "ab", "bu", "cpu", "c", "abu"):
    calc(f"ch16.conf_{k}")(lambda k=k: memo("conf", _conf)[k])
calc("ch16.conf04_abu")(lambda: memo("conf04", lambda: _conf(0.4))["abu"])

from common import CALCS as CALCS_REF

# 2026-11-14h 상자 = 다섯째 자리로 반올림한 c′ 과 a×b 의 합
calc("ch16.med_rounded5_sum")(lambda: round(float(CALCS_REF["ch16.med_cp"]()), 5) + round(float(CALCS_REF["ch16.med_ab"]()), 5))


# §16.2 「논문의 조절 회귀 표 읽기」 = 중심화 모형의 t · p · 95% CI · R²
def _mt():
    if "mt" not in _c:
        h = S().hjs - S().hjs.mean()
        r = S().refl - S().refl.mean()
        b, se, p, r2 = ols(S().mil, [h, r, h * r])
        tc = stats.t.ppf(.975, len(S()) - 4)
        _c["mt"] = {"b": b, "se": se, "p": p, "r2": r2, "lo": b - tc * se, "hi": b + tc * se}
    return _c["mt"]


for _j, _nm in enumerate(("int", "hjs", "refl", "xy")):
    calc(f"ch16.mt_b_{_nm}")(lambda j=_j: _mt()["b"][j])
    calc(f"ch16.mt_se_{_nm}")(lambda j=_j: _mt()["se"][j])
    calc(f"ch16.mt_t_{_nm}")(lambda j=_j: _mt()["b"][j] / _mt()["se"][j])
    calc(f"ch16.mt_p_{_nm}")(lambda j=_j: _mt()["p"][j])
    calc(f"ch16.mt_lo_{_nm}")(lambda j=_j: _mt()["lo"][j])
    calc(f"ch16.mt_hi_{_nm}")(lambda j=_j: _mt()["hi"][j])
calc("ch16.mt_r2")(lambda: _mt()["r2"])
calc("ch16.mt_n")(lambda: len(S()))
