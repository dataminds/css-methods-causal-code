# -*- coding: utf-8 -*-
"""16장 · 조절 — 파이썬 엔진. (매개 · 경로 계산은 2026-11-15 17장 단위로 옮김)"""
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


# §16.1 ④ · §16.3 (2026-11-16 개편) = 단순기울기의 SE · 95% CI (refl 기준을 옮겨 다시 맞춘 hjs 계수) · ΔR² · F
def _ss():
    s = S()
    h, r = s.hjs - s.hjs.mean(), s.refl - s.refl.mean()
    sd, n = s.refl.std(ddof=1), len(s)
    tq = stats.t.ppf(.975, n - 4)
    out = {}
    for lab, v in (("lo", -sd), ("mid", 0.0), ("hi", sd)):
        rr = r - v
        b, se, *_ = ols(s.mil, [h, rr, h * rr])
        out[f"b_{lab}"], out[f"se_{lab}"] = b[1], se[1]
        out[f"cl_{lab}"], out[f"cu_{lab}"] = b[1] - tq * se[1], b[1] + tq * se[1]
    r2a = ols(s.mil, [h, r])[3]
    r2b = ols(s.mil, [h, r, h * r])[3]
    out.update(r2a=r2a, r2b=r2b, dr2=r2b - r2a, f=(r2b - r2a) / ((1 - r2b) / (n - 4)), df2=n - 4)
    pf = stats.f.sf(out["f"], 1, n - 4)
    out["p_lt001"] = .001 if pf < .001 else pf          # 「< .001」 = 부등식 주장
    return out


for k in ("b_lo", "b_mid", "b_hi", "se_lo", "se_mid", "se_hi", "cl_lo", "cu_lo", "cl_mid", "cu_mid",
          "cl_hi", "cu_hi", "r2a", "r2b", "dr2", "f", "df2", "p_lt001"):
    calc(f"ch16.ss_{k}")(lambda k=k: memo("ss", _ss)[k])


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
# §16.4 [다른 논문에서는] Miller & Krosnick (2000) 곱항 .31 (SE .18) 의 공식 구간 (이 책의 계산 ; 2026-11-15 14장에서 옮김)
calc("ch16.mk_ci_lo")(lambda: .31 - 1.96 * .18)
calc("ch16.mk_ci_hi")(lambda: .31 + 1.96 * .18)


# §16.2 「무엇을 중심화하나」 = 실험판 개입 × 기저 의미, 기저 의미를 중심화하지 않음 ⊥ 함 (2026-11-15)
def _ck():
    def f():
        e = E(); k = e.mil_t1 - e.mil_t1.mean()
        bu, _, pu, _ = ols(e.mil, [e.cond, e.mil_t1, e.cond * e.mil_t1])
        bc, _, pc, _ = ols(e.mil, [e.cond, k, e.cond * k])
        return bu, pu, bc, pc
    return memo("ck", f)


calc("ch16.ck_cond_raw")(lambda: _ck()[0][1])
calc("ch16.ck_p_raw")(lambda: _ck()[1][1])
calc("ch16.ck_cond_cen")(lambda: _ck()[2][1])
calc("ch16.ck_p_cen")(lambda: _ck()[3][1])
calc("ch16.ck_int_raw")(lambda: _ck()[0][3])
calc("ch16.ck_int_cen")(lambda: _ck()[2][3])


# 2026-11-15n §16.4 표 = 신뢰도 다섯 줄 전부 (pow10 · pow06 · pow07_int 는 위에서)
for _rel, _lab in ((0.9, "09"), (0.8, "08")):
    calc(f"ch16.pow{_lab}_main")(lambda rel=_rel: memo("pow", _power)[rel][0])
    calc(f"ch16.pow{_lab}_int")(lambda rel=_rel: memo("pow", _power)[rel][1])
calc("ch16.pow07_main")(lambda: memo("pow", _power)[0.7][0])


# 2026-11-15n 실습 4 = refl 기준을 4점 · −1SD 로 옮긴 모형 (hjs 는 평균 중심화)
def _cshift(v):
    s = S(); h = s.hjs - s.hjs.mean(); r = s.refl - v
    b, se, *_ = ols(s.mil, [h, r, h * r])
    return b, se


def _c4():
    return memo("c4", lambda: _cshift(4))


def _cm1():
    return memo("cm1", lambda: _cshift(S().refl.mean() - S().refl.std(ddof=1)))


calc("ch16.c4_b1")(lambda: _c4()[0][1])
calc("ch16.c4_b3")(lambda: _c4()[0][3])
calc("ch16.c4_se3")(lambda: _c4()[1][3])
calc("ch16.cm1_b1")(lambda: _cm1()[0][1])
calc("ch16.cm1_b3")(lambda: _cm1()[0][3])
calc("ch16.cm1_se3")(lambda: _cm1()[1][3])


# 2026-11-15p §16.2 「곱항을 넣기 전후의 계수 · 표준화 계수 β」
def _noint():
    s = S()
    return memo("noint", lambda: ols(s.mil, [s.hjs - s.hjs.mean(), s.refl - s.refl.mean()])[0])


_z = lambda v: (v - v.mean()) / v.std(ddof=1)


def _beta():
    def f():
        s = S()
        spss = ols(_z(s.mil), [_z(s.hjs), _z(s.refl), _z(s.hjs * s.refl)])[0]      # 원점수 곱을 만든 뒤 전부 표준화
        good = ols(_z(s.mil), [_z(s.hjs), _z(s.refl), _z(s.hjs) * _z(s.refl)])[0]  # 먼저 표준화하고 곱한다
        return spss, good
    return memo("beta", f)


calc("ch16.noint_b_hjs")(lambda: _noint()[1])
calc("ch16.noint_b_refl")(lambda: _noint()[2])
for _j, _nm in enumerate(("hjs", "refl", "xy"), start=1):
    calc(f"ch16.beta_spss_{_nm}")(lambda j=_j: _beta()[0][j])
    calc(f"ch16.beta_std_{_nm}")(lambda j=_j: _beta()[1][j])


# 2026-11-15q §16.1 [더 나아가기] 상자 = 집단을 나눠 돌린 기울기 차이 ⊥ 한 모형의 곱항 (코드 7)
def _grp():
    def f():
        s = S(); med = s.refl.median()
        lo, hi = s[s.refl <= med], s[s.refl > med]
        bl, sel, *_ = ols(lo.mil, [lo.hjs]); bh, seh, *_ = ols(hi.mil, [hi.hjs])
        d = bh[1] - bl[1]; sd = np.sqrt(sel[1] ** 2 + seh[1] ** 2)
        g = (s.refl > med).astype(int)
        bg, seg, pg, _ = ols(s.mil, [s.hjs, g, s.hjs * g])
        rl = lo.mil - (bl[0] + bl[1] * lo.hjs); rh = hi.mil - (bh[0] + bh[1] * hi.hjs)
        return dict(se_lo=sel[1], se_hi=seh[1], diff=d, sd_diff=sd, p_sep=2 * stats.norm.sf(abs(d / sd)),
                    b0=bg[0], b_hjs=bg[1], b_g=bg[2], b_int=bg[3], se0=seg[0], se_hjs=seg[1], se_g=seg[2],
                    se_int=seg[3], p_int=pg[3], rsd_lo=rl.std(ddof=1), rsd_hi=rh.std(ddof=1))
    return memo("grp", f)


for _k in ("se_lo", "se_hi", "diff", "sd_diff", "b0", "b_hjs", "b_g", "b_int", "se0", "se_hjs", "se_g", "se_int",
           "rsd_lo", "rsd_hi"):
    calc(f"ch16.grp_{_k}")(lambda k=_k: _grp()[k])
calc("ch16.grp_p_sep_lt001")(lambda: .001 if _grp()["p_sep"] < .001 else _grp()["p_sep"])
calc("ch16.grp_p_int_lt001")(lambda: .001 if _grp()["p_int"] < .001 else _grp()["p_int"])
calc("ch16.grp_rounded_sum")(lambda: 0.873 + 0.564)


# §16.1 요인 실험(2 × 2) 소절 (2026-11-16d) = 요인판 셀 평균 · 0/1 ⊥ 효과 부호화 · 제곱합 유형 I·II ⊥ III · 부분 η²
def _fac():
    f = load("fac")
    y, e, r = f.mil_t2.values, f.elem.values.astype(float), f.frame.values.astype(float)
    n = len(y)
    out = {"n": n, "df2": n - 4}
    for i in (0, 1):
        for j in (0, 1):
            c = f[(f.elem == i) & (f.frame == j)].mil_t2
            out[f"n{i}{j}"], out[f"m{i}{j}"], out[f"sd{i}{j}"] = len(c), c.mean(), c.std(ddof=1)
    b0, se0, p0, _ = ols(y, [e, r, e * r])
    ec, rc = e - .5, r - .5
    b1, se1, p1, _ = ols(y, [ec, rc, ec * rc])
    out.update(b01_elem=b0[1], b01_frame=b0[2], b01_int=b0[3], p01_elem=p0[1],
               bef_elem=b1[1], bef_frame=b1[2], bef_int=b1[3], t_int=b1[3] / se1[3])
    sse = lambda X: float(((y - np.column_stack([np.ones(n)] + X) @ np.linalg.lstsq(
        np.column_stack([np.ones(n)] + X), y, rcond=None)[0]) ** 2).sum())
    full = sse([ec, rc, ec * rc]); mse = full / (n - 4)
    for k, idx in (("elem", 1), ("frame", 2), ("int", 3)):
        t = b1[idx] / se1[idx]
        out[f"F3_{k}"] = t ** 2                                   # 유형 III = 효과 부호화 t²
        out[f"p3_{k}"] = 2 * stats.t.sf(abs(t), n - 4)
        ss3 = t ** 2 * mse
        out[f"eta_{k}"] = ss3 / (ss3 + full)                      # 부분 η² (유형 III 제곱합)
    out["F2_elem"] = (sse([rc]) - sse([ec, rc])) / mse            # 유형 II = 상호작용 없는 모형에서 frame 뒤에 넣은 elem
    out["se_frame_e1"] = out["m11"] - out["m10"]                  # 단순효과 = 성찰 있는 칸들에서 연결의 효과 (2026-11-19 §16.6 승격)
    out["se_frame_e0"] = out["m01"] - out["m00"]                  # 성찰 없는 칸들 (= 0/1 frame 계수)
    return out


for _k in ("n", "df2", "n00", "n01", "n10", "n11", "m00", "m01", "m10", "m11", "sd00", "sd01", "sd10", "sd11",
           "b01_elem", "b01_frame", "b01_int", "p01_elem", "bef_elem", "bef_frame", "bef_int", "t_int",
           "F3_elem", "F3_frame", "F3_int", "p3_elem", "p3_frame", "p3_int", "eta_elem", "eta_frame", "eta_int",
           "F2_elem", "se_frame_e1", "se_frame_e0"):
    calc(f"ch16.fac_{_k}")(lambda k=_k: memo("fac", _fac)[k])


# §16.4 「집단이 셋 이상이면」 (2026-11-16e) = 성별 세 범주(1 남 · 2 여 · 3 그 외 ; 기준 = 남) × hjs, 곱항 둘을 묶은 F
def _g3():
    s = S(); n = len(s); x = s.hjs.values
    d2, d3 = (s.gender == 2).astype(float).values, (s.gender == 3).astype(float).values
    r0 = ols(s.mil, [x, d2, d3])[3]
    b, se, p, r1 = ols(s.mil, [x, d2, d3, x * d2, x * d3])
    F = ((r1 - r0) / 2) / ((1 - r1) / (n - 6))
    out = dict(r2a=r0, r2b=r1, dr2=r1 - r0, F=F, p=stats.f.sf(F, 2, n - 6), df2=n - 6,
               int2=b[4], int3=b[5])
    for g in (1, 2, 3):
        t = s[s.gender == g]
        bb, ss, *_ = ols(t.mil, [t.hjs])
        tq = stats.t.ppf(.975, len(t) - 2)
        out.update({f"n{g}": len(t), f"b{g}": bb[1], f"se{g}": ss[1],
                    f"cl{g}": bb[1] - tq * ss[1], f"cu{g}": bb[1] + tq * ss[1]})
    return out


for _k in ("r2a", "r2b", "dr2", "F", "p", "df2", "int2", "int3", "n1", "n2", "n3", "b1", "b2", "b3",
           "se1", "se2", "se3", "cl1", "cu1", "cl2", "cu2", "cl3", "cu3"):
    calc(f"ch16.g3_{_k}")(lambda k=_k: memo("g3", _g3)[k])
