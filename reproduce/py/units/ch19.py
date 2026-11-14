# -*- coding: utf-8 -*-
"""19장 · 방법과 결과 절 쓰는 법 — 파이썬 엔진. 책 코드(`보고` · `구간`)와 같은 계산."""
import os
import sys

import numpy as np
from scipy import stats

from common import calc, cronbach, load, ols

EXP = lambda: load("exp")                 # 실험판 정제본 366명
EXP_RAW = lambda: load("exp", clean=False)
SVY = lambda: load("svy")                 # 설문판 정제본 566명
SVY_RAW = lambda: load("svy", clean=False)


def _report(x, y):
    """책의 `보고` 함수 = (평균1, SD1, 평균0, SD0, t, p, d, d 구간 아래, 위, sp)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    d = (x.mean() - y.mean()) / sp
    t = (x.mean() - y.mean()) / (sp * np.sqrt(1 / n1 + 1 / n2))
    df = n1 + n2 - 2
    p = 2 * stats.t.sf(abs(t), df)
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    return dict(m1=x.mean(), s1=x.std(ddof=1), m0=y.mean(), s0=y.std(ddof=1), t=t, p=p, d=d,
                lo=d - 1.96 * se_d, hi=d + 1.96 * se_d, sp=sp, n1=n1, n2=n2)


def _exp_rep(v):
    e = EXP()
    return _report(e[e.cond == 1][v].values, e[e.cond == 0][v].values)


for _v in ("mil", "hjs", "flr", "age", "mil_t1"):
    for _k in ("m1", "s1", "m0", "s0", "t", "p", "d", "lo", "hi"):
        calc(f"ch19.exp_{_v}_{_k}")(lambda v=_v, k=_k: _exp_rep(v)[k])


def _tcrit_exp():
    r = _exp_rep("mil")
    return stats.t.ppf(.975, r["n1"] + r["n2"] - 2)


def _mdiff():
    r = _exp_rep("mil")
    md = r["m1"] - r["m0"]
    se = r["sp"] * np.sqrt(1 / r["n1"] + 1 / r["n2"])
    return md, md - _tcrit_exp() * se, md + _tcrit_exp() * se


calc("ch19.mil_mdiff")(lambda: _mdiff()[0])
calc("ch19.mil_mdiff_lo")(lambda: _mdiff()[1])
calc("ch19.mil_mdiff_hi")(lambda: _mdiff()[2])
calc("ch19.exp_n_raw")(lambda: len(EXP_RAW()))
calc("ch19.exp_n_excl")(lambda: len(EXP_RAW()) - len(EXP()))
calc("ch19.exp_n")(lambda: len(EXP()))
calc("ch19.exp_df")(lambda: len(EXP()) - 2)
calc("ch19.hjs_n_items")(lambda: len([c for c in SVY().columns if c.startswith("hjs_")]))
calc("ch19.exp_n_treat")(lambda: int((EXP().cond == 1).sum()))
calc("ch19.exp_n_ctrl")(lambda: int((EXP().cond == 0).sum()))


def _chi_gender():
    import pandas as pd
    e = EXP()
    chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(e.cond, e.gender))
    return chi2, p, dof


calc("ch19.exp_gender_chi2")(lambda: _chi_gender()[0])
calc("ch19.exp_gender_chi2_p")(lambda: _chi_gender()[1])
calc("ch19.exp_gender_chi2_df")(lambda: _chi_gender()[2])


def _ancova():
    e = EXP()
    b, se, p, _ = ols(e.mil, [e.cond, e.mil_t1])
    return b[1], se[1], p[1]


calc("ch19.ancova_b")(lambda: _ancova()[0])
calc("ch19.ancova_se")(lambda: _ancova()[1])
calc("ch19.ancova_p")(lambda: _ancova()[2])


def _inter():
    e = EXP()
    k = e.mil_t1 - e.mil_t1.mean()
    b, se, p, _ = ols(e.mil.values, [e.cond.values, k.values, (e.cond * k).values])
    return b[3], p[3]


calc("ch19.inter_b")(lambda: _inter()[0])
calc("ch19.inter_p")(lambda: _inter()[1])


# §19.4 오차 막대 = 개입 집단 사후 mil ; tcrit 은 §19.3 상자의 t(364)
def _bars():
    e = EXP()
    x, y = e[e.cond == 1].mil.values, e[e.cond == 0].mil.values
    tc = _tcrit_exp()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    return sx, sx / np.sqrt(len(x)), tc * sx / np.sqrt(len(x)), x.mean() - tc * sx / np.sqrt(len(x)), \
        y.mean() + tc * sy / np.sqrt(len(y))


calc("ch19.bar_sd")(lambda: _bars()[0])
calc("ch19.bar_se")(lambda: _bars()[1])
calc("ch19.bar_ci_half")(lambda: _bars()[2])
calc("ch19.bar_treat_lo")(lambda: _bars()[3])
calc("ch19.bar_ctrl_hi")(lambda: _bars()[4])

# 설문판 표본 흐름 · 척도
calc("ch19.svy_n_raw")(lambda: len(SVY_RAW()))
calc("ch19.svy_n_excl")(lambda: len(SVY_RAW()) - len(SVY()))
calc("ch19.svy_excl_pct")(lambda: (len(SVY_RAW()) - len(SVY())) / len(SVY_RAW()) * 100)
calc("ch19.svy_n")(lambda: len(SVY()))
calc("ch19.hjs_alpha")(lambda: cronbach(SVY()[[c for c in SVY().columns if c.startswith("hjs_")]]))
calc("ch19.r_hjs_mil")(lambda: SVY().hjs.corr(SVY().mil))


# S3 위계 회귀 (성별 1 · 2 의 554명)
def _s3():
    s = SVY()[SVY().gender.isin([1, 2])]
    fem = (s.gender == 2).astype(float).values
    h = (s.hjs - s.hjs.mean()).values
    r = (s.refl - s.refl.mean()).values
    y = s.mil.values
    X1 = [s.age.values, fem]
    X2 = [s.age.values, fem, s.hjs.values]
    X3 = [s.age.values, fem, h, r, h * r]
    o1, o2, o3 = ols(y, X1), ols(y, X2), ols(y, X3)
    t2 = stats.t.ppf(.975, len(y) - len(X2) - 1)
    t3 = stats.t.ppf(.975, len(y) - len(X3) - 1)
    sd_r = s.refl.std(ddof=1)
    return dict(n=len(s), dr2_2=o2[3] - o1[3], dr2_3=o3[3] - o2[3],
                b2=o2[0][3], se2=o2[1][3], lo2=o2[0][3] - t2 * o2[1][3], hi2=o2[0][3] + t2 * o2[1][3], p2=o2[2][3],
                b3=o3[0][5], se3=o3[1][5], lo3=o3[0][5] - t3 * o3[1][5], hi3=o3[0][5] + t3 * o3[1][5], p3=o3[2][5],
                slo=o3[0][3] - o3[0][5] * sd_r, shi=o3[0][3] + o3[0][5] * sd_r)


for _k in ("n", "dr2_2", "dr2_3", "b2", "se2", "lo2", "hi2", "b3", "se3", "lo3", "hi3", "slo", "shi"):
    calc(f"ch19.s3_{_k}")(lambda k=_k: _s3()[k])
calc("ch19.s3_dr2_2_pct")(lambda: _s3()["dr2_2"] * 100)
# 「p < .001」 문턱 주장 = 문턱 아래면 .001, 아니면 −1 (실패 ; R 의 NA 는 verify.py 결과 파서가 읽지 못한다)
calc("ch19.s3_p2_below_001")(lambda: .001 if _s3()["p2"] < .001 else -1.0)
calc("ch19.s3_p3_below_001")(lambda: .001 if _s3()["p3"] < .001 else -1.0)


# §19.3 동등성 검정 = 설문판 성별 1 · 2 (차이 = 2 − 1 ; 경계 d ±0.2 를 합동 SD 로)
def _eq(v):
    g = SVY()[SVY().gender.isin([1, 2])]
    a, b = g[g.gender == 1][v], g[g.gender == 2][v]
    na, nb = len(a), len(b)
    df = na + nb - 2
    sp = np.sqrt(((na - 1) * a.var() + (nb - 1) * b.var()) / df)
    dd = b.mean() - a.mean()
    se = sp * np.sqrt(1 / na + 1 / nb)
    t95, t90 = stats.t.ppf(.975, df), stats.t.ppf(.95, df)
    return dict(diff=dd, lo95=dd - t95 * se, hi95=dd + t95 * se, p=2 * stats.t.sf(abs(dd / se), df),
                bound=.2 * sp, lo90=dd - t90 * se, hi90=dd + t90 * se, n=na + nb)


for _v in ("mil", "flr"):
    for _k in ("diff", "lo95", "hi95", "p", "bound", "lo90", "hi90"):
        calc(f"ch19.eq_{_v}_{_k}")(lambda v=_v, k=_k: _eq(v)[k])
calc("ch19.eq_n")(lambda: _eq("mil")["n"])
calc("ch19.eq_flr_neg_bound")(lambda: -_eq("flr")["bound"])


def _gender_max_abs_d():
    g = SVY()[SVY().gender.isin([1, 2])]
    return max(abs(_report(g[g.gender == 1][v].values, g[g.gender == 2][v].values)["d"]) for v in ("mil", "hjs", "flr"))


# 실습 4 「세 변수 모두 d 가 0.1 미만」 = 문턱 주장
calc("ch19.gender_d_below_01")(lambda: 0.1 if _gender_max_abs_d() < 0.1 else -1.0)

# S1 ② 검정력 = 집단당 190명 · d 0.22 · 양측 .05 (비중심 t)
calc("ch19.s1_power")(lambda: (lambda df, c, nc: 1 - stats.nct.cdf(c, df, nc) + stats.nct.cdf(-c, df, nc))(
    378, stats.t.ppf(.975, 378), .22 * np.sqrt(190 / 2)))


# S2 상호작용 (요인판 정제본) = b ± 1.96 SE (S2 와 같은 구간)
def _s2():
    f = load("fac")
    b, se, p, _ = ols(f.mil_t2.values, [f.elem.values, f.frame.values, (f.elem * f.frame).values])
    return b[3], b[3] - 1.96 * se[3], b[3] + 1.96 * se[3]


calc("ch19.s2_b3")(lambda: _s2()[0])
calc("ch19.s2_b3_lo")(lambda: _s2()[1])
calc("ch19.s2_b3_hi")(lambda: _s2()[2])

# 「먼저」 상자 산술 (Rogers 외 2023 연구 6 보고값)
_T445 = lambda: stats.t.ppf(.975, 445)
calc("ch19.box_diff_rp")(lambda: 5.73 - 5.46)
calc("ch19.box_se_rp")(lambda: (5.73 - 5.46) / 1.68)
calc("ch19.box_lo_rp")(lambda: (5.73 - 5.46) - _T445() * (5.73 - 5.46) / 1.68)
calc("ch19.box_hi_rp")(lambda: (5.73 - 5.46) + _T445() * (5.73 - 5.46) / 1.68)
calc("ch19.box_diff_he")(lambda: 5.73 - 5.49)
calc("ch19.box_lo_he")(lambda: (5.73 - 5.49) - _T445() * (5.73 - 5.49) / 1.48)
calc("ch19.box_hi_he")(lambda: (5.73 - 5.49) + _T445() * (5.73 - 5.49) / 1.48)
calc("ch19.box_t_main")(lambda: .54 / .16)

# 모의시행
_cache = {}


def _sims():
    if _cache:
        return _cache
    e = EXP()
    # S1 ⑦ 뒤섞기 = 씨앗 73 · 5,000번 (S1 코드 순서)
    rng = np.random.default_rng(73)
    t_, c_ = e[e.cond == 1], e[e.cond == 0]
    obs = t_.mil.mean() - c_.mil.mean()
    diffs = []
    for _ in range(5000):
        fake = rng.permutation(e.cond.values)
        diffs.append(e.mil.values[fake == 1].mean() - e.mil.values[fake == 0].mean())
    _cache["perm_p"] = float((np.abs(np.array(diffs)) >= abs(obs)).mean())
    # 16장 §16.1 = 참 기울기 0.3, 300명 다음 40명 (씨앗 73) ; 40명의 p
    g = np.random.default_rng(73)
    res = []
    for nn in (300, 40):
        xx = g.normal(0, 1, nn)
        yy = .3 * xx + g.normal(0, 1, nn)
        res.append(ols(yy, [xx])[2][1])
    _cache["p40"] = float(res[1])
    # S2 상호작용 검정력 = 요인판 생성기로 총 448명 400번 (씨앗 1000 + i ; S2 표와 같은 절차)
    mat = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if mat not in sys.path:
        sys.path.insert(0, mat)
    from make_data_family import gen_fac
    hits = 0
    for i in range(400):
        f = gen_fac(np.random.default_rng(1000 + i), n=448)
        f = f[f.attn_1 == 1]
        _, _, pp, _ = ols(f.mil_t2.values, [f.elem.values, f.frame.values, (f.elem * f.frame).values])
        hits += pp[3] < .05
    _cache["s2_power"] = hits / 400
    # 13장 §13.1 부트스트랩 = 평균 차이 구간 [0.03, 0.54] 의 출처 (씨앗 73 · 5,000번 · 개입 먼저 뽑고 통제)
    tv, cv = t_.mil.values, c_.mil.values
    rng = np.random.default_rng(73)
    bd = np.array([rng.choice(tv, len(tv), replace=True).mean() - rng.choice(cv, len(cv), replace=True).mean()
                   for _ in range(5000)])
    _cache["boot_lo"], _cache["boot_hi"] = np.percentile(bd, [2.5, 97.5])
    return _cache


calc("ch19.s1_perm_p")(lambda: _sims()["perm_p"])
calc("ch19.ch16_p40")(lambda: _sims()["p40"])
calc("ch19.s2_power448")(lambda: _sims()["s2_power"])
calc("ch19.boot_mdiff_lo")(lambda: _sims()["boot_lo"])
calc("ch19.boot_mdiff_hi")(lambda: _sims()["boot_hi"])

# 2026-11-14h 평균 차이의 공식(t) 구간
def _tci():
    import numpy as _np
    from scipy import stats as _st
    from common import load as _load
    _e = _load("exp"); _t = _e[_e.cond == 1].mil.values; _c = _e[_e.cond == 0].mil.values
    _n1, _n2 = len(_t), len(_c)
    _sp = _np.sqrt(((_n1 - 1) * _t.var(ddof=1) + (_n2 - 1) * _c.var(ddof=1)) / (_n1 + _n2 - 2))
    _se = _sp * _np.sqrt(1 / _n1 + 1 / _n2); _q = _st.t.ppf(.975, _n1 + _n2 - 2); _d = _t.mean() - _c.mean()
    return _d - _q * _se, _d + _q * _se
calc("ch19.diff_ci_t_lo")(lambda: _tci()[0])
calc("ch19.diff_ci_t_hi")(lambda: _tci()[1])
calc("ch19.diff_ci_hi_gap")(lambda: 0.54 - round(_tci()[1], 2))
