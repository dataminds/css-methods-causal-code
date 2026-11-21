# -*- coding: utf-8 -*-
"""12장 · 가설검정 — 파이썬 엔진.

모의시행은 책 코드(와 구 배터리 materials/check_book_claims.py 의 같은 절차)와 같은 난수 순서를 따른다.
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from common import calc, cohen_d, load

E = lambda: load("exp")                  # 정제본 366명
RAW = lambda: load("exp", clean=False)   # 모집 380명
T = lambda: E()[E().cond == 1].mil       # 개입 183명
C = lambda: E()[E().cond == 0].mil       # 통제 183명

# ── 자료 계산 ──────────────────────────────────────────────
calc("ch12.n_raw")(lambda: len(RAW()))
calc("ch12.n_clean")(lambda: len(E()))
calc("ch12.n_excl")(lambda: len(RAW()) - len(E()))
calc("ch12.n_t")(lambda: len(T()))
calc("ch12.n_c")(lambda: len(C()))
calc("ch12.m_t")(lambda: T().mean())
calc("ch12.m_c")(lambda: C().mean())
calc("ch12.obs")(lambda: T().mean() - C().mean())
# 「0.2827868…」 = 말줄임표로 끊어 쓴 값(반올림하면 0.2827869). 일곱째 자리에서 버린 값으로 대조한다.
calc("ch12.obs_trunc7")(lambda: np.floor((T().mean() - C().mean()) * 1e7) / 1e7)
calc("ch12.sd_t")(lambda: T().std(ddof=1))
calc("ch12.sd_c")(lambda: C().std(ddof=1))
calc("ch12.se")(lambda: np.sqrt(T().var(ddof=1) / len(T()) + C().var(ddof=1) / len(C())))
calc("ch12.obs_over_se")(lambda: (T().mean() - C().mean())
                         / np.sqrt(T().var(ddof=1) / len(T()) + C().var(ddof=1) / len(C())))
calc("ch12.t")(lambda: stats.ttest_ind(T(), C()).statistic)
calc("ch12.p_t")(lambda: stats.ttest_ind(T(), C()).pvalue)
calc("ch12.welch_t")(lambda: stats.ttest_ind(T(), C(), equal_var=False).statistic)
calc("ch12.welch_p")(lambda: stats.ttest_ind(T(), C(), equal_var=False).pvalue)
calc("ch12.df")(lambda: len(E()) - 2)
calc("ch12.tcrit364")(lambda: stats.t.ppf(.975, len(E()) - 2))
calc("ch12.tcrit200")(lambda: stats.t.ppf(.975, 200))
calc("ch12.d")(lambda: cohen_d(T(), C()))
calc("ch12.sval_p")(lambda: -np.log2(stats.ttest_ind(T(), C()).pvalue))
calc("ch12.sval_05")(lambda: -np.log2(.05))


def _chi(correction=False):
    e = E().assign(high=(E().mil >= E().mil.median()).astype(int))
    return stats.chi2_contingency(pd.crosstab(e.cond, e.high), correction=correction)


calc("ch12.chi2")(lambda: _chi()[0])
calc("ch12.chi_p")(lambda: _chi()[1])
calc("ch12.chi_df")(lambda: _chi()[2])
calc("ch12.chi_p_corr")(lambda: _chi(True)[1])
# 「가장 작은 칸의 기대빈도가 90을 넘어」 = 부등식 주장. 넘으면 90 을, 아니면 실제 값을 돌려준다(그러면 대조가 실패한다).
calc("ch12.exp_min_gt90")(lambda: 90.0 if _chi()[3].min() > 90 else float(_chi()[3].min()))

# §12.1 그림 = 정규 근사 · 양측 · d = 0.22 (13장 §13.4 의 참값). 2종 오류 = 검정통계가 문턱 안에 떨어질 확률.
_D = 0.22


def _beta(n, a):
    se = np.sqrt(2 / n)                      # 표준편차 단위의 두 평균 차이 표준오차
    c = stats.norm.ppf(1 - a / 2) * se
    return stats.norm.cdf((c - _D) / se) - stats.norm.cdf((-c - _D) / se)


def _sp():
    t, c = T(), C()
    return np.sqrt(((len(t) - 1) * t.var(ddof=1) + (len(c) - 1) * c.var(ddof=1)) / (len(t) + len(c) - 2))


calc("ch12.eff_pts")(lambda: _D * _sp())
calc("ch12.beta_183_10")(lambda: _beta(183, .10))
calc("ch12.beta_183_05")(lambda: _beta(183, .05))
calc("ch12.beta_183_01")(lambda: _beta(183, .01))
calc("ch12.beta_366_05")(lambda: _beta(366, .05))
calc("ch12.power_183_05")(lambda: 1 - _beta(183, .05))

# §12.4 기저율 상자 = 검정력 .56(본문이 쓴 반올림 값) · 기준 .05 · 1,000건
for br in (10, 25, 50):
    calc(f"ch12.br{br}_true")(lambda br=br: 1000 * br / 100 * .56)
    calc(f"ch12.br{br}_false")(lambda br=br: 1000 * (1 - br / 100) * .05)
    calc(f"ch12.br{br}_ppv")(lambda br=br: 100 * (1000 * br / 100 * .56)
                             / (1000 * br / 100 * .56 + 1000 * (1 - br / 100) * .05))
calc("ch12.br10_n_eff")(lambda: 1000 * .10)
calc("ch12.br10_n_null")(lambda: 1000 * .90)
calc("ch12.br10_n_sig")(lambda: 1000 * .10 * .56 + 1000 * .90 * .05)

# 「먼저」 절 = Rogers et al. (2023) 연구 5 보고값에서 한 산술
calc("ch12.rogers_df")(lambda: 381 - 2)
calc("ch12.rogers_p_pct")(lambda: .031 * 100)

# §12.3 INWORKS 상자 = 원문 90% 구간에서 표준오차를 되짚은 정규 근사
_INW = {"all": (.52, .27, .77), "mid": (1.12, .45, 1.80), "low": (1.38, .20, 2.60)}


def _inw(k, what):
    e, lo, hi = _INW[k]
    s = (hi - lo) / (2 * stats.norm.ppf(.95))
    return {"p1": stats.norm.sf(e / s), "p2": 2 * stats.norm.sf(e / s),
            "lo95": e - stats.norm.ppf(.975) * s, "hi95": e + stats.norm.ppf(.975) * s}[what]


for k in _INW:
    for w in ("p1", "p2", "lo95", "hi95"):
        calc(f"ch12.inw_{k}_{w}")(lambda k=k, w=w: _inw(k, w))

# 실습 2 · 3 = 자료 계산
SV = lambda: load("svy")
calc("ch12.men_mil")(lambda: SV()[SV().gender == 1].mil.mean())
calc("ch12.women_mil")(lambda: SV()[SV().gender == 2].mil.mean())
calc("ch12.mw_diff")(lambda: SV()[SV().gender == 1].mil.mean() - SV()[SV().gender == 2].mil.mean())
calc("ch12.gender3_n")(lambda: int((SV().gender == 3).sum()))
calc("ch12.extreme_diff")(lambda: T().mean() + 2 - C().mean())


# ── 모의시행 (책 코드와 같은 난수 순서) ────────────────────────
_cache: dict = {}


def _perm(seed=73, y=None, c=None, g1=1, g0=0):
    """§12.2 뒤섞기 = default_rng(seed) 로 딱지만 5,000번 섞는다."""
    key = ("perm", seed, g1, g0, None if y is None else float(np.sum(y)))
    if key not in _cache:
        if y is None:
            y, c = E().mil.values, E().cond.values
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(5000):
            fake = rng.permutation(c)
            out.append(y[fake == g1].mean() - y[fake == g0].mean())
        _cache[key] = np.array(out)
    return _cache[key]


def _obs():
    return T().mean() - C().mean()


D = lambda: _perm()
calc("ch12.perm_mean")(lambda: D().mean())
calc("ch12.perm_sd")(lambda: D().std(ddof=1))
calc("ch12.perm_lo")(lambda: np.percentile(D(), 2.5))
calc("ch12.perm_hi")(lambda: np.percentile(D(), 97.5))
calc("ch12.perm_p")(lambda: (np.abs(D()) >= abs(_obs())).mean())
calc("ch12.perm_p_pct")(lambda: 100 * (np.abs(D()) >= abs(_obs())).mean())
calc("ch12.perm_not_p_pct")(lambda: 100 * (1 - (np.abs(D()) >= abs(_obs())).mean()))
calc("ch12.perm_count")(lambda: int((np.abs(D()) >= abs(_obs())).sum()))
calc("ch12.perm_one_count")(lambda: int((D() >= _obs()).sum()))
calc("ch12.perm_one_p")(lambda: (D() >= _obs()).mean())
calc("ch12.perm_rev_one_p")(lambda: (D() >= -_obs()).mean())   # 결과가 반대(−0.283)였을 때 개입 쪽만 센 p
calc("ch12.perm_round_count")(lambda: int((np.abs(D()) >= 0.283).sum()))
calc("ch12.perm_round_p")(lambda: (np.abs(D()) >= 0.283).mean())
calc("ch12.perm_ge01_pct")(lambda: 100 * (np.abs(D()) >= 0.1).mean())


# 실습 2 = 설문판 남(1) · 여(2) 뒤섞기 (씨앗 73 ; 구 배터리 절차)
def _gender_p():
    s = SV()[SV().gender.isin([1, 2])]
    d = _perm(73, s.mil.values, s.gender.values, 1, 2)
    o = s.mil[s.gender == 1].mean() - s.mil[s.gender == 2].mean()
    return (np.abs(d) >= abs(o)).mean()


calc("ch12.gender_perm_p")(_gender_p)


# 실습 3 = 개입 집단 mil + 2 (씨앗 73)
def _extreme_p():
    e = E()
    y = e.mil.values + 2 * (e.cond.values == 1)
    d = _perm(73, y, e.cond.values)
    return (np.abs(d) >= abs(y[e.cond.values == 1].mean() - y[e.cond.values == 0].mean())).mean()


calc("ch12.extreme_p")(_extreme_p)
calc("ch12.extreme_count")(lambda: 5000 * _extreme_p())

# §12.8 재실행 = 씨앗 73 · 37 · 2026
for s in (73, 37, 2026):
    calc(f"ch12.perm_p_seed{s}")(lambda s=s: (np.abs(_perm(s)) >= abs(_obs())).mean())


# §12.4 = 씨앗 37 로 100명씩 스무 번
def _subs():
    if "subs" not in _cache:
        e = E()
        rng2 = np.random.default_rng(37)
        ps, n1s, idx = [], [], []
        for _ in range(20):
            ii = rng2.choice(len(e), size=100, replace=False)
            sub = e.iloc[ii]
            ps.append(stats.ttest_ind(sub.mil[sub.cond == 1], sub.mil[sub.cond == 0]).pvalue)
            n1s.append(int((sub.cond == 1).sum()))
            idx.append(set(ii.tolist()))
        ov = np.mean([len(a & b) for a, b in itertools.combinations(idx, 2)])
        _cache["subs"] = dict(pmin=min(ps), pmax=max(ps), nsig=sum(p < .05 for p in ps), ov=ov,
                              n1min=min(n1s), n1max=max(n1s), never=len(e) - len(set().union(*idx)))
    return _cache["subs"]


for k in ("pmin", "pmax", "nsig", "ov", "n1min", "n1max", "never"):
    calc(f"ch12.sub_{k}")(lambda k=k: _subs()[k])


# 11장 §11.3 의 세 표본 평균 (12장 §12.0 이 다시 인용 ; 씨앗 73 · 50명 · 중복 없이)
def _three():
    if "three" not in _cache:
        pop = load("svy")["mil"].values
        rng = np.random.default_rng(73)
        _cache["three"] = [rng.choice(pop, size=50, replace=False).mean() for _ in range(3)]
    return _cache["three"]


for i in range(3):
    calc(f"ch12.ch11_sample{i + 1}")(lambda i=i: _three()[i])


# §12.5 상자 = 이분화의 검정력 손실 (d 0.23 · 집단당 183명 · 2,000번 · 씨앗 73 ; 구 배터리 절차)
def _dich():
    if "dich" not in _cache:
        rdi = np.random.default_rng(73)
        ht = hc = 0
        for _ in range(2000):
            x1 = rdi.normal(0.23, 1, 183); x0 = rdi.normal(0, 1, 183)
            ht += stats.ttest_ind(x1, x0).pvalue < .05
            med = np.median(np.concatenate([x1, x0]))
            tb = np.array([[np.sum(x0 < med), np.sum(x0 >= med)],
                           [np.sum(x1 < med), np.sum(x1 >= med)]])
            hc += stats.chi2_contingency(tb, correction=False)[1] < .05
        _cache["dich"] = (100 * ht / 2000, 100 * hc / 2000)
    return _cache["dich"]


calc("ch12.dich_t_pct")(lambda: _dich()[0])
calc("ch12.dich_chi_pct")(lambda: _dich()[1])
calc("ch12.dich_gap_pct")(lambda: _dich()[0] - _dich()[1])


# §12.5 상자 = 큰 분산이 작은 집단(SD 3 · 20명 대 SD 1 · 60명)일 때 등분산 t 의 위양성률 (14장 §14.4 코드 그대로)
def _welch_world():
    g = np.random.default_rng(73)
    hits = 0
    for _ in range(2000):
        x, y = g.normal(0, 3, 20), g.normal(0, 1, 60)
        hits += stats.ttest_ind(x, y).pvalue < .05
    return 100 * hits / 2000


calc("ch12.fp_bigvar_small_pct")(_welch_world)


# §12.8 = 결과변수 셋 가운데 잘 나온 것을 고르는 위양성률 (구 배터리 절차 그대로 ; 씨앗 73 · 100명 · 3,000번)
def _three_outcomes():
    rpre = np.random.default_rng(73)
    hit3 = 0
    for _ in range(3000):
        g1 = rpre.normal(0, 1, 100); g0 = rpre.normal(0, 1, 100)
        y2 = rpre.normal(0, 1, 100); y3 = rpre.normal(0, 1, 100)
        ps3 = [stats.ttest_ind(g1, g0).pvalue, stats.ttest_ind(y2, y3).pvalue,
               stats.ttest_ind(g1, y2).pvalue]
        hit3 += min(ps3) < .05
    return 100 * hit3 / 3000


calc("ch12.fp_three_outcomes_pct")(_three_outcomes)

# 2026-11-14h 원고 정정 = 독립 결과변수 셋에서 하나라도 p < .05 일 확률 (이론값)
calc("ch12.fp_three_indep_pct")(lambda: 100 * (1 - 0.95 ** 3))
calc("ch12.fp_three_indep_frac")(lambda: 1 - 0.95 ** 3)
