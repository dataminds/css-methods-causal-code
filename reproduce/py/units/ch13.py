# -*- coding: utf-8 -*-
"""13장 · 신뢰구간 · 효과크기 · 검정력 — 파이썬 엔진.

모의시행은 책 코드와 같은 난수 순서를 따른다(블록마다 default_rng(73) 을 새로 만든다).
"""
import numpy as np
from scipy import optimize, stats

from common import calc, cohen_d, load

E = lambda: load("exp")
RAW = lambda: load("exp", clean=False)
T = lambda: E()[E().cond == 1].mil
C = lambda: E()[E().cond == 0].mil

# ── 자료 계산 ──────────────────────────────────────────────
calc("ch13.n_clean")(lambda: len(E()))
calc("ch13.n_raw")(lambda: len(RAW()))
calc("ch13.n_t")(lambda: len(T()))
calc("ch13.n_c")(lambda: len(C()))
calc("ch13.n_assigned_t")(lambda: int((RAW().cond == 1).sum()))
calc("ch13.obs")(lambda: T().mean() - C().mean())
calc("ch13.sd_t")(lambda: T().std(ddof=1))
calc("ch13.sd_c")(lambda: C().std(ddof=1))
_se = lambda: np.sqrt(T().var(ddof=1) / len(T()) + C().var(ddof=1) / len(C()))
calc("ch13.se")(_se)
calc("ch13.se_rounded_sd")(lambda: np.sqrt(1.18 ** 2 / 183 + 1.27 ** 2 / 183))
calc("ch13.ci_lo")(lambda: T().mean() - C().mean() - 1.96 * _se())
calc("ch13.ci_hi")(lambda: T().mean() - C().mean() + 1.96 * _se())
calc("ch13.obs_over_se")(lambda: (T().mean() - C().mean()) / _se())
calc("ch13.p_t")(lambda: stats.ttest_ind(T(), C()).pvalue)
calc("ch13.d")(lambda: cohen_d(T(), C()))
calc("ch13.ps_pct")(lambda: 100 * stats.norm.cdf(0.23 / np.sqrt(2)))     # 본문이 쓴 d 0.23 그대로
calc("ch13.z975")(lambda: stats.norm.ppf(.975))
calc("ch13.t975_49")(lambda: stats.t.ppf(.975, 49))
calc("ch13.pop_mean")(lambda: load("svy").mil.mean())

# §13.3 효과크기 환산 (두 집단 크기 같음 · 정규)
_r = lambda d: d / np.sqrt(d ** 2 + 4)
for d in (0.2, 0.5, 0.8):
    k = str(d).replace(".", "")
    calc(f"ch13.r_{k}")(lambda d=d: _r(d))
    calc(f"ch13.eta2_{k}")(lambda d=d: _r(d) ** 2)
    calc(f"ch13.or_{k}")(lambda d=d: np.exp(d * np.pi / np.sqrt(3)))
calc("ch13.eta2_08_pct")(lambda: 100 * _r(0.8) ** 2)
_d_from_r = lambda r: 2 * r / np.sqrt(1 - r ** 2)
calc("ch13.d_from_r30")(lambda: _d_from_r(.30))
calc("ch13.d_from_r63")(lambda: _d_from_r(.63))


# 두 집단 t 검정의 검정력 (비중심 t ; 집단당 m 명 · 양측 .05)
def pw(d, m):
    df_ = 2 * m - 2
    c_ = stats.t.ppf(.975, df_)
    nc_ = d * np.sqrt(m / 2)
    return stats.nct.sf(c_, df_, nc_) + stats.nct.cdf(-c_, df_, nc_)


# 「먼저」 절 = Rogers et al. (2023) 연구 5 보고값에서 한 산술
calc("ch13.rg_diff")(lambda: 5.14 - 4.83)
calc("ch13.rg_sd_pooled")(lambda: np.sqrt((1.40 ** 2 + 1.42 ** 2) / 2))
calc("ch13.rg_ps_pct")(lambda: 100 * stats.norm.cdf(.22 / np.sqrt(2)))
calc("ch13.rg_se")(lambda: (5.14 - 4.83) / 2.16)
calc("ch13.rg_ci_lo")(lambda: .31 - stats.t.ppf(.975, 378) * .31 / 2.16)
calc("ch13.rg_ci_hi")(lambda: .31 + stats.t.ppf(.975, 378) * .31 / 2.16)
# 가장 작은 집단 크기 = d .44 를 검정력 .80 으로 잡는 최소 집단당 n
calc("ch13.rg_n_for_44")(lambda: next(m for m in range(10, 500) if pw(.44, m) >= .80))
calc("ch13.rg_pw_44_83")(lambda: pw(.44, 83))
calc("ch13.rg_n_per")(lambda: (378 + 2) / 2)
calc("ch13.rg_n_total")(lambda: 378 + 2)
calc("ch13.rg_mdes_190")(lambda: optimize.brentq(lambda d: pw(d, 190) - .80, .05, .8))
calc("ch13.pw_22_190")(lambda: pw(.22, 190))
calc("ch13.pw_22_183")(lambda: pw(.22, 183))
# 「.80에 닿으려면 집단당 300명 넘게」 = 부등식 주장. 필요한 n 이 300 을 넘으면 300 을, 아니면 실제 n 을 돌려준다.
calc("ch13.n_for_80_gt300")(lambda: (lambda n: 300.0 if n > 300 else float(n))(
    next(m for m in range(10, 2000) if pw(.22, m) >= .80)))


# §13.4 상자 = 신뢰도 .7 에서 검정력 .57(집단당 190명 · 참 d .22)을 되찾는 집단당 n
_d07 = lambda: .22 * np.sqrt(.7)
_n07 = lambda: next(m for m in range(190, 2000) if pw(_d07(), m) >= pw(.22, 190))
calc("ch13.rel07_d")(_d07)
calc("ch13.rel07_n")(_n07)
# 「160명 넘게 더」 = 부등식 주장 (두 집단 합).
calc("ch13.rel07_extra_gt160")(lambda: (lambda x: 160.0 if x > 160 else float(x))(2 * (_n07() - 190)))

# ── 모의시행 ─────────────────────────────────────────────
_cache: dict = {}


def _boot():
    if "boot" not in _cache:
        tv, cv = T().values, C().values
        rng = np.random.default_rng(73)
        _cache["boot"] = np.array([rng.choice(tv, len(tv), replace=True).mean()
                                   - rng.choice(cv, len(cv), replace=True).mean()
                                   for _ in range(5000)])
    return _cache["boot"]


calc("ch13.boot_lo")(lambda: np.percentile(_boot(), 2.5))
calc("ch13.boot_hi")(lambda: np.percentile(_boot(), 97.5))
calc("ch13.boot_mean")(lambda: _boot().mean())
calc("ch13.boot_sd")(lambda: _boot().std(ddof=1))
calc("ch13.boot_width")(lambda: np.percentile(_boot(), 97.5) - np.percentile(_boot(), 2.5))


def _cover(crit):
    pop = load("svy")["mil"].values
    truth = pop.mean()
    rng = np.random.default_rng(73)
    hit = 0
    for _ in range(100):
        s = rng.choice(pop, size=50, replace=False)
        half = crit * s.std(ddof=1) / np.sqrt(50)
        if s.mean() - half <= truth <= s.mean() + half:
            hit += 1
    return hit


calc("ch13.cover")(lambda: _cover(1.96))
calc("ch13.cover_miss")(lambda: 100 - _cover(1.96))
calc("ch13.cover_t")(lambda: _cover(stats.t.ppf(.975, 49)))


def _power_table():
    if "pt" not in _cache:
        rng = np.random.default_rng(73)
        out = {}
        for n in (50, 100, 190, 300, 500, 800):
            hits = sum(stats.ttest_ind(rng.standard_normal(n),
                                       rng.standard_normal(n) + 0.22).pvalue < .05
                       for _ in range(2000))
            out[n] = hits / 2000
        _cache["pt"] = out
    return _cache["pt"]


for n in (50, 100, 190, 300, 500, 800):
    calc(f"ch13.power_{n}")(lambda n=n: _power_table()[n])
calc("ch13.power_190_pct")(lambda: 100 * _power_table()[190])


def _inverse():
    if "inv" not in _cache:
        out = {}
        for n in (80, 90, 100, 110):
            rng = np.random.default_rng(73)
            hits = sum(stats.ttest_ind(rng.standard_normal(n),
                                       rng.standard_normal(n) + 0.4).pvalue < .05
                       for _ in range(2000))
            out[n] = hits / 2000
        _cache["inv"] = out
    return _cache["inv"]


calc("ch13.inv_first_n80")(lambda: next(n for n in (80, 90, 100, 110) if _inverse()[n] > .80))


def _noise():
    if "noise" not in _cache:
        out = {}
        for rel in (1.00, 0.90, 0.70):
            rng = np.random.default_rng(73)
            noise = np.sqrt((1 - rel) / rel) if rel < 1 else 0.0
            hits = 0
            for _ in range(2000):
                a = rng.normal(0.22, 1, 190); b = rng.normal(0, 1, 190)
                if noise:
                    a = a + rng.normal(0, noise, 190)
                    b = b + rng.normal(0, noise, 190)
                hits += stats.ttest_ind(a, b).pvalue < .05
            out[rel] = hits / 2000
        _cache["noise"] = out
    return _cache["noise"]


calc("ch13.noise_10")(lambda: _noise()[1.00])
calc("ch13.noise_07")(lambda: _noise()[0.70])
