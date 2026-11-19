# -*- coding: utf-8 -*-
"""B1 · 확률과 분포 — 파이썬 엔진 (책 코드와 같은 난수 순서 ; 씨앗 73)."""
import numpy as np
from scipy import stats

from common import calc, load

POP = lambda: load("svy").mil.values     # 정제본 566명
SD = lambda: POP().std(ddof=1)

# ── 자료 · 정해진 산술 ──
calc("b1.svy_n")(lambda: len(POP()))
calc("b1.mil_mean")(lambda: POP().mean())
calc("b1.mil_sd")(SD)
calc("b1.mil_skew")(lambda: float(stats.skew(POP())))
calc("b1.mean_minus_sd")(lambda: POP().mean() - SD())
calc("b1.mean_plus_sd")(lambda: POP().mean() + SD())
calc("b1.mean_minus_2sd")(lambda: POP().mean() - 2 * SD())
calc("b1.mean_plus_2sd")(lambda: POP().mean() + 2 * SD())
calc("b1.within_1sd_pct")(lambda: np.mean(np.abs(POP() - POP().mean()) <= SD()) * 100)
calc("b1.se_formula30")(lambda: SD() / np.sqrt(30))
for _k in (1, 2, 3):
    calc(f"b1.normal_{_k}sd_pct")(lambda _k=_k: (stats.norm.cdf(_k) - stats.norm.cdf(-_k)) * 100)
calc("b1.z975")(lambda: stats.norm.ppf(.975))
calc("b1.t975_df5")(lambda: stats.t.ppf(.975, 5))
calc("b1.t975_df200")(lambda: stats.t.ppf(.975, 200))
# 실습 1 「다른 씨앗이면 0.4~0.6」 = 100번 비율의 가운데 95% 범위(정규 근사 0.5 ± 1.96 × 0.05)
calc("b1.coin100_lo95")(lambda: .5 - stats.norm.ppf(.975) * np.sqrt(.25 / 100))
calc("b1.coin100_hi95")(lambda: .5 + stats.norm.ppf(.975) * np.sqrt(.25 / 100))
calc("b1.sqrt20")(lambda: np.sqrt(20))
calc("b1.frac100_pct")(lambda: 100 * 100 / len(POP()))     # 100명이 566명의 몇 %인가

# ── 모의시행 = 책 코드 순서 ──
_c: dict = {}


def sims():
    if _c:
        return _c
    g = np.random.default_rng(73)                         # §B1.1
    for n in (10, 100, 1000, 10000):
        _c[f"coin{n}"] = g.integers(0, 2, n).mean()
    g = np.random.default_rng(73)                         # §B1.2
    uni = g.uniform(0, 10, 5000)
    g.normal(5, 1.5, 5000)                                # 정규(책 코드 순서를 지키려고 뽑는다)
    e = g.exponential(5, 5000)
    _c.update(unif_sd=uni.std(ddof=1), exp_mean=e.mean(), exp_median=float(np.median(e)),
              exp_skew=float(stats.skew(e)))
    g = np.random.default_rng(73)                         # §B1.3
    z = g.normal(0, 1, 100000)
    for k in (1, 2, 3):
        _c[f"z{k}"] = float(np.mean(np.abs(z) <= k))
    pop = POP()                                           # §B1.4 표 = 사람 수마다 씨앗을 다시 정한다
    for seed in (73, 37):
        for n in (5, 30, 100):
            g = np.random.default_rng(seed)
            m = np.array([g.choice(pop, n, replace=False).mean() for _ in range(2000)])
            _c[f"sd{n}_{seed}"] = m.std(ddof=1)
            _c[f"sk{n}_{seed}"] = float(stats.skew(m))
    return _c


for _n in (10, 100, 10000):
    calc(f"b1.coin{_n}")(lambda _n=_n: sims()[f"coin{_n}"])
calc("b1.unif_sd")(lambda: sims()["unif_sd"])
calc("b1.exp_mean")(lambda: sims()["exp_mean"])
calc("b1.exp_median")(lambda: sims()["exp_median"])
calc("b1.exp_skew")(lambda: sims()["exp_skew"])
for _k in (1, 2, 3):
    calc(f"b1.z_within{_k}")(lambda _k=_k: sims()[f"z{_k}"])
for _n in (5, 30, 100):
    calc(f"b1.means_sd{_n}")(lambda _n=_n: sims()[f"sd{_n}_73"])
    calc(f"b1.means_skew{_n}")(lambda _n=_n: sims()[f"sk{_n}_73"])
    calc(f"b1.means_skew{_n}_seed37")(lambda _n=_n: sims()[f"sk{_n}_37"])
calc("b1.sd_ratio_5_100")(lambda: sims()["sd5_73"] / sims()["sd100_73"])   # 실습 3 : 유한모집단 보정이 든 실제 비
calc("b1.skew_seed_maxdiff")(lambda: max(abs(sims()[f"sk{n}_73"] - sims()[f"sk{n}_37"]) for n in (5, 30, 100)))
