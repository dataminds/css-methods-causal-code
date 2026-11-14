# -*- coding: utf-8 -*-
"""11장 · 우연은 시뮬레이션으로 — 파이썬 엔진 (책 코드와 같은 난수 순서 ; 씨앗 73)."""
import numpy as np
import pandas as pd
from scipy import stats

from common import calc, load

S = lambda: load("svy")                  # 정제본 566명
RAW = lambda: load("svy", clean=False)   # 정제 전 590명
POP = lambda: S().mil.values             # §11.2 「566명을 모집단으로 선언」
SD = lambda: POP().std(ddof=1)


def fpc(n):
    N = len(POP())
    return np.sqrt((N - n) / (N - 1))


# ── 자료로 정해지는 수 ──
calc("ch11.svy_n")(lambda: len(S()))
calc("ch11.raw_n")(lambda: len(RAW()))
calc("ch11.mil_mean")(lambda: POP().mean())
calc("ch11.mil_sd")(SD)
calc("ch11.r_hjs_mil")(lambda: S().hjs.corr(S().mil))
calc("ch11.men_minus_women")(lambda: S()[S().gender == 1].mil.mean() - S()[S().gender == 2].mil.mean())
calc("ch11.men_n")(lambda: int((S().gender == 1).sum()))
calc("ch11.women_n")(lambda: int((S().gender == 2).sum()))
calc("ch11.se_formula50")(lambda: SD() / np.sqrt(50))
calc("ch11.se_formula25")(lambda: SD() / np.sqrt(25))
calc("ch11.se_formula100")(lambda: SD() / np.sqrt(100))
calc("ch11.se_formula200")(lambda: SD() / np.sqrt(200))
calc("ch11.mean_minus_se")(lambda: POP().mean() - SD() / np.sqrt(50))
calc("ch11.mean_plus_se")(lambda: POP().mean() + SD() / np.sqrt(50))
calc("ch11.sqrt50")(lambda: np.sqrt(50))
calc("ch11.sqrt8")(lambda: np.sqrt(8))
calc("ch11.frac50_pct")(lambda: 50 / len(POP()) * 100)
calc("ch11.frac100_pct")(lambda: 100 / len(POP()) * 100)
calc("ch11.frac200_pct")(lambda: 200 / len(POP()) * 100)
calc("ch11.se_fpc50")(lambda: SD() / np.sqrt(50) * fpc(50))
calc("ch11.se_fpc100")(lambda: SD() / np.sqrt(100) * fpc(100))
calc("ch11.se_fpc200")(lambda: SD() / np.sqrt(200) * fpc(200))
calc("ch11.fpc50_shrink_pct")(lambda: (1 - fpc(50)) * 100)
calc("ch11.mil_skew")(lambda: float(stats.skew(POP())))
calc("ch11.normal_2sd_pct")(lambda: (stats.norm.cdf(2) - stats.norm.cdf(-2)) * 100)
calc("ch11.n_minus_1_of_1000")(lambda: 1000 - 1)
# 「먼저」 상자 ③ = Rogers et al. (2023) 연구 1 보고값(β = .74 · 자유도 590)으로 다시 계산
calc("ch11.rogers_se")(lambda: np.sqrt((1 - .74 ** 2) / 590))
# 실습 3 극단값 = 책 코드 그대로. 전원 5점이면 어떤 난수로 뽑아도 평균이 5 = 표집분포 SD 0 (그래서 data)
def _flat():
    flat = np.full(566, 5.0)
    rng = np.random.default_rng(73)
    return np.array([rng.choice(flat, size=50, replace=False).mean() for _ in range(1000)]).std(ddof=1)


calc("ch11.flat_sd")(_flat)


# ── 모의시행 = 책 코드와 같은 순서 ──
_c: dict = {}


def sims():
    if _c:
        return _c
    pop = POP()
    rng = np.random.default_rng(73)          # §11.2 둘째 블록 (첫 블록과 같은 씨앗 → 첫 세 평균이 같다)
    means = np.array([rng.choice(pop, size=50, replace=False).mean() for _ in range(1000)])
    _c["means"] = means
    for n in (25, 50, 100, 200):             # §11.4 본문 · 상자 = 사람 수마다 씨앗 73
        r = np.random.default_rng(73)
        _c[f"sim{n}"] = np.array([r.choice(pop, size=n, replace=False).mean() for _ in range(1000)]).std(ddof=1)
    r = np.random.default_rng(73)            # §11.5
    skew_pop = r.exponential(1.0, 20000)
    r2 = np.random.default_rng(73)
    sm = np.array([r2.choice(skew_pop, size=40, replace=False).mean() for _ in range(2000)])
    _c["skew_pop"] = pd.Series(skew_pop).skew()
    _c["skew_means"] = pd.Series(sm).skew()
    # 12장 실습 2 = 남녀 뒤섞기 (설문판 gender 1 · 2 ; 씨앗 73 · 5,000번)
    sub = S()[S().gender.isin([1, 2])]
    v, gnd = sub.mil.values, sub.gender.values
    obs = v[gnd == 1].mean() - v[gnd == 2].mean()
    rg = np.random.default_rng(73)
    d = []
    for _ in range(5000):
        s = rg.permutation(gnd)
        d.append(v[s == 1].mean() - v[s == 2].mean())
    _c["perm_p"] = float(np.mean(np.abs(np.array(d)) >= abs(obs)))
    # B1 §B1.1 동전 (10 · 100 · … 순서로 같은 발생기)
    g = np.random.default_rng(73)
    _c["coin10"] = g.integers(0, 2, 10).mean()
    _c["coin100"] = g.integers(0, 2, 100).mean()
    return _c


calc("ch11.sample1")(lambda: sims()["means"][0])
calc("ch11.sample2")(lambda: sims()["means"][1])
calc("ch11.sample3")(lambda: sims()["means"][2])
calc("ch11.sample2_minus_1")(lambda: sims()["means"][1] - sims()["means"][0])
calc("ch11.means_min")(lambda: sims()["means"].min())
calc("ch11.means_max")(lambda: sims()["means"].max())
calc("ch11.means_range")(lambda: sims()["means"].max() - sims()["means"].min())
calc("ch11.means_mean")(lambda: sims()["means"].mean())
calc("ch11.means_sd")(lambda: sims()["means"].std(ddof=1))
calc("ch11.means_2sd")(lambda: 2 * sims()["means"].std(ddof=1))
for _n in (25, 50, 100, 200):
    calc(f"ch11.sim_se{_n}")(lambda _n=_n: sims()[f"sim{_n}"])
calc("ch11.sim25_over_sim200")(lambda: sims()["sim25"] / sims()["sim200"])
calc("ch11.formula_over_sim200_pct")(lambda: (SD() / np.sqrt(200) / sims()["sim200"] - 1) * 100)
calc("ch11.clt_skew_pop")(lambda: sims()["skew_pop"])
calc("ch11.clt_skew_means")(lambda: sims()["skew_means"])
calc("ch11.perm_p_gender")(lambda: sims()["perm_p"])
calc("ch11.b1_coin100")(lambda: sims()["coin100"])
