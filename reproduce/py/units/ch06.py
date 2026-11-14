# -*- coding: utf-8 -*-
"""6장 · 인과란 무엇인가 — 파이썬 엔진."""
import numpy as np
from scipy import stats

from common import calc, ols, load

S = lambda: load("svy")          # 정제본 566명

# §6.0 · §6.1 · §6.2 · §6.6 = 5장의 상관 .63 과 14장의 회귀 계수 0.98
calc("ch06.r_hjs_mil")(lambda: S().hjs.corr(S().mil))
calc("ch06.b_mil_on_hjs")(lambda: ols(S().mil.values, [S().hjs.values])[0][1])

# §6.1 두 방향 세계의 모집단 상관 = .6 / √(.6² + 1) (생성 규칙에서 바로 나오는 산술)
calc("ch06.two_world_rho")(lambda: .6 / np.sqrt(.6 ** 2 + 1))

# §6.2 갑돌이 예시의 산술 (먹은 세계 60 − 안 먹은 세계 63 · 55)
calc("ch06.gapdol_eff_63")(lambda: 60 - 63)
calc("ch06.gapdol_eff_55")(lambda: 60 - 55)


# §6.1 무관한 스무 쌍 · §6.1 두 방향 세계 · §6.2 저그 여덟 마리 = 책 코드와 같은 난수 순서
# (세 코드 블록이 각자 default_rng(73) 을 새로 만든다)
def _sims():
    out = {}
    rng = np.random.default_rng(73)
    hits, biggest = 0, 0
    for _ in range(20):
        a, b = rng.normal(0, 1, 30), rng.normal(0, 1, 30)
        r = np.corrcoef(a, b)[0, 1]
        t = abs(r) * np.sqrt(28 / (1 - r ** 2))
        hits += 2 * stats.t.sf(t, 28) < .05
        biggest = max(biggest, abs(r))
    out["hits"], out["biggest"] = int(hits), biggest
    rng = np.random.default_rng(73)
    A = rng.normal(0, 1, 500); B = .6 * A + rng.normal(0, 1, 500)
    B2 = rng.normal(0, 1, 500); A2 = .6 * B2 + rng.normal(0, 1, 500)
    out["r_ab"], out["r_ba"] = np.corrcoef(A, B)[0, 1], np.corrcoef(A2, B2)[0, 1]
    rng = np.random.default_rng(73)
    y0 = rng.normal(60, 8, 8).round(1)
    eff = rng.normal(-6, 3, 8).round(1)
    y1 = (y0 + eff).round(1)
    d4 = [round(y1[i] - y0[i], 1) for i in range(4)]
    out.update(y0_1=y0[0], y1_1=y1[0], eff4_max=max(d4), eff4_min=min(d4), eff_mean=eff.mean())
    return out


_cache = {}


def sims():
    if not _cache:
        _cache.update(_sims())
    return _cache


for k in ("hits", "biggest", "r_ab", "r_ba", "y0_1", "y1_1", "eff4_max", "eff4_min", "eff_mean"):
    calc(f"ch06.sim_{k}")(lambda k=k: sims()[k])
