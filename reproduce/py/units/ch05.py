# -*- coding: utf-8 -*-
"""5장 · 기술과 시각화 — 파이썬 엔진."""
import numpy as np

from common import calc, cohen_d, cronbach, load

S = lambda: load("svy")          # 정제본 566명
RAW = lambda: load("svy", clean=False)

for v in ("age", "hjs", "mil", "flr", "swl"):
    calc(f"ch05.{v}_mean")(lambda v=v: S()[v].mean())
    calc(f"ch05.{v}_sd")(lambda v=v: S()[v].std(ddof=1))
for v in ("hjs", "mil", "flr", "swl"):                    # 표 1 관측 범위 (2026-11-19 열 신설)
    calc(f"ch05.{v}_obs_min")(lambda v=v: S()[v].min())
    calc(f"ch05.{v}_obs_max")(lambda v=v: S()[v].max())
calc("ch05.flr_eq7_pct")(lambda: (S().flr == 7).mean() * 100)   # 실습 2 만점 비율
calc("ch05.age_min")(lambda: S().age.min())
calc("ch05.age_max")(lambda: S().age.max())
calc("ch05.svy_n")(lambda: len(S()))
calc("ch05.svy_excluded")(lambda: len(RAW()) - len(S()))
for code, nm in ((1, "men"), (2, "women"), (3, "other")):
    calc(f"ch05.{nm}_n")(lambda code=code: int((S().gender == code).sum()))
    calc(f"ch05.{nm}_pct")(lambda code=code: (S().gender == code).mean() * 100)
calc("ch05.men_mil_mean")(lambda: S()[S().gender == 1].mil.mean())
calc("ch05.men_mil_sd")(lambda: S()[S().gender == 1].mil.std(ddof=1))
calc("ch05.women_mil_mean")(lambda: S()[S().gender == 2].mil.mean())
calc("ch05.women_mil_sd")(lambda: S()[S().gender == 2].mil.std(ddof=1))
calc("ch05.men_minus_women_mil")(lambda: S()[S().gender == 1].mil.mean() - S()[S().gender == 2].mil.mean())
calc("ch05.mil_median")(lambda: S().mil.median())
calc("ch05.mil_ge65_pct")(lambda: (S().mil >= 6.5).mean() * 100)
calc("ch05.hjs_ge65_pct")(lambda: (S().hjs >= 6.5).mean() * 100)
calc("ch05.hjs_max")(lambda: S().hjs.max())
calc("ch05.flr_ge65_pct")(lambda: (S().flr >= 6.5).mean() * 100)
calc("ch05.flr_le15_pct")(lambda: (S().flr <= 1.5).mean() * 100)
calc("ch05.flr_max")(lambda: S().flr.max())
calc("ch05.hjs_mean_minus_sd")(lambda: S().hjs.mean() - S().hjs.std(ddof=1))
calc("ch05.hjs_mean_plus_sd")(lambda: S().hjs.mean() + S().hjs.std(ddof=1))
calc("ch05.age_mean_minus_sd")(lambda: S().age.mean() - S().age.std(ddof=1))
calc("ch05.age_mean_plus_sd")(lambda: S().age.mean() + S().age.std(ddof=1))
calc("ch05.raw_mil_mean")(lambda: RAW().mil.mean())
calc("ch05.clean_minus_raw_mil")(lambda: S().mil.mean() - RAW().mil.mean())
calc("ch05.mil_nunique")(lambda: S().mil.nunique())
calc("ch05.hjs_nunique")(lambda: S().hjs.nunique())
calc("ch05.raw_gender_mean")(lambda: RAW().gender.mean())
calc("ch05.mil_alpha")(lambda: cronbach(S()[[f"mil_{i}" for i in range(1, 5)]]))

# 2026-11-21 개념 절 결과 표 (코드는 복습 절로 옮김)
for v in ("hjs", "mil", "flr", "swl"):                     # §5.1 바닥 · 천장 표
    calc(f"ch05.tail_{v}_ge65")(lambda v=v: (S()[v] >= 6.5).mean() * 100)
    calc(f"ch05.tail_{v}_le15")(lambda v=v: (S()[v] <= 1.5).mean() * 100)
    calc(f"ch05.tail_{v}_max")(lambda v=v: S()[v].max())
for v in ("hjs", "mil", "age"):                            # §5.1 ±1SD 표
    calc(f"ch05.sd1_{v}_lo")(lambda v=v: S()[v].mean() - S()[v].std(ddof=1))
    calc(f"ch05.sd1_{v}_hi")(lambda v=v: S()[v].mean() + S()[v].std(ddof=1))
    calc(f"ch05.sd1_{v}_in")(lambda v=v: ((S()[v] > S()[v].mean() - S()[v].std(ddof=1))
                                          & (S()[v] < S()[v].mean() + S()[v].std(ddof=1))).mean() * 100)
calc("ch05.pts_item")(lambda: S().groupby(["mil_1", "mil_2"]).ngroups)    # §5.5 겹친 점 (문항 둘)
calc("ch05.pts_scale")(lambda: S().groupby(["hjs", "mil"]).ngroups)       # §5.5 척도 점수 둘
for v in ("hjs", "flr", "swl"):                            # §5.3 중앙값
    calc(f"ch05.med_{v}")(lambda v=v: S()[v].median())

# §5.4 간격 가정 계산 (자료 없이 정해진 산술)
_A = np.array([1] * 10 + [2] * 30 + [3] * 30 + [4] * 20 + [5] * 10)
_B = np.array([1] * 20 + [2] * 15 + [3] * 25 + [4] * 25 + [5] * 15)
_W = {1: 0, 2: 1, 3: 2, 4: 5, 5: 9}
calc("ch05.likert_equal_diff")(lambda: _A.mean() - _B.mean())
calc("ch05.likert_weighted_diff")(lambda: np.mean([_W[x] for x in _A]) - np.mean([_W[x] for x in _B]))
calc("ch05.likert_weighted_rescaled")(
    lambda: (np.mean([_W[x] for x in _A]) - np.mean([_W[x] for x in _B])) * 4 / 9)


# §5.1 · §5.2 모의시행 = 책 코드와 같은 난수 순서 (씨앗 73)
def _sims():
    rng = np.random.default_rng(73)
    n = 566
    fit = lambda x: (np.asarray(x, float) - np.mean(x)) / np.std(x, ddof=1) * 1.25 + 4.89
    shapes = [fit(rng.normal(0, 1, n)),
              fit(np.concatenate([rng.normal(-1, .35, n // 2), rng.normal(1, .35, n - n // 2)])),
              fit(rng.exponential(1, n)), fit(rng.uniform(0, 1, n))]
    out = {"medians": [float(np.median(v)) for v in shapes], "maxes": [v.max() for v in shapes],
           "skew": shapes[2]}
    for lab, sd in (("big", 1.25), ("small", 0.25)):
        a, b = rng.normal(5.0, sd, 300), rng.normal(4.6, sd, 300)
        out[lab] = (a.mean() - b.mean(), cohen_d(a, b), np.mean(b > a.mean()) * 100)
    return out


_cache = {}


def sims():
    if not _cache:
        _cache.update(_sims())
    return _cache


calc("ch05.sim_median_min")(lambda: min(sims()["medians"]))
calc("ch05.sim_median_max")(lambda: max(sims()["medians"]))
calc("ch05.sim_max_min")(lambda: min(sims()["maxes"]))
calc("ch05.sim_max_max")(lambda: max(sims()["maxes"]))
for _i in range(4):                                        # §5.1 네 분포 표 (종 · 두 봉우리 · 치우침 · 고른)
    calc(f"ch05.sim_med_{_i}")(lambda i=_i: sims()["medians"][i])
    calc(f"ch05.sim_mx_{_i}")(lambda i=_i: sims()["maxes"][i])
calc("ch05.sim_big_diff")(lambda: sims()["big"][0])
calc("ch05.sim_small_diff")(lambda: sims()["small"][0])
calc("ch05.sim_big_d")(lambda: sims()["big"][1])
calc("ch05.sim_small_d")(lambda: sims()["small"][1])
calc("ch05.sim_big_over_pct")(lambda: sims()["big"][2])
calc("ch05.sim_small_over_pct")(lambda: sims()["small"][2])


def _inside1sd(v):
    m, sd = v.mean(), v.std(ddof=1)
    return ((v > m - sd) & (v < m + sd)).mean() * 100


calc("ch05.sim_skew_mean")(lambda: sims()["skew"].mean())
calc("ch05.sim_skew_median")(lambda: float(np.median(sims()["skew"])))
calc("ch05.sim_skew_within1sd_pct")(lambda: _inside1sd(sims()["skew"]))

# 정규분포 ±1SD 이론 비율
from scipy import stats as _st  # noqa: E402
calc("ch05.normal_within1sd_pct")(lambda: (_st.norm.cdf(1) - _st.norm.cdf(-1)) * 100)

# 「먼저」 절 = 논문 값으로 한 산술
calc("ch05.paper_age_minus_sd")(lambda: 46.09 - 16.06)
calc("ch05.paper_age_plus_sd")(lambda: 46.09 + 16.06)

# §5.2 상관 표 (정제본)
for _a, _b in (("hjs", "mil"), ("hjs", "flr"), ("hjs", "swl"), ("mil", "flr"), ("mil", "swl"), ("flr", "swl")):
    calc(f"ch05.r_{_a}_{_b}")(lambda a=_a, b=_b: np.corrcoef(S()[a], S()[b])[0, 1])

# 실습 5 값의 가짓수 (정제본)
calc("ch05.gender_nunique")(lambda: S().gender.nunique())
calc("ch05.mil1_nunique")(lambda: S().mil_1.nunique())

# §5.4 논문 상자 = Cheung & Lucas (2014) 감쇠 교정 역산
calc("ch05.cl_reliab_product")(lambda: (.62 / .78) ** 2)
calc("ch05.cl_swls_alpha")(lambda: (.62 / .78) ** 2 / .72)

# §5.3 「남의 표 1 읽기」 = 실험판 집단별 표 (정제본 366명 ; cond 1 = 개입 · 0 = 통제)
E = lambda: load("exp")
for _c, _g in ((1, "int"), (0, "ctl")):
    for _v in ("age", "mil_t1", "hjs", "mil", "flr"):
        calc(f"ch05.exp_{_g}_{_v}_mean")(lambda c=_c, v=_v: E()[E().cond == c][v].mean())
        calc(f"ch05.exp_{_g}_{_v}_sd")(lambda c=_c, v=_v: E()[E().cond == c][v].std(ddof=1))
    calc(f"ch05.exp_{_g}_n")(lambda c=_c: int((E().cond == c).sum()))
    calc(f"ch05.exp_{_g}_women_n")(lambda c=_c: int((E()[E().cond == c].gender == 2).sum()))
    calc(f"ch05.exp_{_g}_women_pct")(lambda c=_c: (E()[E().cond == c].gender == 2).mean() * 100)
calc("ch05.exp_women_pct_gap")(lambda: ((E()[E().cond == 0].gender == 2).mean() - (E()[E().cond == 1].gender == 2).mean()) * 100)
calc("ch05.exp_raw_n")(lambda: len(load("exp", clean=False)))
calc("ch05.exp_n")(lambda: len(E()))
