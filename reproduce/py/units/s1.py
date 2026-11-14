# -*- coding: utf-8 -*-
"""시나리오 S1 · 두 집단 실험 — 파이썬 엔진."""
import numpy as np
import pandas as pd
from scipy import stats

from common import calc, cohen_d, cronbach, load, ols

E = lambda: load("exp")                 # 정제본 366명
RAW = lambda: load("exp", clean=False)  # 원자료 380행
T = lambda: E()[E().cond == 1]
C = lambda: E()[E().cond == 0]

# ③ 정제 · 균형
calc("s1.raw_n")(lambda: len(RAW()))
calc("s1.raw_n_per_group")(lambda: int((RAW().cond == 1).sum()))
calc("s1.excluded")(lambda: len(RAW()) - len(E()))
calc("s1.n")(lambda: len(E()))
calc("s1.n_treat")(lambda: len(T()))
calc("s1.n_ctrl")(lambda: len(C()))
calc("s1.excl_treat")(lambda: int(((RAW().attn_1 == 0) & (RAW().cond == 1)).sum()))
calc("s1.excl_ctrl")(lambda: int(((RAW().attn_1 == 0) & (RAW().cond == 0)).sum()))

for v in ("age", "mil_t1", "hjs", "mil", "flr"):
    calc(f"s1.{v}_m_t")(lambda v=v: T()[v].mean())
    calc(f"s1.{v}_sd_t")(lambda v=v: T()[v].std(ddof=1))
    calc(f"s1.{v}_m_c")(lambda v=v: C()[v].mean())
    calc(f"s1.{v}_sd_c")(lambda v=v: C()[v].std(ddof=1))
    calc(f"s1.{v}_d")(lambda v=v: cohen_d(T()[v], C()[v]))

calc("s1.female_pct_t")(lambda: (T().gender == 2).mean() * 100)
calc("s1.female_pct_c")(lambda: (C().gender == 2).mean() * 100)
calc("s1.female_gap_pp")(lambda: ((C().gender == 2).mean() - (T().gender == 2).mean()) * 100)


def _chi():
    chi2, p, df, _ = stats.chi2_contingency(pd.crosstab(E().cond, E().gender))
    return chi2, p, df


calc("s1.chi_p")(lambda: _chi()[1])
calc("s1.chi_df")(lambda: _chi()[2])

# ④ 신뢰도
_items = lambda df: [c for c in df.columns if c.startswith("hjs_")]
calc("s1.hjs_items")(lambda: len(_items(E())))
calc("s1.hjs_alpha")(lambda: cronbach(E()[_items(E())]))
S = lambda: load("svy")
calc("s1.svy_mil_alpha")(lambda: cronbach(S()[[f"mil_{i}" for i in range(1, 5)]]))
calc("s1.svy_flr_alpha")(lambda: cronbach(S()[[f"flr_{i}" for i in range(1, 9)]]))

# ⑤ 치우침 · 천장
calc("s1.skew_min")(lambda: min(float(stats.skew(E()[v])) for v in ("mil_t1", "hjs", "mil", "flr")))
calc("s1.ceil_t_pct")(lambda: (T().mil >= 6.5).mean() * 100)
calc("s1.ceil_c_pct")(lambda: (C().mil >= 6.5).mean() * 100)
calc("s1.ceil_all_pct")(lambda: (E().mil >= 6.5).mean() * 100)
calc("s1.svy_ceil_pct")(lambda: (S().mil >= 6.5).mean() * 100)
calc("s1.mil_max")(lambda: E().mil.max())
calc("s1.mil_full_n")(lambda: int((E().mil == 7).sum()))

# ⑥ 조작 점검 · ⑦ 주 분석 (t 검정)
calc("s1.hjs_t")(lambda: stats.ttest_ind(T().hjs, C().hjs).statistic)
calc("s1.hjs_p")(lambda: stats.ttest_ind(T().hjs, C().hjs).pvalue)
calc("s1.obs_diff")(lambda: T().mil.mean() - C().mil.mean())
calc("s1.mil_t")(lambda: stats.ttest_ind(T().mil, C().mil).statistic)
calc("s1.mil_p")(lambda: stats.ttest_ind(T().mil, C().mil).pvalue)
calc("s1.t_df")(lambda: len(E()) - 2)
calc("s1.flr_p")(lambda: stats.ttest_ind(T().flr, C().flr).pvalue)

# ⑧ 회귀 (공변인 조정 · 조정 전 · 상호작용)
calc("s1.adj_b")(lambda: ols(E().mil, [E().cond, E().mil_t1])[0][1])
calc("s1.adj_se")(lambda: ols(E().mil, [E().cond, E().mil_t1])[1][1])
calc("s1.adj_p")(lambda: ols(E().mil, [E().cond, E().mil_t1])[2][1])
calc("s1.unadj_b")(lambda: ols(E().mil, [E().cond])[0][1])
calc("s1.unadj_se")(lambda: ols(E().mil, [E().cond])[1][1])


def _inter():
    k = E().mil_t1 - E().mil_t1.mean()
    b, _, p, _ = ols(E().mil, [E().cond, k, E().cond * k])
    return b[3], p[3]


calc("s1.inter_b")(lambda: _inter()[0])
calc("s1.inter_p")(lambda: _inter()[1])


# 검정력 (d = 0.22 · 집단당 190명 · 양측 .05) = 비중심 t 해석해
def _power():
    m, d = 190, 0.22
    df = 2 * m - 2
    c = stats.t.ppf(.975, df)
    nc = d * np.sqrt(m / 2)
    return 1 - stats.nct.cdf(c, df, nc) + stats.nct.cdf(-c, df, nc)


calc("s1.power")(_power)

# ⑦ 뒤섞기 · ⑧ 부트스트랩 = 책 코드와 같은 난수 순서 (블록마다 새 씨앗 73)
_cache = {}


def _perm():
    if "perm" not in _cache:
        exp = E()
        obs = T().mil.mean() - C().mil.mean()
        rng = np.random.default_rng(73)
        diffs = []
        for _ in range(5000):
            fake = rng.permutation(exp.cond.values)
            diffs.append(exp.mil.values[fake == 1].mean() - exp.mil.values[fake == 0].mean())
        _cache["perm"] = float((np.abs(np.array(diffs)) >= abs(obs)).mean())
    return _cache["perm"]


def _boot():
    if "boot" not in _cache:
        tv, cv = T().mil.values, C().mil.values
        rng = np.random.default_rng(73)
        bd = np.array([rng.choice(tv, len(tv), replace=True).mean()
                       - rng.choice(cv, len(cv), replace=True).mean() for _ in range(5000)])
        _cache["boot"] = np.percentile(bd, [2.5, 97.5])
    return _cache["boot"]


calc("s1.perm_p")(_perm)
calc("s1.perm_pct")(lambda: _perm() * 100)
calc("s1.boot_lo")(lambda: _boot()[0])
calc("s1.boot_hi")(lambda: _boot()[1])
