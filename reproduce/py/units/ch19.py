# -*- coding: utf-8 -*-
"""19장 · 분석법 선택의 나무 — 파이썬 엔진."""
import numpy as np
from scipy import stats

from common import calc, load, ols

E = lambda: load("exp")          # 정제본 366명
_c: dict = {}


def _ab():
    e = E()
    return e[e.cond == 1].mil.values, e[e.cond == 0].mil.values


calc("ch19.exp_n")(lambda: len(E()))
calc("ch19.exp_df")(lambda: len(E()) - 2)
calc("ch19.reg_t")(lambda: (lambda r: r[0][1] / r[1][1])(ols(E().mil, [E().cond])))
calc("ch19.reg_p")(lambda: ols(E().mil, [E().cond])[2][1])
calc("ch19.f_two")(lambda: stats.f_oneway(*_ab()).statistic)
calc("ch19.f_two_p")(lambda: stats.f_oneway(*_ab()).pvalue)


def _perm():
    """책 §19.4 코드와 같은 난수 순서 (씨앗 73, 1만 번)."""
    if "perm" not in _c:
        a, b = _ab()
        rng = np.random.default_rng(73)
        pool = np.concatenate([a, b]); null = []
        for _ in range(10000):
            p = rng.permutation(pool)
            null.append(p[:len(a)].mean() - p[len(a):].mean())
        _c["perm"] = float(np.mean(np.abs(null) >= abs(a.mean() - b.mean())))
    return _c["perm"]


calc("ch19.perm_p")(_perm)


# 요인판 네 칸 = 칸 표시 변수 셋을 넣은 회귀 전체의 F (= B2 의 일원배치 F)
def _fac():
    f = load("fac")
    cell = f.elem * 2 + f.frame
    D = [(cell == k).astype(float).values for k in (1, 2, 3)]
    b, se, p, r2 = ols(f.mil_t2, D)
    n, k = len(f), 3
    return r2 / k / ((1 - r2) / (n - k - 1)), n - k - 1


calc("ch19.fac_F")(lambda: _fac()[0])
calc("ch19.fac_df2")(lambda: _fac()[1])


# 패널판 1차 · 3차
def _panel():
    if "pan" not in _c:
        w = load("panel").pivot(index="id", columns="wave", values="mil").dropna()
        x1, x3 = w[1].values, w[3].values
        d = x3 - x1
        sp = np.sqrt((x1.var(ddof=1) + x3.var(ddof=1)) / 2)
        _c["pan"] = dict(n=len(d), r=np.corrcoef(x1, x3)[0, 1],
                         t_pair=d.mean() / (d.std(ddof=1) / np.sqrt(len(d))),
                         t_ind=(x3.mean() - x1.mean()) / (sp * np.sqrt(2 / len(d))))
    return _c["pan"]


for _k in ("n", "r", "t_pair", "t_ind"):
    calc(f"ch19.pan_{_k}")(lambda k=_k: _panel()[k])
calc("ch19.z975")(lambda: stats.norm.ppf(.975))

# §19.6 오즈 산술
_odds = lambda p: p / (1 - p)
_p1 = lambda p0: _odds(p0) * 2 / (1 + _odds(p0) * 2)
calc("ch19.odds_p05")(lambda: _odds(.5))
calc("ch19.odds_p08")(lambda: _odds(.8))
for _p0 in (.01, .40, .50):
    calc(f"ch19.or2_ratio_{int(round(_p0 * 100)):02d}")(lambda p0=_p0: _p1(p0) / p0)
