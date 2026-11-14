# -*- coding: utf-8 -*-
"""8장 · 제3변수의 세 구조 — 파이썬 엔진. 수는 모두 책 코드와 같은 난수 순서로 낸다(씨앗 73)."""
import numpy as np
import pandas as pd

from common import calc

# §8.1 먼저 절 산술 = 보고된 b ÷ SE
calc("ch08.t_from_rounded")(lambda: .26 / .09)


def make_simpson(seed=73, n=400, sexeff=20.0):
    """§8.1 make_simpson · §8.6 make2 (sexeff = 20 이면 make_simpson 과 같다)."""
    rng = np.random.default_rng(seed)
    sex = rng.integers(0, 2, n)
    dose = np.clip(rng.normal(2 + 4 * sex, 1.5), 0, 10)
    vigor = 55 + sexeff * sex - 1.5 * dose + rng.normal(0, 4, n)
    return pd.DataFrame({"sex": sex, "dose": dose.round(1), "vigor": vigor.round(1)})


_c = {}


def zg():
    if "zg" not in _c:
        _c["zg"] = make_simpson()
    return _c["zg"]


slope = lambda d: np.polyfit(d.dose, d.vigor, 1)[0]
calc("ch08.sim_slope_all")(lambda: slope(zg()))
calc("ch08.sim_slope_f")(lambda: slope(zg()[zg().sex == 0]))
calc("ch08.sim_slope_m")(lambda: slope(zg()[zg().sex == 1]))
calc("ch08.sim_dose_f")(lambda: zg()[zg().sex == 0].dose.mean())
calc("ch08.sim_dose_m")(lambda: zg()[zg().sex == 1].dose.mean())
calc("ch08.sim_vigor_f")(lambda: zg()[zg().sex == 0].vigor.mean())
calc("ch08.sim_vigor_m")(lambda: zg()[zg().sex == 1].vigor.mean())
calc("ch08.sim_r_sex_dose")(lambda: np.corrcoef(zg().sex, zg().dose)[0, 1])
calc("ch08.sim_r_sex_vigor")(lambda: np.corrcoef(zg().sex, zg().vigor)[0, 1])


def _cell(sex, lo, hi):
    d = zg()
    return d[(d.sex == sex) & (d.dose > lo) & (d.dose <= hi)].vigor.mean()   # pd.cut 의 (lo, hi] 칸


calc("ch08.sim_cell_f_0_2")(lambda: _cell(0, 0, 2))
calc("ch08.sim_cell_f_6_8")(lambda: _cell(0, 6, 8))
calc("ch08.sim_cell_m_2_4")(lambda: _cell(1, 2, 4))
calc("ch08.sim_cell_m_8_10")(lambda: _cell(1, 8, 10))
calc("ch08.sim_slope_cut_leg")(lambda: slope(make_simpson(sexeff=0.0)))


# §8.2 충돌
def _collider():
    rng = np.random.default_rng(73)
    look, skill = rng.normal(0, 1, 2000), rng.normal(0, 1, 2000)
    win = (look + skill) > 1.8
    return np.corrcoef(look, skill)[0, 1], np.corrcoef(look[win], skill[win])[0, 1], int(win.sum())


calc("ch08.sim_coll_r_all")(lambda: _collider()[0])
calc("ch08.sim_coll_r_win")(lambda: _collider()[1])
calc("ch08.sim_coll_n_win")(lambda: _collider()[2])


# §8.4 세 세계 (상자 코드)
def _three():
    if "three" in _c:
        return _c["three"]
    g = np.random.default_rng(73); n = 2000
    C = g.normal(0, 1, n); X1 = .8 * C + g.normal(0, 1, n); Y1 = .8 * C + g.normal(0, 1, n)
    X2 = g.normal(0, 1, n); M2 = .8 * X2 + g.normal(0, 1, n); Y2 = .8 * M2 + g.normal(0, 1, n)
    X3 = g.normal(0, 1, n); Y3 = g.normal(0, 1, n); C3 = .8 * X3 + .8 * Y3 + g.normal(0, 1, n)

    def diff(X, Y, keep=None):
        m = np.median(X[keep]) if keep is not None else np.median(X)
        sel = np.ones(len(X), bool) if keep is None else keep
        return Y[sel & (X > m)].mean() - Y[sel & (X <= m)].mean()

    out = {}
    for name, (X, Y, Z) in {"conf": (X1, Y1, C), "chain": (X2, Y2, M2), "coll": (X3, Y3, C3)}.items():
        q = pd.qcut(Z, 10, labels=False)
        out[name] = (diff(X, Y), float(np.mean([diff(X, Y, q == k) for k in range(10)])))
    _c["three"] = out
    return out


for w in ("conf", "chain", "coll"):
    calc(f"ch08.sim_{w}_pre")(lambda w=w: _three()[w][0])
    calc(f"ch08.sim_{w}_post")(lambda w=w: _three()[w][1])


# §8.5 집계
def _agg():
    g = np.random.default_rng(73)
    ng, per = 30, 40
    grp = np.repeat(np.arange(ng), per)
    lvl = g.normal(0, 1, ng)[grp]
    e = g.normal(0, 1, ng * per)
    x = lvl + e
    y = lvl - 0.62 * e + g.normal(0, .6, ng * per)
    d = pd.DataFrame({"g": grp, "x": x, "y": y})
    gm = d.groupby("g").mean()
    wi = d.groupby("g")[["x", "y"]].transform(lambda v: v - v.mean())
    return d.x.corr(d.y), wi.x.corr(wi.y), gm.x.corr(gm.y)


calc("ch08.sim_agg_indiv")(lambda: _agg()[0])
calc("ch08.sim_agg_within")(lambda: _agg()[1])
calc("ch08.sim_agg_between")(lambda: _agg()[2])
