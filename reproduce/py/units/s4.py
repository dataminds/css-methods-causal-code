# -*- coding: utf-8 -*-
"""S4 · 종단 패널(S4a)과 코호트(S4b) — 파이썬 엔진. 책 코드와 같은 계산."""
import numpy as np

from common import calc, cohen_d, load, ols

PAN = lambda: load("panel")
CO = lambda: load("cohort")


def _w(k):
    p = PAN()
    return p[p.wave == k].set_index("id")


def _wide():
    return PAN().pivot(index="id", columns="wave", values=["hjs", "mil"]).dropna()


def _z(v):
    return (v - v.mean()) / v.std(ddof=1)


def _clpm(y, y1, x1):
    w = _wide()
    b, se, p, _ = ols(_z(w[y]), [_z(w[y1]), _z(w[x1])])
    return float(b[2]), float(p[2])


_PATHS = {"hm12": (("mil", 2), ("mil", 1), ("hjs", 1)), "hm23": (("mil", 3), ("mil", 2), ("hjs", 2)),
          "mh12": (("hjs", 2), ("hjs", 1), ("mil", 1)), "mh23": (("hjs", 3), ("hjs", 2), ("mil", 2))}


def _c(d, v):
    return d[v] - d.groupby("id")[v].transform("mean")


def _stay_left():
    w1, w2 = _w(1), _w(2)
    return w1.loc[w1.index.intersection(w2.index)], w1.loc[w1.index.difference(w2.index)]


# S4a
for _k in (1, 2, 3):
    calc(f"s4.n_wave{_k}")(lambda k=_k: len(_w(k)))
calc("s4.n_waves")(lambda: PAN().wave.nunique())
calc("s4.n_complete")(lambda: len(_wide()))
calc("s4.stay_mil1")(lambda: _stay_left()[0].mil.mean())
calc("s4.left_mil1")(lambda: _stay_left()[1].mil.mean())
calc("s4.stay_left_d")(lambda: cohen_d(_stay_left()[0].mil, _stay_left()[1].mil))
calc("s4.r_w1_all")(lambda: _w(1).hjs.corr(_w(1).mil))
calc("s4.r_w1_complete")(lambda: _w(1).loc[_wide().index].hjs.corr(_w(1).loc[_wide().index].mil))
calc("s4.r_within")(lambda: np.corrcoef(_c(PAN(), "hjs"), _c(PAN(), "mil"))[0, 1])
calc("s4.r_within_hjs_only")(lambda: np.corrcoef(_c(PAN(), "hjs"), PAN().mil)[0, 1])
calc("s4.r_within_mil_only")(lambda: np.corrcoef(PAN().hjs, _c(PAN(), "mil"))[0, 1])
calc("s4.r_pooled_raw")(lambda: PAN().hjs.corr(PAN().mil))
for _lab, _args in _PATHS.items():
    calc(f"s4.beta_{_lab}")(lambda a=_args: _clpm(*a)[0])
    calc(f"s4.p_{_lab}")(lambda a=_args: _clpm(*a)[1])
    # 「p < .001」 문턱 주장 = 문턱 아래면 .001 을 돌려준다
    calc(f"s4.p_{_lab}_below_001")(lambda a=_args: 0.001 if _clpm(*a)[1] < .001 else -1.0)


def _beta420():
    w1, w2 = _w(1), _w(2)
    idx = w1.index.intersection(w2.index)
    b, *_ = ols(_z(w2.loc[idx].mil).values, [_z(w1.loc[idx].mil).values, _z(w1.loc[idx].hjs).values])
    return float(b[2])


calc("s4.beta_hm12_n420")(_beta420)
calc("s4.n_stay_w2")(lambda: len(_w(1).index.intersection(_w(2).index)))


def _change():
    w = _wide()
    dh = w[("hjs", 3)] - w[("hjs", 1)]
    dm = w[("mil", 3)] - w[("mil", 1)]
    b, se, p, _ = ols(dm.values, [dh.values])
    return float(b[1]), float(p[1])


calc("s4.change_b")(lambda: _change()[0])
calc("s4.change_p")(lambda: _change()[1])
# 논문 상자 산술 (Hamaker 외 2015 보고값)
calc("s4.hamaker_chi_diff")(lambda: 66.18 - 9.85)
calc("s4.hamaker_df_diff")(lambda: 11 - 8)

# S4b
calc("s4.cohort_n_per_year")(lambda: (lambda s: float(s.iloc[0]) if s.nunique() == 1 else -1.0)(CO().groupby("year").size()))
for _y in (2015, 2020, 2025):
    calc(f"s4.mil_{_y}")(lambda y=_y: CO()[CO().year == y].mil.mean())
    calc(f"s4.age_{_y}")(lambda y=_y: CO()[CO().year == y].age.mean())
calc("s4.hjs_cohort1950")(lambda: CO()[CO().cohort == 1950].hjs.mean())
calc("s4.hjs_cohort2000")(lambda: CO()[CO().cohort == 2000].hjs.mean())


def _age_slopes():
    return [ols(g.mil.values, [g.age.values])[0][1] * 10 for _, g in CO().groupby("year")]


calc("s4.age_slope10_min")(lambda: min(_age_slopes()))
calc("s4.age_slope10_max")(lambda: max(_age_slopes()))
