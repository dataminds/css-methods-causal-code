# -*- coding: utf-8 -*-
"""S5 · 2차 자료 자연실험 (단절 시계열) — 파이썬 엔진. 책 코드와 같은 계산 · 같은 난수 순서(씨앗 73)."""
import numpy as np

from common import calc, load, ols

TS = lambda: load("ts")


def _pre_post():
    ts = TS()
    return ts[ts.campaign == 0], ts[ts.campaign == 1]


def _full():
    ts = TS()
    return ols(ts.wellbeing, [ts.week, ts.campaign])


def _placebo(cut):
    pre = _pre_post()[0]
    b, se, p, _ = ols(pre.wellbeing, [(pre.week >= cut + 1).astype(float)])
    return b[1], p[1]


def _resid():
    ts = TS()
    b = _full()[0]
    X1 = np.column_stack([np.ones(len(ts)), ts.week, ts.campaign])
    fit = X1 @ b
    return X1, fit, ts.wellbeing.values - fit


calc("s5.n_weeks")(lambda: len(TS()))
calc("s5.first_campaign_week")(lambda: TS().week[TS().campaign == 1].min())
calc("s5.n_pre")(lambda: len(_pre_post()[0]))
calc("s5.n_post")(lambda: len(_pre_post()[1]))
calc("s5.pre_mean")(lambda: _pre_post()[0].wellbeing.mean())
calc("s5.post_mean")(lambda: _pre_post()[1].wellbeing.mean())
calc("s5.post_minus_pre")(lambda: _pre_post()[1].wellbeing.mean() - _pre_post()[0].wellbeing.mean())
calc("s5.b_week")(lambda: _full()[0][1])
calc("s5.p_week")(lambda: _full()[2][1])
calc("s5.b_campaign")(lambda: _full()[0][2])
# 「p < .001」 = 문턱 주장. 문턱 아래면 .001 을 돌려주고(표기 .001 과 맞는다), 아니면 −1 을 돌려줘 실패하게 한다 (R 의 NA 는 verify.py 의 결과 파서가 읽지 못한다).
calc("s5.p_campaign_below_001")(lambda: 0.001 if _full()[2][2] < 0.001 else -1.0)
calc("s5.se_campaign_naive")(lambda: _full()[1][2])
calc("s5.placebo13_b")(lambda: _placebo(13)[0])
calc("s5.placebo26_b")(lambda: _placebo(26)[0])
calc("s5.placebo26_p")(lambda: _placebo(26)[1])
calc("s5.placebo39_b")(lambda: _placebo(39)[0])
calc("s5.resid_autocorr")(lambda: np.corrcoef(_resid()[2][:-1], _resid()[2][1:])[0, 1])
calc("s5.nonresp_pct")(lambda: 100 - 37)                                  # 논문 응답률 37% 의 나머지
calc("s5.block_se_x196")(lambda: 0.084 * 1.96)                              # 책의 산술 그대로(인쇄된 0.084 × 1.96)


_cache = {}


def _sims():
    if _cache:
        return _cache
    ts = TS()
    # §⑧ 블록 다시 뽑기 = 책 코드 순서 (씨앗 73, 2000번, 여덟 주 토막 열셋)
    X1, fit, resid = _resid()
    rng = np.random.default_rng(73)
    boot = []
    for _ in range(2000):
        starts = rng.integers(0, len(ts) - 8, 13)
        e = np.concatenate([resid[s:s + 8] for s in starts])[:len(ts)]
        bb, *_ = np.linalg.lstsq(X1, fit + e, rcond=None)
        boot.append(bb[2])
    _cache["block_se"] = float(np.std(boot, ddof=1))
    # 실습 4 = wellbeing 을 뒤섞은 자료 (새 생성기, 씨앗 73)
    g = np.random.default_rng(73)
    b, _, p, _ = ols(g.permutation(ts.wellbeing.values), [ts.week, ts.campaign])
    _cache["perm_b2"], _cache["perm_p2"] = float(b[2]), float(p[2])
    # 18장 §18.3 추세 세계 (씨앗 73) = 사건 효과 0 · 주당 +0.012
    rng = np.random.default_rng(73)
    wk = np.arange(1, 105)
    camp = (wk >= 53).astype(int)
    y = 4.5 + 0.012 * wk + rng.normal(0, .25, 104)
    _cache["trend_diff"] = float(y[camp == 1].mean() - y[camp == 0].mean())
    return _cache


calc("s5.block_se")(lambda: _sims()["block_se"])
calc("s5.block_over_naive_pct")(lambda: (_sims()["block_se"] / _full()[1][2] - 1) * 100)
calc("s5.perm_b2")(lambda: _sims()["perm_b2"])
calc("s5.perm_p2")(lambda: _sims()["perm_p2"])
calc("s5.trend_world_diff")(lambda: _sims()["trend_diff"])
