# -*- coding: utf-8 -*-
"""4장 · 측정 — 파이썬 엔진."""
import numpy as np

from common import calc, cronbach, load

S = lambda: load("svy")                  # 정제본 566명
RAW = lambda: load("svy", clean=False)
H = [f"hjs_{t}{i}" for t in "psqactl" for i in (1, 2, 3)]
EL = {t: [f"hjs_{t}{i}" for i in (1, 2, 3)] for t in "psqactl"}

calc("ch04.svy_n")(lambda: len(S()))
calc("ch04.svy_raw_n")(lambda: len(RAW()))

# 「먼저」 절 = Rogers 외 2023 연구 1 요소 알파 일곱 (p. 757)
_PA = [.86, .80, .90, .75, .82, .86, .89]
calc("ch04.paper_elem_alpha_mean")(lambda: float(np.mean(_PA)))
calc("ch04.paper_sb_pred")(lambda: 7 * np.mean(_PA) / (1 + 6 * np.mean(_PA)))

# §4.2 알파
calc("ch04.alpha_hjs21")(lambda: cronbach(S()[H]))
calc("ch04.alpha_hjs21_pct")(lambda: cronbach(S()[H]) * 100)
for _s, _k in (("mil", 4), ("flr", 8), ("swl", 5), ("gen", 3)):
    calc(f"ch04.alpha_{_s}")(lambda s=_s, k=_k: cronbach(S()[[f"{s}_{i}" for i in range(1, k + 1)]]))
for _t in "psqactl":
    calc(f"ch04.alpha_el_{_t}")(lambda t=_t: cronbach(S()[EL[t]]))
_el = lambda: [cronbach(S()[EL[t]]) for t in "psqactl"]
calc("ch04.alpha_el_min")(lambda: min(_el()))
calc("ch04.alpha_el_max")(lambda: max(_el()))
calc("ch04.alpha_el_mean")(lambda: float(np.mean(_el())))
calc("ch04.sb_pred")(lambda: 7 * np.mean(_el()) / (1 + 6 * np.mean(_el())))
calc("ch04.alpha_a_drop1")(lambda: cronbach(S()[["hjs_a2", "hjs_a3"]]))


def _itc(c, incl):
    d = S()
    tot = d[H].sum(axis=1)
    return d[c].corr(tot if incl else tot - d[c])


calc("ch04.itc_a1_incl")(lambda: _itc("hjs_a1", True))
calc("ch04.itc_a1_excl")(lambda: _itc("hjs_a1", False))
calc("ch04.itc_a1_inflation")(lambda: _itc("hjs_a1", True) - _itc("hjs_a1", False))
calc("ch04.itc_mean_inflation")(lambda: float(np.mean([_itc(c, True) - _itc(c, False) for c in H])))


def _item_r(same):
    R = S()[H].corr().to_numpy()
    f = np.array([c[4] for c in H])
    m = (f[:, None] == f[None, :]) if same else (f[:, None] != f[None, :])
    if same:
        m &= ~np.eye(len(H), dtype=bool)
    return R[m].mean()


calc("ch04.item_r_within")(lambda: _item_r(True))
calc("ch04.item_r_between")(lambda: _item_r(False))


# 역채점 미처리
def _rev():
    X = S()[H].copy()
    X["hjs_a1"] = 8 - X["hjs_a1"]
    return X


calc("ch04.alpha_rev21")(lambda: cronbach(_rev()))
calc("ch04.itc_rev_a1")(lambda: _rev().hjs_a1.corr(_rev().sum(axis=1) - _rev().hjs_a1))
calc("ch04.itc_rev_a2")(lambda: _rev().hjs_a2.corr(_rev().sum(axis=1) - _rev().hjs_a2))
calc("ch04.alpha_rev3")(lambda: cronbach(_rev()[EL["a"]]))


def _elem_r():
    sc = np.column_stack([S()[EL[t]].mean(axis=1) for t in "psqactl"])
    R = np.corrcoef(sc, rowvar=False)
    return R[np.triu_indices(7, 1)]


calc("ch04.elem_score_r_min")(lambda: _elem_r().min())
calc("ch04.elem_score_r_max")(lambda: _elem_r().max())
calc("ch04.r_legacy_gen")(lambda: np.corrcoef(S()[EL["l"]].mean(axis=1), S().gen)[0, 1])


# §4.2 · §4.3 반례 = 책 코드와 같은 난수 순서 (씨앗 73)
def _cx():
    g = np.random.default_rng(73)
    n = 566
    A = g.normal(0, 1, n)
    B = g.normal(0, 1, n)
    X = np.column_stack([A + g.normal(0, .5, n) for _ in range(3)] + [B + g.normal(0, .5, n) for _ in range(3)])
    ev = np.sort(np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)))[::-1]
    return cronbach(X), np.corrcoef(X[:, :3].mean(1), X[:, 3:].mean(1))[0, 1], ev


calc("ch04.cx_alpha")(lambda: _cx()[0])
calc("ch04.cx_r")(lambda: _cx()[1])
calc("ch04.cx_ev1")(lambda: _cx()[2][0])
calc("ch04.cx_ev2")(lambda: _cx()[2][1])
calc("ch04.cx_ev_rest_max")(lambda: _cx()[2][2:].max())
calc("ch04.cx_ev_rest_min")(lambda: _cx()[2][2:].min())


# §4.5 감쇠 = 책 코드 (씨앗 73 · 5만 명) ; true 를 바꾸면 실습 3
def _att(true, rel_want):
    rng = np.random.default_rng(73)
    n = 50000
    tx = rng.normal(0, 1, n)
    ty = true * tx + np.sqrt(1 - true ** 2) * rng.normal(0, 1, n)
    for rel in (1.0, .9, .8, .7, .6, .5):
        e = np.sqrt((1 - rel) / rel)
        ox, oy = tx + rng.normal(0, e, n), ty + rng.normal(0, e, n)
        if rel == rel_want:
            return np.corrcoef(ox, oy)[0, 1]


calc("ch04.att_obs_07")(lambda: _att(0.60, .7))
calc("ch04.att30_obs_07")(lambda: _att(0.30, .7))
calc("ch04.att_pred_07")(lambda: 0.60 * 0.7)
calc("ch04.att_back")(lambda: 0.42 / 0.7)


# §4.6 공통 방법 = 책 코드 (씨앗 73 · 500명)
def _cm():
    g = np.random.default_rng(73)
    n = 500
    A, B, M = g.normal(0, 1, n), g.normal(0, 1, n), g.normal(0, 1, n)
    out = {}
    for w in (0.0, 0.3, 0.5, 0.7):
        a = A + w * M + g.normal(0, .5, n)
        b = B + w * M + g.normal(0, .5, n)
        out[w] = np.corrcoef(a, b)[0, 1]
    return out


calc("ch04.cm_00")(lambda: _cm()[0.0])
calc("ch04.cm_05")(lambda: _cm()[0.5])
calc("ch04.cm_07")(lambda: _cm()[0.7])
