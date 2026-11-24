# -*- coding: utf-8 -*-
"""B3 · 상관 — 파이썬 엔진.

2026-11-14 14장 「상관과 회귀」에서 갈라 나온 단위. 14장에서 옮긴 대조 줄은 계산 이름 ch14.* 를 그대로 쓰고
(py/units/ch14.py), 이 파일은 B3 에서 새로 쓴 문장의 수를 계산한다.
그림의 모의 수치는 materials/make_book_figures.py 의 fig_b3_grid · fig_b3_traps 와 같은 난수 순서를 따른다
(그림 파일을 쓰지 않으려고 생성기를 불러오지 않고 계산만 옮겼다)."""
import numpy as np
from scipy import stats

from common import calc, load

S = lambda: load("svy")                  # 정제본 566명
RAW = lambda: load("svy", clean=False)   # 정제 전 590명

_c: dict = {}

# ── 설문판 hjs · mil (§B3.0 · §B3.1 · §B3.4 · 실습 2 ~ 4) ─────────────
calc("b3.n_svy")(lambda: len(S()))
calc("b3.n_raw")(lambda: len(RAW()))
calc("b3.r")(lambda: np.corrcoef(S().hjs, S().mil)[0, 1])
calc("b3.r_raw")(lambda: np.corrcoef(RAW().hjs, RAW().mil)[0, 1])
calc("b3.r2")(lambda: np.corrcoef(S().hjs, S().mil)[0, 1] ** 2)
calc("b3.r2_pct")(lambda: np.corrcoef(S().hjs, S().mil)[0, 1] ** 2 * 100)
_hi = lambda: S()[S().hjs >= 5]
calc("b3.hjs5_n")(lambda: len(_hi()))
calc("b3.hjs5_r")(lambda: np.corrcoef(_hi().hjs, _hi().mil)[0, 1])
calc("b3.spearman")(lambda: stats.spearmanr(S().hjs, S().mil).statistic)

# ── §B3.2 다섯 사람 손계산 ─────────────────────────────────────────
_X5 = np.array([1, 2, 3, 4, 5], float)
_Y5 = np.array([1, 3, 5, 2, 9], float)
_prod = lambda: (_X5 - _X5.mean()) * (_Y5 - _Y5.mean())
calc("b3.five_xbar")(lambda: _X5.mean())
calc("b3.five_ybar")(lambda: _Y5.mean())
calc("b3.five_sumprod")(lambda: _prod().sum())
calc("b3.five_prod4")(lambda: _prod()[3])
calc("b3.five_prod5")(lambda: _prod()[4])
calc("b3.five_cov")(lambda: _prod().sum() / (len(_X5) - 1))
calc("b3.five_sdx")(lambda: _X5.std(ddof=1))
calc("b3.five_sdy")(lambda: _Y5.std(ddof=1))
calc("b3.five_sdprod")(lambda: _X5.std(ddof=1) * _Y5.std(ddof=1))
calc("b3.five_r")(lambda: _prod().sum() / (len(_X5) - 1) / (_X5.std(ddof=1) * _Y5.std(ddof=1)))

# ── §B3.4 r = .3 의 제곱 (산술) ────────────────────────────────────
calc("b3.r3_sq")(lambda: .3 ** 2)
calc("b3.r3_sq_pct")(lambda: .3 ** 2 * 100)
calc("b3.r3_wrong_pct")(lambda: .3 * 100)        # AI 가 틀리는 곳 = r 을 몫으로 읽은 값


# ── §B3.1 격자 = make_book_figures.fig_b3_grid (씨앗 73 · 칸마다 200명 · 표본 상관을 목표값에 정확히) ──
_TARGETS = [.9, .7, .5, .3, 0, -.3, -.7, -.9]


def _grid():
    if "grid" not in _c:
        rng = np.random.default_rng(73)
        got = {}
        for r in _TARGETS:                      # 칸 순서대로 난수를 쓴다 (_pair_exact_r)
            x = rng.normal(0, 1, 200); z = rng.normal(0, 1, 200)
            zx = (x - x.mean()) / x.std(ddof=1)
            zz = z - z.mean(); zz = zz - (zz @ zx) / (zx @ zx) * zx; zz = zz / zz.std(ddof=1)
            got[r] = np.corrcoef(zx, r * zx + np.sqrt(1 - r ** 2) * zz)[0, 1]
        _c["grid"] = got
    return _c["grid"]


for _r in (.9, .7, .5, .3):
    calc(f"b3.grid_r{int(round(_r * 100)):02d}")(lambda r=_r: _grid()[r])


# ── §B3.3 네 함정 = make_book_figures.fig_b3_traps (씨앗 73 ; 그림과 같은 순서로 난수를 쓴다) ──
def _traps():
    if "traps" not in _c:
        rng = np.random.default_rng(73)
        out = {}
        x = np.linspace(-3, 3, 200); y = x ** 2 + rng.normal(0, 1, 200)          # (가) U자
        out["u"] = np.corrcoef(x, y)[0, 1]
        x = rng.normal(0, 1, 30); y = rng.normal(0, 1, 30)                         # (나) 극단점 하나
        out["out_without"] = np.corrcoef(x, y)[0, 1]
        out["out_with"] = np.corrcoef(np.append(x, 6.0), np.append(y, 6.0))[0, 1]
        x = rng.normal(0, 1, 300)                                                  # (다) 범위 제한 (_pair_with_r .6)
        y = .6 * x + np.sqrt(1 - .6 ** 2) * rng.normal(0, 1, 300)
        keep = x > 0.8
        out["range_all"] = np.corrcoef(x, y)[0, 1]
        out["range_cut"] = np.corrcoef(x[keep], y[keep])[0, 1]
        out["range_n"] = float(keep.sum())
        xs, ys, gs = [], [], []                                                    # (라) 두 집단 섞임
        for k, (mx, my) in enumerate(((0, 0), (3, 4))):
            xa = rng.normal(mx, .8, 100)
            ya = my - 0.6 * (xa - mx) + rng.normal(0, .6, 100)
            xs.append(xa); ys.append(ya); gs.append(np.full(100, k))
        x, y, g = np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)
        out["mix_all"] = np.corrcoef(x, y)[0, 1]
        out["mix_g0"] = np.corrcoef(x[g == 0], y[g == 0])[0, 1]
        out["mix_g1"] = np.corrcoef(x[g == 1], y[g == 1])[0, 1]
        _c["traps"] = out
    return _c["traps"]


for _k in ("u", "out_without", "out_with", "range_all", "range_cut", "range_n", "mix_all", "mix_g0", "mix_g1"):
    calc(f"b3.trap_{_k}")(lambda k=_k: _traps()[k])


# §B3.4 「논문의 상관행렬 읽기」 = 설문판 정제본. 평균 · 표준편차 · 변수 쌍의 r 은 5장 계산(ch05.*)을 다시 쓴다.
from common import cronbach  # noqa: E402

_ITEMS = {"flr": [f"flr_{i}" for i in range(1, 9)], "swl": [f"swl_{i}" for i in range(1, 6)]}
for _v, _cols in _ITEMS.items():
    calc(f"b3.alpha_{_v}")(lambda cols=_cols: cronbach(S()[cols]))
for _v in ("hjs", "mil", "flr", "swl"):
    calc(f"b3.r_age_{_v}")(lambda v=_v: np.corrcoef(S().age, S()[v])[0, 1])
calc("b3.r_age_absmax")(lambda: max(abs(np.corrcoef(S().age, S()[v])[0, 1]) for v in ("hjs", "mil", "flr", "swl")))
# |r| 이 이 값을 넘으면 양측 p < .05 (df = n − 2)
calc("b3.r_crit05")(lambda: (lambda t, df: t / np.sqrt(t * t + df))(stats.t.ppf(.975, len(S()) - 2), len(S()) - 2))

# §B3.5 결과 보고 본보기 (2026-11-24) = r(N − 2) · 피셔 z 95% 구간 · p
_rz = lambda: np.arctanh(np.corrcoef(S().hjs, S().mil)[0, 1])
_rh = lambda: stats.norm.ppf(.975) / np.sqrt(len(S()) - 3)
calc("b3.r_df")(lambda: len(S()) - 2)
calc("b3.r_ci_lo")(lambda: np.tanh(_rz() - _rh()))
calc("b3.r_ci_hi")(lambda: np.tanh(_rz() + _rh()))
